/**
 * The assistant changes Minds and makes them (#202), against real models:
 * what the User types decides whether it answers, changes the Mind in focus,
 * adds to a Mind it names, or makes one, in English and in Chinese. Each
 * model runs only when its key (or, for Ollama, its name) is in the
 * environment, and is skipped otherwise; the variables are those of
 * ./assistantAnswers.live.test.ts.
 *
 * Set INCARNAMIND_LIVE_REPORT to a file to keep, for each request, the Tools
 * the model called, what they changed, its reply, and its time and cost.
 */
import { appendFile } from "node:fs/promises";
import { describe, expect, onTestFinished, test } from "vitest";
import type { ChatProviderKind, ConversationAnswer, Core, SearchScope } from "../../src/core";
import { catalogProvider } from "../../src/core/providers/catalog/providers";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import { createOllamaModels } from "../../src/core/providers/ollamaModels";
import { pendingProposals } from "../../src/shared/proposals";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { createSourceFolder, waitForProcessing, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { heading, note, resolveChanges, shownWithChanges, writeMind } from "../helpers/minds";

const LIVE: { name: string; kind: ChatProviderKind; apiKey: string | undefined; model: string }[] =
  [
    {
      name: "Anthropic",
      kind: "anthropic",
      apiKey: process.env.INCARNAMIND_LIVE_ANTHROPIC_KEY,
      model:
        process.env.INCARNAMIND_LIVE_ANTHROPIC_MODEL ??
        (catalogProvider("anthropic")?.roles.quickTasks as string),
    },
    {
      name: "OpenAI",
      kind: "openai",
      apiKey: process.env.INCARNAMIND_LIVE_OPENAI_KEY,
      model:
        process.env.INCARNAMIND_LIVE_OPENAI_MODEL ??
        (catalogProvider("openai")?.roles.quickTasks as string),
    },
    {
      name: "Ollama cloud",
      kind: "ollama",
      apiKey: process.env.INCARNAMIND_LIVE_OLLAMA_CLOUD_MODEL ? "" : undefined,
      model: process.env.INCARNAMIND_LIVE_OLLAMA_CLOUD_MODEL ?? "",
    },
  ];

/** What the model should do with a request: answer, or change one of the Minds. */
type Expected = "answer" | "focused" | "other" | "new";

const LANGUAGES = [
  {
    language: "en",
    folder: "Harbour Roasters",
    documents: {
      "Board update.md": [
        "# Board update, September",
        "Subscribers rose to 9,420 in September, up from 8,100 in August.",
        "The decaf launch waits on a cash-flow forecast from the CFO.",
        "The courier contract renews in February; the current courier was late on 12% of deliveries.",
      ],
      "Customer interviews.md": [
        "# Customer interviews",
        "Most customers said the roast date on the bag is why they subscribe.",
        "Three customers cancelled after late deliveries in August.",
        "Customers asked for a decaf option more than for any other change.",
      ],
    },
    focused: "Steering summary",
    summary: [
      "The quarter went very well overall, and in many ways it was really the best quarter that the business has had so far, because subscribers grew strongly from August to September and most customers say that the roast date printed on the bag is the main reason why they keep subscribing month after month.",
      "Deliveries are still a real worry, though, because the courier was late on a noticeable share of orders and a few customers cancelled because of it, which matters a great deal since the courier contract comes up for renewal in February.",
    ],
    other: "Project plan",
    requests: [
      { text: "Shorten this summary", expected: "focused" },
      { text: "Add a risks section", expected: "focused" },
      { text: "Summarise these Documents into a new Mind", expected: "new" },
      {
        text: "Add today's decision to Project plan: we launch decaf in March",
        expected: "other",
      },
      { text: "What did customers say about deliveries?", expected: "answer" },
    ],
  },
  {
    language: "zh-CN",
    folder: "港湾咖啡",
    documents: {
      "董事会简报.md": [
        "# 九月董事会简报",
        "九月订阅用户增至 9420 人，八月为 8100 人。",
        "低因咖啡的上市要等财务总监的现金流预测。",
        "快递合同二月续签；现有快递有 12% 的订单迟到。",
      ],
      "客户访谈.md": [
        "# 客户访谈",
        "大多数客户说，包装上的烘焙日期是他们订阅的原因。",
        "八月有三位客户因为送货迟到而退订。",
        "客户最想要的改变是增加低因咖啡。",
      ],
    },
    focused: "指导委员会摘要",
    summary: [
      "这个季度总体上非常非常顺利，从很多方面来看，这其实是公司到目前为止最好的一个季度，因为从八月到九月订阅用户增长得很快，而且大多数客户都说，包装上印的烘焙日期是他们一个月又一个月继续订阅的主要原因。",
      "不过送货仍然是一个真正让人担心的问题，因为快递有相当一部分订单迟到，也有几位客户因此退订，而快递合同二月就要续签，所以这件事非常重要。",
    ],
    other: "项目计划",
    requests: [
      { text: "把这段摘要缩短", expected: "focused" },
      { text: "加一个风险部分", expected: "focused" },
      { text: "把这些文档总结成一篇新笔记", expected: "new" },
      { text: "把今天的决定加到“项目计划”里：三月推出低因咖啡", expected: "other" },
      { text: "客户对送货说了什么？", expected: "answer" },
    ],
  },
] as const satisfies readonly {
  language: string;
  folder: string;
  documents: Record<string, readonly string[]>;
  focused: string;
  summary: readonly string[];
  other: string;
  requests: readonly { text: string; expected: Expected }[];
}[];

/** Asks in a new conversation beside `mindId`, and waits until its Answer has ended. */
async function ask(core: Core, text: string, mindId: string, scope: SearchScope) {
  const started = Date.now();
  const result = await core.askInConversation({ text, mindId, scope });
  if (!result.asked) throw new Error(`Not asked: ${JSON.stringify(result)}`);
  await new Promise<void>((resolve) => {
    const stops = [
      core.on("answer.finished", ({ answerId }) => answerId === result.answerId && done()),
      core.on("answer.failed", ({ answerId }) => answerId === result.answerId && done()),
    ];
    function done() {
      for (const stop of stops) stop();
      resolve();
    }
  });
  const { messages } = await core.getConversation(result.conversation.id);
  const answer = messages.find((message) => message.id === result.answerId);
  if (answer?.kind !== "answer") throw new Error("No Answer.");
  return { answer, seconds: (Date.now() - started) / 1000 };
}

/** What the chat requests recorded since `after` (rowid) cost, and the last rowid. */
function spentSince(dataDir: string, after: number): { cost: number; last: number } {
  const rows = queryDatabase<{ rowid: number; cost: number | null }>(
    dataDir,
    "SELECT rowid, cost FROM model_requests WHERE rowid > ? ORDER BY rowid",
    [after],
  );
  return rows.reduce(
    (sum, row) => ({ cost: sum.cost + (row.cost ?? 0), last: Math.max(sum.last, row.rowid) }),
    { cost: 0, last: after },
  );
}

/** What an Answer did: answered, or changed which Mind. */
function didWhat(answer: ConversationAnswer, focusedId: string, otherId: string): Expected[] {
  const changes = answer.toolCalls.flatMap((call) => (call.mindChange ? [call.mindChange] : []));
  if (changes.length === 0) return ["answer"];
  return changes.map((change) =>
    change.created
      ? "new"
      : change.mindId === focusedId
        ? "focused"
        : change.mindId === otherId
          ? "other"
          : "answer",
  );
}

describe.each(LIVE)("The assistant changes Minds, live: $name", (live) => {
  test.skipIf(live.apiKey === undefined)(
    `${live.name}: answers a Question, changes the Mind in focus, adds to a Mind it names, and makes a new one, in English and Chinese`,
    { timeout: 1_800_000 },
    async () => {
      const dataDir = await createTempDataFolder();
      const core = startCore(dataDir, {
        createChatModel: createAiSdkChatModel,
        ...(live.kind === "ollama" && { ollamaModels: createOllamaModels() }),
      });
      core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
      await core.saveChatProvider({
        kind: live.kind,
        ...(live.apiKey && { apiKey: live.apiKey }),
        modelId: live.model,
      });

      const report: string[] = [`## ${live.name} · ${live.model}`];
      onTestFinished(async () => {
        if (process.env.INCARNAMIND_LIVE_REPORT)
          await appendFile(process.env.INCARNAMIND_LIVE_REPORT, `${report.join("\n")}\n\n`);
      });
      let seen = 0;
      const sources = await createSourceFolder();

      for (const each of LANGUAGES) {
        const folder = await core.createLibraryGroup({ name: each.folder, description: "" });
        const paths = await Promise.all(
          Object.entries(each.documents).map(([name, lines]) =>
            writeSourceFile(sources, name, lines.join("\n\n")),
          ),
        );
        const added = await core.addDocuments(paths, { groupId: folder.id });
        await waitForProcessing(
          core,
          added.documents.map((document) => document.id),
        );
        const scope = { folderIds: [folder.id], tagIds: [], documentIds: [] };
        const focused = await core.createMind({ title: each.focused, folderId: folder.id });
        const client = await connectToMind(core, focused.id);
        writeMind(client, [
          heading(2, each.language === "en" ? "Summary" : "摘要"),
          ...each.summary.map((text) => note(text)),
        ]);
        await client.settled();
        const other = await core.createMind({ title: each.other, folderId: folder.id });
        const otherClient: MindClient = await connectToMind(core, other.id);
        writeMind(otherClient, [
          heading(2, each.language === "en" ? "Decisions" : "决定"),
          note(each.language === "en" ? "Hire a designer." : "招一位设计师。"),
        ]);
        await otherClient.settled();

        for (const request of each.requests) {
          // Each request starts from Minds with nothing waiting in them.
          await resolveChanges(client, "reject");
          await resolveChanges(otherClient, "reject");
          const { answer, seconds } = await ask(core, request.text, focused.id, scope);
          const spent = spentSince(dataDir, seen);
          seen = spent.last;
          const did = didWhat(answer, focused.id, other.id);
          const calls = answer.toolCalls
            .map((call) =>
              call.status === "failed"
                ? `${call.tool} (failed: ${JSON.stringify(call.input).slice(0, 160)})`
                : call.tool,
            )
            .join(", ");
          await client.settled();
          await otherClient.settled();
          report.push(
            `- ${each.language} "${request.text}": expected ${request.expected}, did ${did.join(" + ")}; ${answer.status}, ${seconds.toFixed(1)} s, $${spent.cost.toFixed(4)}`,
            `  calls: ${calls || "none"}`,
            `  > ${answer.text.replace(/\s+/g, " ").slice(0, 400)}`,
            ...(did.includes("focused")
              ? [`  focused: ${shownWithChanges(client).join(" ⏎ ").slice(0, 600)}`]
              : []),
            ...(did.includes("other")
              ? [`  other: ${shownWithChanges(otherClient).join(" ⏎ ").slice(0, 400)}`]
              : []),
          );
          expect.soft(answer.status, request.text).toBe("done");
          expect.soft(did, request.text).toContain(request.expected);
          if (request.expected === "answer") {
            expect.soft(pendingProposals(client.blocks), request.text).toEqual([]);
          }
        }
      }
    },
  );
});
