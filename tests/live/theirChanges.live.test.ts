/**
 * Someone else's tracked changes in an imported Word file, read by real
 * models (#734): asked beside the Mind what the counterparty changed, a model
 * names the clause they deleted and the notice they changed, and who changed
 * it, in English and in Chinese. Each model runs only when its key is in the
 * environment, with the variables of ./assistantAnswers.live.test.ts:
 *
 *   INCARNAMIND_LIVE_ANTHROPIC_KEY   Claude (INCARNAMIND_LIVE_ANTHROPIC_MODEL, Haiku by default)
 *   INCARNAMIND_LIVE_OPENAI_KEY      OpenAI (INCARNAMIND_LIVE_OPENAI_MODEL, its quick-tasks model by default)
 *   INCARNAMIND_LIVE_OLLAMA_CLOUD_MODEL   an Ollama cloud model, e.g. gpt-oss:20b-cloud (sent to ollama.com)
 *
 * Set INCARNAMIND_LIVE_REPORT to a file to keep what each model wrote.
 */
import { appendFile } from "node:fs/promises";
import { describe, expect, onTestFinished, test } from "vitest";
import type { ChatProviderKind, Core } from "../../src/core";
import { catalogProvider } from "../../src/core/providers/catalog/providers";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import { createOllamaModels } from "../../src/core/providers/ollamaModels";
import { createTempDataFolder, startCore } from "../helpers/core";
import { paragraph, run, wordFile } from "../helpers/word";
import { importWord } from "../helpers/wordImports";

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

const PRIYA = 'w:author="Priya Nair" w:date="2026-10-02T09:00:00Z"';

/** A master services agreement's termination clause, as the counterparty's lawyer marked it up. */
const CONTRACT = wordFile({
  body: [
    paragraph("9. Termination", { style: "Heading1" }),
    paragraph(
      `${run("9.1 Either party may terminate this Agreement by giving the other ")}<w:del w:id="1" ${PRIYA}><w:r><w:delText xml:space="preserve">30</w:delText></w:r></w:del><w:ins w:id="2" ${PRIYA}>${run("90")}</w:ins>${run(" days' written notice.")}`,
    ),
    `<w:p><w:pPr><w:rPr><w:del w:id="3" ${PRIYA}/></w:rPr></w:pPr><w:del w:id="4" ${PRIYA}><w:r><w:delText xml:space="preserve">9.2 The Customer may terminate this Agreement for convenience at any time, without paying a termination fee.</w:delText></w:r></w:del></w:p>`,
    paragraph("9.3 On termination, the Supplier shall return all Customer Data within 30 days."),
  ].join(""),
});

const QUESTIONS = [
  {
    text: "What did the counterparty change in 9.2?",
    says: [/\b(delet|remov|struck|strik)/i, /9\.2/],
  },
  {
    text: "Who changed the notice period in 9.1, and to what?",
    says: [/Priya/, /\b90\b/],
  },
  { text: "对方在第 9 条改了什么？", says: [/9\.2|9\.1/, /删|90/] },
] as const;

async function ask(core: Core, text: string, mindId: string) {
  const result = await core.askInConversation({ conversationId: null, text, mindId });
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
  return answer;
}

describe.each(LIVE)("Someone's tracked changes, read live: $name", (live) => {
  test.skipIf(live.apiKey === undefined)(
    `${live.name}: asked what the counterparty changed, it names the deleted clause and the new notice, and who changed it`,
    { timeout: 300_000 },
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
      const { mindId } = await importWord(core, dataDir, CONTRACT, { name: "MSA redline.docx" });

      const report: string[] = [`## ${live.name} · ${live.model}`];
      onTestFinished(async () => {
        if (process.env.INCARNAMIND_LIVE_REPORT)
          await appendFile(process.env.INCARNAMIND_LIVE_REPORT, `${report.join("\n")}\n\n`);
      });
      for (const question of QUESTIONS) {
        const answer = await ask(core, question.text, mindId);
        report.push(
          `- "${question.text}": ${answer.status}`,
          `  > ${answer.text.replace(/\s+/g, " ")}`,
        );
        expect(answer.status).toBe("done");
        for (const said of question.says) expect(answer.text).toMatch(said);
      }
    },
  );
});
