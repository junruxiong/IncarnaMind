/**
 * Where an Answer in the assistant found what it says: the record its steps
 * line is read from. A small scope's Documents go with the Question as
 * Passages, so the model cites one with no search: the stored Answer has no
 * search call, its Citations are that Document's, and the Mind in focus is the
 * only place it shows as read.
 */
import { describe, expect, test } from "vitest";
import { askAssistant } from "../helpers/assistant";
import { shownPassages } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { heading, note, writeMind } from "../helpers/minds";
import { type ModelCall, scriptedModel, scriptedModels } from "../helpers/models";

const LEASE = [
  "租赁合同-2026",
  "租期：2026 年 3 月 1 日至 2029 年 2 月 28 日，共三年。",
  "月租金：人民币 4,800 元整。",
  "押金：人民币 9,600 元。",
].join("\n");
const CREDIT_FINDING = "申请人的息税折旧摊销前利润为 3.1 亿元，贷款协议规定杠杆比率不超过 3.5 倍。";
const CREDIT_REPORT = ["授信委员会报告（草稿）", CREDIT_FINDING].join("\n");
const QUESTION = "租赁合同-2026 里的租期、月租金和押金分别是多少？";
const RENT = "月租金：人民币 4,800 元整。";

const upFrontIn = (prompt: string) =>
  /<document-text>[\s\S]*?<\/document-text>/.exec(prompt)?.[0] ?? "";

/** A model that cites the lease's rent from the Passages given with the Question, and searches only when it has none. */
function readingWhatItIsGiven() {
  const searched: string[] = [];
  const model = scriptedModel((call: ModelCall) => {
    const lease = shownPassages(upFrontIn(call.prompt)).find((each) => each.text.includes(RENT));
    if (!lease) {
      searched.push(call.prompt);
      return { calls: [{ tool: "search_documents", input: { query: QUESTION } }] };
    }
    if (!call.results.some((result) => result.tool === "cite")) {
      return {
        calls: [
          { tool: "cite", input: { citations: [{ marker: 1, passage: lease.id, quote: RENT }] } },
        ],
      };
    }
    return { text: "月租金为人民币 4,800 元[^1]。" };
  });
  return { model, searched };
}

describe("An Answer from Documents given with the Question", { timeout: 60_000 }, () => {
  // protects=a Question about a short Document, asked beside another Mind, is answered from the Document's text given up front: its stored Answer has no search call, its Citation is the lease's and found, and it records the Mind in focus as read; fails_when=a search call is invented, the Citation points elsewhere, or the Mind isn't recorded; no_claim=how a real model chooses between a search and the text it was given.
  test("cites the Document with no search, and records only the Mind in focus as read", async () => {
    const { model, searched } = readingWhatItIsGiven();
    const core = startCore(await createTempDataFolder(), {
      createChatModel: scriptedModels(model).createChatModel,
    });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const sources = await createSourceFolder();
    const [report, lease] = await addAndProcess(core, [
      await writeSourceFile(sources, "授信委员会报告（草稿）.md", CREDIT_REPORT),
      await writeSourceFile(sources, "租赁合同-2026.md", LEASE),
    ]);
    if (!report || !lease) throw new Error("The Documents weren't added.");
    const mind = await core.createMind({ title: "授信委员会报告（草稿）" });
    const client = await connectToMind(core, mind.id);
    writeMind(client, [heading(1, "授信委员会报告（草稿）"), note(CREDIT_FINDING)]);
    await client.settled();

    const answer = await askAssistant(core, { text: QUESTION, mindId: mind.id });

    expect(searched).toEqual([]);
    expect(answer.toolCalls).toEqual([]);
    expect(answer.citations).toHaveLength(1);
    expect(answer.citations[0]).toMatchObject({ documentId: lease.id, check: "found" });
    expect(answer.read).toMatchObject({ mindId: mind.id, deleted: false });
  });
});
