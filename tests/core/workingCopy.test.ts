/**
 * A Mind imported from Word is the working copy of its original (#1036):
 * asked "What is this résumé about?" beside it, an Answer cites the original
 * though no word of the Question is in it. A short Document's whole text goes
 * with the Question; a long one's opening comes back for a search that finds
 * nothing. A Question about another Document searches it as before.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { estimateTokens } from "../../src/core/answers/context";
import { UP_FRONT_TOKEN_LIMIT } from "../../src/core/answers/documentText";
import { wordFileOf } from "../../src/shared/wordFiles";
import { askAssistant } from "../helpers/assistant";
import { shownPassages } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, waitForProcessing, writeSourceFile } from "../helpers/documents";
import { turnOnEmbeddings } from "../helpers/embedding";
import {
  type ModelCall,
  type ScriptedReply,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";
import { paragraph, wordFile } from "../helpers/word";
import { importWordMind } from "../helpers/wordMinds";

const RESUME = [
  paragraph("张伟", { style: "Title" }),
  paragraph("工作经历", { style: "Heading1" }),
  paragraph("2019 年至今在北京一家物流公司担任数据分析师，负责运输成本模型和周报。"),
  paragraph("教育背景", { style: "Heading1" }),
  paragraph("2015 年至 2019 年就读于武汉大学统计学专业，获学士学位。"),
];

/** A model that searches the Question's words first, as a small model does, then cites the first Passage it was given. */
function citingFirst(): { model: ReturnType<typeof scriptedModel>; prompts: string[] } {
  const prompts: string[] = [];
  const model = scriptedModel((call: ModelCall): ScriptedReply => {
    prompts.push(call.prompt);
    const searches = call.results.filter((result) => result.tool === "search_documents");
    if (searches.length === 0) {
      return { calls: [{ tool: "search_documents", input: { query: lastQuestion(call) } }] };
    }
    if (!call.results.some((result) => result.tool === "cite")) {
      const [first] = [
        ...shownPassages(searches.at(-1)?.text ?? ""),
        ...shownPassages(call.prompt),
      ];
      if (!first) return { text: "That Document isn't among your Documents." };
      const quote = first.text.split("\n").find((line) => line.trim().length > 8) ?? first.text;
      return {
        calls: [{ tool: "cite", input: { citations: [{ marker: 1, passage: first.id, quote }] } }],
      };
    }
    return { text: "It is a résumé [^1]." };
  });
  return { model, prompts };
}

const lastQuestion = (call: ModelCall) => call.prompt.split("\n").filter(Boolean).at(-1) ?? "";

async function setUp(body: readonly string[], { embeddings = false } = {}) {
  const { model, prompts } = citingFirst();
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  if (embeddings) await turnOnEmbeddings(core);
  const sources = await createTempDataFolder();
  await mkdir(sources, { recursive: true });
  const path = join(sources, "ZW-2026.docx");
  await writeFile(path, wordFile({ body: body.join("") }));
  const imported = await importWordMind(core, { path, origin: "own", addOriginal: true });
  const documentId = wordFileOf(imported.client.doc)?.documentId;
  if (!documentId) throw new Error("The original wasn't added.");
  await waitForProcessing(core, [documentId]);
  return { core, mindId: imported.mindId, documentId, prompts, sources };
}

describe("A Mind imported from Word, and its original", { timeout: 60_000 }, () => {
  // protects=asked about "this résumé" beside its imported Mind, in Chinese and English, with embeddings off and on, an Answer cites the original, found, though the Question shares no word with it, and the model is told the Mind is its working copy; fails_when=the Answer has no Citation, says the file isn't there, or the model isn't told; no_claim=what a real model writes.
  test.each([
    ["这份简历讲了什么？", false],
    ["What is this résumé about?", false],
    ["这份简历讲了什么？", true],
    ["What is this résumé about?", true],
  ] as const)("“%s” cites the original (embeddings on: %s)", async (text, embeddings) => {
    const { core, mindId, documentId, prompts } = await setUp(RESUME, { embeddings });

    const answer = await askAssistant(core, { text, mindId });

    expect(answer.text).toBe("It is a résumé [^1].");
    expect(answer.citations).toHaveLength(1);
    expect(answer.citations[0]).toMatchObject({ documentId, check: "found" });
    const first = prompts[0] ?? "";
    expect(first).toContain("<document-text>");
    expect(estimateTokens(first.slice(first.indexOf("<document-text>")))).toBeLessThan(
      UP_FRONT_TOKEN_LIMIT,
    );
    expect(answer.steps).toBeDefined();
  });

  // protects=a working copy's original too long to give whole still searches, and a search that finds nothing gets its opening Passages, which the Answer cites; fails_when=its whole text is sent, or the empty search ends the Answer with no Citation; no_claim=none beyond protects.
  test("too long to give whole, an empty search gets the original's opening", async () => {
    const long = Array.from({ length: 200 }, (_, at) =>
      paragraph(
        `第 ${at + 1} 项：负责华北区运输成本模型的第 ${at + 1} 个版本，并按周向管理层汇报结果与偏差原因。`,
      ),
    );
    const { core, mindId, documentId, prompts } = await setUp([...RESUME, ...long]);

    const answer = await askAssistant(core, { text: "What is this résumé about?", mindId });

    expect(prompts[0]).not.toContain("<document-text>");
    expect(answer.citations[0]).toMatchObject({ documentId, check: "found" });
  });

  // protects=beside the working copy, a Question about another Document finds and cites that one as before; fails_when=it cites the original, or finds nothing; no_claim=none beyond protects.
  test("a Question about another Document cites that Document", async () => {
    const { core, mindId, sources } = await setUp(RESUME);
    const lease = await writeSourceFile(
      sources,
      "Lease.md",
      `# Lease\n\n${Array.from({ length: 80 }, (_, at) => `Clause ${at + 1}: the rent is reviewed every year in March, and the tenant keeps the garden.`).join("\n\n")}\n`,
    );
    const [added] = await addAndProcess(core, [lease]);

    const answer = await askAssistant(core, {
      text: "When is the rent reviewed in March?",
      mindId,
    });

    expect(answer.citations[0]).toMatchObject({ documentId: added?.id, check: "found" });
  });
});
