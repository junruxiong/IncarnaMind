import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { COVERED_AT_ONCE, coverageOf, documentsAsked } from "../../src/core/answers/coverage";
import { askAssistant } from "../helpers/assistant";
import { createTempDataFolder, startCore } from "../helpers/core";
import { waitForProcessing } from "../helpers/documents";
import { type ScriptedReply, scriptedModel, scriptedModels } from "../helpers/models";

describe("What a Question asks of the Documents in scope", () => {
  // protects=a summary of the set, in English and Chinese, asks about each Document; fails_when=one reads as an ordinary Question; no_claim=every wording.
  test.each([
    "Write a summary page of these documents",
    "Summarise these papers",
    "summarise all the files",
    "Give me an overview of the 14 reports",
    "总结这些文档",
    "把这几篇论文概括一下",
    "写一份所有文件的摘要",
  ])("%s asks about each", (question) => {
    expect(documentsAsked(question)).toEqual({ each: true, main: null });
  });

  // protects=Questions about a topic, or one Document, don't; fails_when=one raises the search limit or caps the scope; no_claim=none beyond protects.
  test.each([
    "What do the documents say about leverage?",
    "Overview of what the documents say about leverage",
    "总结这些文档中关于杠杆的内容",
    "Summarise the 10-K",
    "这份文件讲了什么？",
    "Give me an overview of the sources",
    "总结一下这两个",
    "每个都总结一下",
  ])("%s doesn't", (question) => {
    expect(documentsAsked(question).each).toBe(false);
  });

  // protects=a request for the main few names how many; fails_when=the count is lost; no_claim=none beyond protects.
  test.each([
    ["Write a summary about the two main documents", 2],
    ["compare the 3 key papers", 3],
    ["summarise the main document", 1],
    ["总结两份主要文档", 2],
    ["比较最重要的三篇", 3],
  ])("%s asks for the main %i", (question, main) => {
    expect(documentsAsked(question).main).toBe(main);
  });

  // protects=only Document words ask for the main few; fails_when=a Question about something else cuts the search to a few Documents; no_claim=none beyond protects.
  test.each([
    "What are the two main sources of revenue?",
    "Summarise the 2 main ones",
    "List the three key articles of the treaty",
  ])("%s asks for no main few", (question) => {
    expect(documentsAsked(question).main).toBeNull();
  });

  const sized = (count: number) =>
    Array.from({ length: count }, (_, at) => ({
      id: `d${at}`,
      name: `Doc ${at}`,
      size: 100 - at,
    }));

  // protects=a summary of more than a few covers the largest and keeps the rest's ids; the main few are the largest; fewer covers all; fails_when=it covers all of 14, picks other than the largest, or caps 8; no_claim=none beyond protects.
  test("covers the largest of many, the main few, or all", () => {
    const many = coverageOf({ each: true, main: null }, sized(14));
    expect(many?.kind).toBe("largest");
    expect(many?.used.map((each) => each.id)).toEqual(
      sized(COVERED_AT_ONCE).map((each) => each.id),
    );
    expect(many?.rest).toEqual(["d8", "d9", "d10", "d11", "d12", "d13"]);
    expect(coverageOf({ each: false, main: 2 }, sized(8))).toEqual({
      kind: "main",
      used: [
        { id: "d0", name: "Doc 0" },
        { id: "d1", name: "Doc 1" },
      ],
      rest: ["d2", "d3", "d4", "d5", "d6", "d7"],
    });
    expect(coverageOf({ each: true, main: null }, sized(COVERED_AT_ONCE))).toBeNull();
    expect(coverageOf({ each: false, main: 2 }, sized(2))).toBeNull();
  });
});

describe("A summary of many Documents, through the core", { timeout: 60_000 }, () => {
  async function withDocuments(count: number) {
    const folder = await mkdtemp(join(tmpdir(), "coverage-"));
    const paths = await Promise.all(
      Array.from({ length: count }, async (_, at) => {
        const path = join(folder, `Report ${at + 1}.md`);
        const body = Array.from(
          { length: (count - at) * 4 },
          (_, line) => `Report ${at + 1}, finding ${line + 1}: the tide rose ${line} cm.`,
        ).join("\n\n");
        await writeFile(path, `# Report ${at + 1}\n\n${body}\n`);
        return path;
      }),
    );
    const calls: { system: string; results: string[] }[] = [];
    const model = scriptedModel((call): ScriptedReply => {
      calls.push({ system: call.system, results: call.results.map((each) => each.text) });
      if (call.prompt.includes("search Report 3 only") && call.results.length === 0) {
        return {
          calls: [{ tool: "search_documents", input: { query: "tide", document: "Report 3" } }],
        };
      }
      return { text: "Covered." };
    });
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir, { createChatModel: scriptedModels(model).createChatModel });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const group = await core.createLibraryGroup({ name: "Reports", description: "" });
    const added = await core.addDocuments(paths, { groupId: group.id });
    await waitForProcessing(
      core,
      added.documents.map((document) => document.id),
      30_000,
    );
    const mind = await core.createMind({ title: "Notes", folderId: group.id });
    const scope = { folderIds: [group.id], tagIds: [], documentIds: [] };
    return { core, mind, scope, calls };
  }

  // protects=a summary of 10 Documents covers the 8 largest, tells the model which, and records the 2 others to offer; fails_when=it covers all, the smaller ones, or forgets the rest; no_claim=what the model writes.
  test("of 10 covers the 8 largest and keeps the 2 others", async () => {
    const { core, mind, scope, calls } = await withDocuments(10);

    const answer = await askAssistant(core, {
      text: "Write a summary page of these documents",
      mindId: mind.id,
      scope,
    });

    expect(answer.steps?.coverage?.kind).toBe("largest");
    expect(answer.steps?.coverage?.used.map((each) => each.name)).toEqual(
      Array.from({ length: 8 }, (_, at) => `Report ${at + 1}`),
    );
    expect(answer.steps?.coverage?.rest).toHaveLength(2);
    expect(calls[0]?.system).toContain("This Answer covers the 8 largest of 10 Documents");
    expect(calls[0]?.system).not.toContain("Report 9");
  });

  // protects=a Question about a topic across many Documents searches them all, with no cap; fails_when=it covers only the largest 8; no_claim=what the model writes.
  test("“what the documents say about” a topic searches all 10", async () => {
    const { core, mind, scope, calls } = await withDocuments(10);

    const answer = await askAssistant(core, {
      text: "Overview of what the documents say about leverage",
      mindId: mind.id,
      scope,
    });

    expect(answer.steps?.coverage).toBeUndefined();
    expect(calls[0]?.system).not.toContain("This Answer covers the 8 largest");
  });

  // protects=the two main Documents are the two largest, named to the model, with the rest kept to offer; fails_when=the model is asked to choose, or the rest is lost; no_claim=none beyond protects.
  test("“the two main documents” are the two largest", async () => {
    const { core, mind, scope, calls } = await withDocuments(4);

    const answer = await askAssistant(core, {
      text: "Write a summary about the two main documents",
      mindId: mind.id,
      scope,
    });

    expect(answer.steps?.coverage).toMatchObject({
      kind: "main",
      used: [{ name: "Report 1" }, { name: "Report 2" }],
    });
    expect(answer.steps?.coverage?.rest).toHaveLength(2);
    expect(calls[0]?.system).toContain("IncarnaMind took “Report 1” and “Report 2”");
  });

  // protects=search_documents with a Document's name searches only that one, so a summary can reach each; fails_when=another Document's Passages come back; no_claim=none beyond protects.
  test("a search can name one Document", async () => {
    const { core, mind, scope, calls } = await withDocuments(4);

    await askAssistant(core, { text: "search Report 3 only", mindId: mind.id, scope });

    const found = calls.at(-1)?.results.at(-1) ?? "";
    const named = [...found.matchAll(/document="([^"]+)"/g)].map((match) => match[1]);
    expect(named.length).toBeGreaterThan(0);
    expect(new Set(named)).toEqual(new Set(["Report 3"]));
  });
});
