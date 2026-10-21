import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { Document } from "../../src/core";
import {
  documentMeant,
  type InView,
  type NamedDocument,
} from "../../src/core/answers/documentMeant";
import { askAssistant, changesIn } from "../helpers/assistant";
import { setUpWithDocuments, shownPassages } from "../helpers/citations";
import { writeMind } from "../helpers/minds";
import { type ModelCall, scriptedModel } from "../helpers/models";

/**
 * "This document", "this one", "这份", "这个" mean one Document (#1334), the same way in English
 * and Chinese: the one the User has in view, or the one whose name follows "this", or the User is
 * asked which. The model is told; the search still reaches every Document, so a comparison keeps
 * its other side.
 */

const ZH: NamedDocument = { id: "zh", name: "劳动合同" };
const EN: NamedDocument = { id: "en", name: "Employment agreement" };
const NDA: NamedDocument = { id: "nda", name: "Mutual NDA" };
const CONTRACTS = [ZH, EN, NDA];
const LEASES: NamedDocument[] = [
  { id: "l1", name: "Lease 1" },
  { id: "l2", name: "Lease 2" },
];
const NOTHING: InView = { open: null, selected: null, original: null, mind: [], lastAnswer: [] };

const meant = (
  question: string,
  documents: readonly NamedDocument[],
  inView: Partial<InView> = {},
  folder = false,
) =>
  documentMeant(question, {
    documents: () => documents,
    inView: { ...NOTHING, ...inView },
    folder,
  });

describe("What “this” means", () => {
  // protects=the words after "this" that a Document's name holds pick it, in English and Chinese, whatever language the others are in; fails_when=another Document is picked, or the User is asked; no_claim=names that share none of the words.
  test.each([
    ["请为这份劳动合同写一个简短的摘要。", ZH],
    ["这份合同里的年假是几天？", ZH],
    ["Write a short summary of this agreement.", EN],
    ["What does this employment agreement say about notice?", EN],
    ["Compare this agreement with the NDA", EN],
  ])("%s means %o", (question, document) => {
    expect(meant(question, CONTRACTS)).toEqual({ kind: "one", document });
  });

  // protects=English and its Chinese twin mean the same Document in view, whatever noun follows "this" or none: the open Document comes first; fails_when=one language resolves and its twin doesn't; no_claim=pronouns outside these.
  test.each([
    ["Summarise this document", "总结这份文件"],
    ["Summarise this one", "总结一下这个"],
    ["Summarise this", "总结一下这份"],
    ["Summarise this statement", "总结这份报表"],
    ["What is the total on this invoice?", "这张发票的总额是多少？"],
    ["Summarise this brief", "总结这份诉状"],
    ["Summarise this spreadsheet", "总结这张表"],
    ["What does this budget say?", "该预算说了什么？"],
    ["When does this end?", "此文件什么时候结束？"],
  ])("“%s” and “%s” mean the open Document", (english, chinese) => {
    for (const question of [english, chinese]) {
      expect(meant(question, CONTRACTS, { open: "nda", lastAnswer: ["en"] }), question).toEqual({
        kind: "one",
        document: NDA,
      });
    }
  });

  // protects=with nothing open, what is in view decides in order: the selected text's Citation, the working copy's own Document, the Mind's Citation, the last Answer's; fails_when=a later source wins, or the User is asked; no_claim=none beyond protects.
  test("with nothing open: the selection, the working copy, the Mind, the last Answer", () => {
    const all = { selected: ["l1"], original: "l2", mind: ["l2"], lastAnswer: ["l2"] };
    expect(meant("Summarise this lease", LEASES, all)).toEqual({
      kind: "one",
      document: LEASES[0],
    });
    expect(meant("总结这份租约", LEASES, { ...all, selected: null })).toEqual({
      kind: "one",
      document: LEASES[1],
    });
    expect(meant("Summarise this lease", LEASES, { mind: ["l2"] })).toEqual({
      kind: "one",
      document: LEASES[1],
    });
    expect(meant("总结这个", LEASES, { lastAnswer: ["l1"] })).toEqual({
      kind: "one",
      document: LEASES[0],
    });
  });

  // protects=two that fit the words equally ask which, unless something in view is one of them; a Folder scope means its Documents, so nothing is asked there; fails_when=one is picked unasked, or a Folder's Question asks which; no_claim=none beyond protects.
  test("two that fit equally: which, unless one is in view; nothing to ask in a Folder", () => {
    expect(meant("Summarise this lease", LEASES)).toEqual({ kind: "which", documents: LEASES });
    expect(meant("总结这份租约", LEASES, {}, true)).toBeNull();
    expect(meant("Summarise this lease", LEASES, { lastAnswer: ["l2"] })).toEqual({
      kind: "one",
      document: LEASES[1],
    });
  });

  // protects=with nothing in view and no name to go by, the User is asked which, in both languages, but not in a Folder, which "this" leaves as its Documents; fails_when=a Document is picked unasked, or a Folder's Question asks; no_claim=none beyond protects.
  test("nothing in view and no name: which, but not in a Folder", () => {
    for (const question of ["Summarise this one", "总结一下这个", "Summarise this document."]) {
      expect(meant(question, [EN, NDA]), question).toEqual({ kind: "which", documents: [] });
      expect(meant(question, [EN, NDA], {}, true), question).toBeNull();
    }
  });

  // protects=the only Document in scope is the one meant; fails_when=the User is asked which; no_claim=none beyond protects.
  test("the only Document in scope is the one", () => {
    expect(meant("这份文件讲了什么？", [EN])).toEqual({ kind: "one", document: EN });
    expect(meant("What is this file about?", [EN])).toEqual({ kind: "one", document: EN });
  });

  // protects="this" that means a folder, the User's own text, the chat or a time, or selected text that cites nothing, means no Document, in both languages; "应该" and "因此" say no "this"; fails_when=one of them narrows to a Document or asks which; no_claim=every wording.
  test.each([
    "Summarise the documents in this folder",
    "总结这个文件夹里的文档",
    "这个文件夹里的租约通知期是多久？",
    "Make this paragraph shorter",
    "写一份这篇笔记的总结",
    "这个问题怎么解决？",
    "What changed this year?",
    "这个月的收入是多少？",
    "我应该先读哪份？",
    "因此结论是什么？",
    "Summarise the agreements",
    "总结这些文档",
  ])("%s means no Document", (question) => {
    expect(meant(question, CONTRACTS, { lastAnswer: ["en"] })).toBeNull();
  });

  // protects=with text selected that cites no Document, "this" is that text; fails_when=a Document is meant instead; no_claim=none beyond protects.
  test("selected text with no Citation is what “this” means", () => {
    expect(meant("Summarise this", CONTRACTS, { selected: [], open: "en" })).toBeNull();
  });
});

const FILES = [
  {
    name: "劳动合同.md",
    contents:
      "# 劳动合同书\n\n乙方每年享有带薪年休假十天。任何一方提前三十日书面通知对方，可以解除本合同。\n",
  },
  {
    name: "Employment agreement.md",
    contents:
      "# Employment agreement\n\nTwenty-five days of paid leave a year. Either party may end this agreement with one month's written notice.\n",
  },
  {
    name: "Mutual NDA.md",
    contents: "# Mutual NDA\n\nEither party may end this NDA with thirty days' written notice.\n",
  },
];

/** Searches for the notice, then writes what it found into the Mind with add_to_mind. */
function writer(calls: ModelCall[]) {
  return scriptedModel((call) => {
    if (!call.tools.includes("add_to_mind")) return { text: "Done." };
    calls.push(call);
    if (!call.results.some((each) => each.tool === "search_documents")) {
      return { calls: [{ tool: "search_documents", input: { query: "notice 通知 解除" } }] };
    }
    if (!call.results.some((each) => each.tool === "add_to_mind")) {
      return {
        calls: [{ tool: "add_to_mind", input: { content: "## Summary\n\nLeave and notice." } }],
      };
    }
    return { text: "Done." };
  });
}

const searched = (calls: readonly ModelCall[]) =>
  new Set(
    calls.flatMap((call) =>
      call.results
        .filter((each) => each.tool === "search_documents")
        .flatMap((each) => shownPassages(each.text).map((passage) => passage.document)),
    ),
  );

/** The Documents the Question was given whole, by name. */
const givenWhole = (call: ModelCall | undefined) =>
  shownPassages(/<document-text>[\s\S]*?<\/document-text>/.exec(call?.prompt ?? "")?.[0] ?? "").map(
    (passage) => passage.document,
  );

const cites = (document: Document): JSONContent => ({
  type: "paragraph",
  content: [
    { type: "text", text: "The notice period is one month." },
    {
      type: "citation",
      attrs: {
        passageId: "passage-1",
        documentId: document.id,
        documentName: document.name,
        contentHash: document.contentHash,
        pageFrom: null,
        pageTo: null,
        quote: "one month's written notice",
        check: "found",
        checkReason: null,
      },
    },
  ],
});

describe("“This” through the core, as the assistant asks", { timeout: 60_000 }, () => {
  // protects=the Chinese request and its English twin are told which contract "this" is, given it whole, told to keep to five lines, and write into the empty Mind; fails_when=the model isn't told, is given the other contract, or the Chinese one stays in the panel; no_claim=that a real model keeps to it.
  test.each([
    ["请为这份劳动合同写一个简短的摘要。", "劳动合同"],
    ["Write a short summary of this agreement.", "Employment agreement"],
  ])("“%s” means “%s”, given whole, written into the empty Mind", async (request, name) => {
    const calls: ModelCall[] = [];
    const { core, mind } = await setUpWithDocuments(writer(calls), FILES);

    const answer = await askAssistant(core, { text: request, mindId: mind.id });

    const system = calls[0]?.system ?? "";
    expect(system).toContain(`By "this", the User means the Document "${name}"`);
    expect(system).toContain("five lines at most");
    expect(system).toContain("write it there with add_to_mind");
    expect(new Set(givenWhole(calls[0]))).toEqual(new Set([name]));
    expect(changesIn(answer).map((change) => [change.mindId, change.created])).toEqual([
      [mind.id, false],
    ]);
  });

  // protects=a comparison keeps its other side: "this agreement" is named to the model while the search still reaches the NDA; fails_when=the search is cut to the agreement; no_claim=what the model compares.
  test("“Compare this agreement with the NDA” still searches the NDA", async () => {
    const calls: ModelCall[] = [];
    const { core, mind } = await setUpWithDocuments(writer(calls), FILES);

    await askAssistant(core, { text: "Compare this agreement with the NDA", mindId: mind.id });

    expect(calls[0]?.system).toContain('the User means the Document "Employment agreement"');
    expect(searched(calls)).toContain("Mutual NDA");
  });

  // protects=the Document open in the focused group, and the one the Mind cites, are what "this" means, the same in English and Chinese; fails_when=the User is asked which, or the last-cited one wins over the open one; no_claim=none beyond protects.
  test("the open Document, else the one the Mind cites, is what “this” means", async () => {
    const calls: ModelCall[] = [];
    const { core, mind, client, documents } = await setUpWithDocuments(writer(calls), FILES);
    const nda = documents.find((each) => each.name === "Mutual NDA");
    const agreement = documents.find((each) => each.name === "Employment agreement");
    if (!nda || !agreement) throw new Error("No Documents.");

    await askAssistant(core, {
      text: "Summarise this document.",
      mindId: mind.id,
      viewing: nda.id,
    });
    expect(calls.at(-1)?.system).toContain('the User means the Document "Mutual NDA"');

    writeMind(client, [cites(agreement)]);
    await client.settled();
    await askAssistant(core, { text: "总结一下这份文件。", mindId: mind.id });
    expect(calls.at(-1)?.system).toContain('the User means the Document "Employment agreement"');
  });

  // protects=a Chinese Question about the Folder in scope means its Documents: no Document is named and nobody is asked which; fails_when=it reads as one Document; no_claim=none beyond protects.
  test("“总结这个文件夹里的文档” in a Folder is about the Folder", async () => {
    const calls: ModelCall[] = [];
    const { core, mind, documents } = await setUpWithDocuments(writer(calls), FILES);
    const group = await core.createLibraryGroup({ name: "Contracts", description: "" });
    for (const document of documents) await core.assignDocumentGroup(document.id, group.id);

    await askAssistant(core, {
      text: "总结这个文件夹里的文档",
      mindId: mind.id,
      scope: { folderIds: [group.id], tagIds: [], documentIds: [] },
    });

    const system = calls[0]?.system ?? "";
    expect(system).not.toContain('By "this"');
  });

  // protects=when two Documents fit "this lease" equally and nothing is in view, the model is told to ask which and nothing is owed to the Mind; fails_when=the request still routes as a write, or the model is asked again to write; no_claim=the wording of a real model's question.
  test("two that fit equally: the model asks which, and is not asked again to write", async () => {
    const calls: ModelCall[] = [];
    const model = scriptedModel((call) => {
      if (call.tools.includes("add_to_mind")) calls.push(call);
      return { text: "Which lease: Lease 1 or Lease 2?" };
    });
    const { core, mind } = await setUpWithDocuments(model, [
      { name: "Lease 1.md", contents: "# Lease 1\n\nThe rent is 1,000 euros.\n" },
      { name: "Lease 2.md", contents: "# Lease 2\n\nThe rent is 1,100 euros.\n" },
    ]);

    const answer = await askAssistant(core, {
      text: "Write a short summary of this lease.",
      mindId: mind.id,
    });

    expect(calls).toHaveLength(1);
    expect(calls[0]?.system).toContain("Ask them which they mean, in one line");
    expect(calls[0]?.system).not.toContain("write it there with add_to_mind");
    expect(changesIn(answer)).toEqual([]);
  });
});
