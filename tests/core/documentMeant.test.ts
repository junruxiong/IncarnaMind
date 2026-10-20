import { describe, expect, test } from "vitest";
import { documentMeant, type NamedDocument } from "../../src/core/answers/documentMeant";
import { askAssistant, changesIn } from "../helpers/assistant";
import { setUpWithDocuments, shownPassages } from "../helpers/citations";
import { type ModelCall, scriptedModel } from "../helpers/models";

/**
 * "This document", "this one", "这份劳动合同" mean one Document (#1334): the request is answered,
 * or written, from that one alone, in English and Chinese alike, or the User is asked which.
 */

const ZH: NamedDocument = { id: "zh", name: "劳动合同" };
const EN: NamedDocument = { id: "en", name: "Employment agreement" };
const CONTRACTS = [ZH, EN];
const LEASES: NamedDocument[] = [
  { id: "l1", name: "Lease 1" },
  { id: "l2", name: "Lease 2" },
];

const meant = (
  question: string,
  documents: readonly NamedDocument[],
  { cited = [] as string[], original = null as NamedDocument | null } = {},
) => documentMeant(question, { documents: () => documents, original, cited });

describe("The one Document a request points at", () => {
  // protects=a request that points with the words of one Document's name means that one, in English and Chinese, whatever language the other is in; fails_when=it means both, or the other; no_claim=names that share none of the words.
  test.each([
    ["请为这份劳动合同写一个简短的摘要。", ZH],
    ["这份合同里的年假是几天？", ZH],
    ["Write a short summary of this agreement.", EN],
    ["What does this employment agreement say about notice?", EN],
  ])("%s means %o", (question, document) => {
    expect(meant(question, CONTRACTS)).toEqual({ kind: "one", document });
  });

  // protects=with no name to go by, "this one", "this document", "这个文件", "这篇" mean the one Document the last Answer cited, and the User is asked which when it cited two; fails_when=it means every Document, or picks one of two; no_claim=Documents the conversation named without citing.
  test.each([
    ["Summarise this one.", ["en"], { kind: "one", document: EN }],
    ["Write a short summary of this document.", ["zh"], { kind: "one", document: ZH }],
    ["总结一下这个文件", ["en"], { kind: "one", document: EN }],
    ["这篇讲了什么？", ["zh"], { kind: "one", document: ZH }],
    ["总结一下这个文件", ["en", "zh"], { kind: "which", documents: CONTRACTS }],
    ["Summarise this one.", [], { kind: "which", documents: [] }],
  ] as const)("%s, after an Answer citing %o, means %o", (question, cited, expected) => {
    expect(meant(question, CONTRACTS, { cited: [...cited] })).toEqual(expected);
  });

  // protects=two Documents that fit the words equally leave the User to say which, unless the last Answer cited one of them; fails_when=one is picked unasked; no_claim=none beyond protects.
  test("two that fit equally: which, unless the last Answer cited one", () => {
    expect(meant("Summarise this lease", LEASES)).toEqual({ kind: "which", documents: LEASES });
    expect(meant("Summarise this lease", LEASES, { cited: ["l2"] })).toEqual({
      kind: "one",
      document: LEASES[1],
    });
  });

  // protects=the only Document in scope, or the one the Mind is the working copy of, is the one meant, whatever the words; fails_when=the User is asked which; no_claim=none beyond protects.
  test("the only Document in scope, or the working copy's own, is the one", () => {
    expect(meant("这份文件讲了什么？", [EN])).toEqual({ kind: "one", document: EN });
    expect(meant("Summarise this document", CONTRACTS, { original: ZH })).toEqual({
      kind: "one",
      document: ZH,
    });
  });

  // protects=words that point at the User's own text, at a question, or at no Document at all point at none; fails_when=one narrows the search or asks which Document; no_claim=every wording.
  test.each([
    "Make this paragraph shorter",
    "Write this one-page summary",
    "This is the agreement we signed",
    "写一份这篇笔记的总结",
    "把这份总结改短一点",
    "这个问题怎么解决？",
    "Summarise the agreements",
    "总结这些文档",
  ])("%s points at no Document", (question) => {
    expect(meant(question, CONTRACTS)).toBeNull();
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
];

/** Searches, then writes what it found into the Mind with add_to_mind. */
function writer(calls: ModelCall[]) {
  return scriptedModel((call) => {
    if (!call.tools.includes("add_to_mind")) return { text: "Done." };
    calls.push(call);
    if (!call.results.some((each) => each.tool === "search_documents")) {
      return {
        calls: [{ tool: "search_documents", input: { query: "leave notice 年休假 解除" } }],
      };
    }
    if (!call.results.some((each) => each.tool === "add_to_mind")) {
      return {
        calls: [{ tool: "add_to_mind", input: { content: "## Summary\n\nLeave and notice." } }],
      };
    }
    return { text: "Done." };
  });
}

const documentsFound = (calls: readonly ModelCall[]) => [
  ...new Set(
    calls.flatMap((call) =>
      call.results
        .filter((each) => each.tool === "search_documents")
        .flatMap((each) => shownPassages(each.text).map((passage) => passage.document)),
    ),
  ),
];

describe("A short summary of “this” Document, in an empty Mind", { timeout: 60_000 }, () => {
  // protects=the Chinese request and its English twin each search their own contract alone, are told to keep it to five lines, and write into the empty Mind the User is in, not a new one; fails_when=the search covers both contracts, the Chinese one is left in the panel or goes to a new Mind, or the length isn't asked for; no_claim=that a real model keeps to five lines.
  test.each([
    ["请为这份劳动合同写一个简短的摘要。", "劳动合同"],
    ["Write a short summary of this agreement.", "Employment agreement"],
  ])("“%s” is written from “%s” alone, into the empty Mind", async (request, name) => {
    const calls: ModelCall[] = [];
    const { core, mind } = await setUpWithDocuments(writer(calls), FILES);

    const answer = await askAssistant(core, { text: request, mindId: mind.id });

    expect(documentsFound(calls)).toEqual([name]);
    const system = calls[0]?.system ?? "";
    expect(system).toContain(`The User points at one Document, "${name}"`);
    expect(system).toContain("five lines at most");
    expect(system).toContain("write it there with add_to_mind");
    expect(changesIn(answer).map((change) => [change.mindId, change.created])).toEqual([
      [mind.id, false],
    ]);
  });

  // protects=when two Documents fit "this one" equally, the model is told to ask which and nothing is owed to the Mind, so the app doesn't ask it again to write; fails_when=the request still routes as a write, or the model is asked again; no_claim=the wording of a real model's question.
  test("two that fit equally: the model asks which, and is not asked again to write", async () => {
    const calls: ModelCall[] = [];
    const model = scriptedModel((call) => {
      if (call.tools.includes("add_to_mind")) calls.push(call);
      return { text: "Which one: 劳动合同 or Employment agreement?" };
    });
    const { core, mind } = await setUpWithDocuments(model, FILES);

    const answer = await askAssistant(core, {
      text: "Write a short summary of this document.",
      mindId: mind.id,
    });

    expect(calls).toHaveLength(1);
    expect(calls[0]?.system).toContain("Ask them which they mean, in one line");
    expect(calls[0]?.system).not.toContain("write it there with add_to_mind");
    expect(changesIn(answer)).toEqual([]);
  });
});
