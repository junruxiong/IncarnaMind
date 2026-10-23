import { type Editor, getSchema, type JSONContent } from "@tiptap/core";
import { EditorState, type Transaction } from "@tiptap/pm/state";
import { describe, expect, test, vi } from "vitest";
import { CITATION_NODE } from "../../src/core/api";
import { citationNumbersPlugin } from "../../src/renderer/src/editor/citationNumbers";
import type { ShownClaim } from "../../src/renderer/src/editor/claimChecks";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { sentenceSpans } from "../../src/shared/sentences";

vi.mock("../../src/renderer/src/core", () => ({ core: {} }));
vi.mock("../../src/renderer/src/editWithAi", () => ({
  resolveProposalsIn: vi.fn(),
  requestEdit: vi.fn(),
  useEdits: { getState: () => ({ pending: {} }) },
}));
vi.mock("../../src/renderer/src/modelInUse", () => ({ modelInUseOf: vi.fn() }));

const { cutSentence, offersNotFound, searchedLess } = await import(
  "../../src/renderer/src/unsupportedFixes"
);

const schema = getSchema(noteExtensions());

const paragraph = (text: string, ...inline: JSONContent[]): JSONContent => ({
  type: "paragraph",
  content: [...(text === "" ? [] : [{ type: "text", text }]), ...inline],
});

const citation: JSONContent = {
  type: CITATION_NODE,
  attrs: {
    passageId: null,
    documentId: "lease",
    documentName: "Lease",
    contentHash: "v1",
    pageFrom: 1,
    pageTo: 1,
    location: null,
    quote: "a quote",
    check: "found",
    checkReason: null,
  },
};

function editorHolding(content: JSONContent[], isEditable = true) {
  let state = EditorState.create({
    doc: schema.nodeFromJSON({ type: "doc", content }),
    plugins: [citationNumbersPlugin()],
  });
  const dispatched: Transaction[] = [];
  const editor = {
    get state() {
      return state;
    },
    isEditable,
    view: {
      dispatch(tr: Transaction) {
        dispatched.push(tr);
        state = state.apply(tr);
      },
    },
  } as unknown as Editor;
  return { editor, dispatched };
}

/** The `index`th sentence of the `block`th top-level block, as a check of it would hold it. */
function claimOn(
  editor: Editor,
  block: number,
  index: number,
  over: { state?: string; citedCaseUnchecked?: boolean } = {},
): ShownClaim {
  let start = 0;
  for (let i = 0; i < block; i++) start += editor.state.doc.child(i).nodeSize;
  const node = editor.state.doc.child(block);
  const text = node.textBetween(0, node.content.size, "\n", "￼");
  const span = sentenceSpans(text)[index];
  if (!span) throw new Error("No such sentence.");
  const from = start + 1 + span.from;
  const to = start + 1 + span.to;
  return {
    check: { citedCaseUnchecked: false, scope: { folderId: null, widened: false }, ...over },
    from,
    to,
    sentence: { from, to, text: text.slice(span.from, span.to) },
    shown: { state: over.state ?? "unsupported" },
  } as unknown as ShownClaim;
}

const textsOf = (editor: Editor) => editor.state.doc.children.map((each) => each.textContent);

describe("Cut it", () => {
  test.each([
    ["the first sentence", 0, ["Two dogs. Three birds."]],
    ["a sentence in the middle", 1, ["One cat. Three birds."]],
    ["the last sentence", 2, ["One cat. Two dogs."]],
  ])("takes out %s with the space that kept it from its neighbour", (_name, index, expected) => {
    const { editor, dispatched } = editorHolding([paragraph("One cat. Two dogs. Three birds.")]);
    cutSentence(editor, claimOn(editor, 0, index));
    expect(textsOf(editor)).toEqual(expected);
    expect(dispatched).toHaveLength(1);
  });

  test("takes out a sentence of Chinese, which has no spaces", () => {
    const { editor } = editorHolding([paragraph("猫在睡觉。狗在跑步。鸟在唱歌。")]);
    cutSentence(editor, claimOn(editor, 0, 1));
    expect(textsOf(editor)).toEqual(["猫在睡觉。鸟在唱歌。"]);
  });

  test("takes out the spaces around the last sentence of a paragraph that ends in one", () => {
    const { editor } = editorHolding([paragraph("One cat. Two dogs. ")]);
    cutSentence(editor, claimOn(editor, 0, 1));
    expect(textsOf(editor)).toEqual(["One cat."]);
  });

  test("takes out the paragraph when the sentence is all it says", () => {
    const { editor } = editorHolding([paragraph("One cat."), paragraph("Two dogs.")]);
    cutSentence(editor, claimOn(editor, 0, 0));
    expect(textsOf(editor)).toEqual(["Two dogs."]);
  });

  test("leaves one empty paragraph when it was the only one", () => {
    const { editor } = editorHolding([paragraph("One cat.")]);
    cutSentence(editor, claimOn(editor, 0, 0));
    expect(textsOf(editor)).toEqual([""]);
  });

  test("takes out the list item when the sentence is all it says", () => {
    const item = (text: string): JSONContent => ({
      type: "listItem",
      content: [paragraph(text)],
    });
    const { editor } = editorHolding([
      { type: "bulletList", content: [item("One cat."), item("Two dogs.")] },
    ]);
    const list = editor.state.doc.child(0);
    const first = list.child(0);
    const from = 3;
    const to = from + first.textContent.length;
    cutSentence(editor, {
      check: {},
      from,
      to,
      sentence: { from, to, text: first.textContent },
      shown: { state: "unsupported" },
    } as unknown as ShownClaim);
    expect(editor.state.doc.child(0).childCount).toBe(1);
    expect(editor.state.doc.textContent).toBe("Two dogs.");
  });
});

describe("a sentence that cites nothing and nothing backs", () => {
  test("is offered Cut it and Keep as my view", () => {
    const { editor } = editorHolding([paragraph("One cat.")]);
    expect(offersNotFound(editor, claimOn(editor, 0, 0))).toBe(true);
  });

  test("is not offered them when the sentence has a Citation, or a verdict other than unsupported", () => {
    const { editor } = editorHolding([paragraph("One cat.", citation), paragraph("Two dogs.")]);
    expect(offersNotFound(editor, claimOn(editor, 0, 0))).toBe(false);
    expect(offersNotFound(editor, claimOn(editor, 1, 0, { state: "supported" }))).toBe(false);
    expect(offersNotFound(editor, claimOn(editor, 1, 0, { state: "contradicted" }))).toBe(false);
  });

  test("is not offered them for a court decision no Document holds, which has a card of its own", () => {
    const { editor } = editorHolding([paragraph("One cat.")]);
    expect(offersNotFound(editor, claimOn(editor, 0, 0, { citedCaseUnchecked: true }))).toBe(false);
  });

  test("is not offered them in a Mind that can't be edited", () => {
    const { editor } = editorHolding([paragraph("One cat.")], false);
    expect(offersNotFound(editor, claimOn(editor, 0, 0))).toBe(false);
  });
});

describe("Search all Documents is offered while the check looked in less than every Document", () => {
  const scoped = (scope: Record<string, unknown>) =>
    ({ check: { scope } }) as unknown as ShownClaim;
  const chosen = { folderIds: [], tagIds: [], documentIds: ["lease"] };

  test("in the Mind's Folder, or in the scope the User chose", () => {
    expect(searchedLess(scoped({ folderId: "f1", widened: false }))).toBe(true);
    expect(searchedLess(scoped({ folderId: null, widened: false, chosen }))).toBe(true);
    expect(searchedLess(scoped({ folderId: "f1", widened: false, chosen }))).toBe(true);
  });

  test("not once it looked in every Document, or when it had no Folder", () => {
    expect(searchedLess(scoped({ folderId: "f1", widened: true }))).toBe(false);
    expect(searchedLess(scoped({ folderId: null, widened: false }))).toBe(false);
  });
});
