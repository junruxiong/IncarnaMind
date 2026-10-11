import { Editor, type JSONContent } from "@tiptap/core";
import { EditorState, TextSelection } from "@tiptap/pm/state";
import { describe, expect, onTestFinished, test } from "vitest";
import { PROPOSAL_ATTRIBUTE } from "../../src/core";
import { selectionForEdit } from "../../src/renderer/src/editor/editSelection";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import {
  editTarget,
  ProposalReview,
  pendingAtSelection,
  pendingIn,
  proposalReviewKey,
  proposalsAt,
} from "../../src/renderer/src/editor/proposalReview";

const inserted = (id: string) => ({ type: "proposalInsert", attrs: { id, request: "r1" } });
const removed = (id: string) => ({ type: "proposalDelete", attrs: { id, request: "r1" } });

const text = (value: string, marks?: JSONContent["marks"]): JSONContent => ({
  type: "text",
  text: value,
  ...(marks && { marks }),
});

const citation = (quote: string, proposal: unknown = null): JSONContent => ({
  type: "citation",
  attrs: {
    documentId: "d1",
    documentName: "Tides",
    quote,
    check: "found",
    [PROPOSAL_ATTRIBUTE]: proposal,
  },
});

/**
 * The note's state with the review's plugin, as the editor has it once it is
 * shown (a headless editor adds no plugins): changes go through `apply`.
 */
function noteWith(content: JSONContent[]) {
  const editor = new Editor({
    element: null,
    extensions: [...noteExtensions(), ProposalReview],
    content: { type: "doc", content },
  });
  onTestFinished(() => editor.destroy());
  let state = EditorState.create({
    doc: editor.state.doc,
    plugins: editor.extensionManager.plugins,
  });
  return {
    get state() {
      return state;
    },
    apply(change: (tr: EditorState["tr"]) => void) {
      const tr = state.tr;
      change(tr);
      state = state.applyTransaction(tr).state;
    },
    /** The position just after the first `words` in the text. */
    after(words: string): number {
      let found = -1;
      state.doc.descendants((node, pos) => {
        if (found >= 0 || !node.isText) return found < 0;
        const at = node.text?.indexOf(words) ?? -1;
        if (at >= 0) found = pos + at + words.length;
        return false;
      });
      if (found < 0) throw new Error(`"${words}" isn't in the document.`);
      return found;
    },
  };
}

/** The class a decoration gives, for what the review draws. */
const classOf = (decoration: unknown) =>
  (decoration as { type: { attrs: { class?: string } } }).type.attrs.class;

describe("Proposals in the editor", () => {
  test("are found from their marks and attributes, in order, with what they remove and add", () => {
    const note = noteWith([
      {
        type: "paragraph",
        attrs: { id: "p1" },
        content: [
          text("The tides are "),
          text("very ", [removed("a")]),
          text("strong and "),
          text("really", [removed("b")]),
          text("truly", [inserted("b")]),
          text(" regular "),
          citation("Spring tides", { id: "c", request: "r1", kind: "delete" }),
          text("."),
        ],
      },
      {
        type: "paragraph",
        attrs: { id: "p2", [PROPOSAL_ATTRIBUTE]: { id: "d", request: "r1", kind: "insert" } },
        content: [text("A new paragraph.")],
      },
    ]);

    const proposals = pendingIn(note.state);
    expect(proposals.map((each) => [each.id, each.removed, each.added, each.block])).toEqual([
      ["a", "very ", "", null],
      ["b", "really", "truly", null],
      ["c", "￼", "", null],
      ["d", "", "A new paragraph.", "insert"],
    ]);
    expect(proposals.map((each) => each.droppedCitations)).toEqual([0, 0, 1, 0]);
    expect(proposals[1]?.ranges).toHaveLength(1);
    // The Citation and the new paragraph are drawn as changes: node views draw no marks.
    const drawn = proposalReviewKey.getState(note.state)?.decorations.find().map(classOf).sort();
    expect(drawn).toEqual([
      "proposal-node proposal-node--delete",
      "proposal-node proposal-node--insert",
    ]);
  });

  test("the cursor is on the change it is in or beside, and a selection on every change it touches", () => {
    const note = noteWith([
      {
        type: "paragraph",
        attrs: { id: "p1" },
        content: [
          text("one "),
          text("two ", [removed("a")]),
          text("three "),
          text("four", [inserted("b")]),
          text(" five"),
        ],
      },
    ]);
    const proposals = pendingIn(note.state);
    const at = (from: number, to = from) =>
      proposalsAt(proposals, { from, to, empty: from === to }).map((each) => each.id);

    expect(at(note.after("one "))).toEqual(["a"]);
    expect(at(note.after("tw"))).toEqual(["a"]);
    expect(at(note.after("thr"))).toEqual([]);
    // A section: from the first change to the second.
    expect(at(note.after("o"), note.after("fou"))).toEqual(["a", "b"]);

    note.apply((tr) => tr.setSelection(TextSelection.create(tr.doc, note.after("fo"))));
    expect(pendingAtSelection(note.state).map((each) => each.id)).toEqual(["b"]);
    // The change the cursor is in is washed.
    const washed = proposalReviewKey.getState(note.state)?.decorations.find();
    expect(washed?.map((each) => note.state.doc.textBetween(each.from, each.to))).toEqual(["four"]);
  });

  test("what the User types into a change is their own", () => {
    const note = noteWith([
      {
        type: "paragraph",
        attrs: { id: "p1" },
        content: [
          text("Keep "),
          text("added words", [inserted("a")]),
          text(" and "),
          text("old", [removed("b")]),
        ],
      },
    ]);
    note.apply((tr) => tr.insertText(" mine", note.after("added")));
    note.apply((tr) => tr.insertText("X", note.after("ol")));

    const marked: string[] = [];
    note.state.doc.descendants((node) => {
      if (node.isText) {
        marked.push(`${node.text}:${node.marks.map((mark) => mark.type.name).join("+")}`);
      }
    });
    expect(marked).toEqual([
      "Keep :",
      "added:proposalInsert",
      " mine:",
      " words:proposalInsert",
      " and :",
      "ol:proposalDelete",
      "X:",
      "d:proposalDelete",
    ]);
    expect(pendingIn(note.state).map((each) => each.id)).toEqual(["a", "b"]);
  });

  test("text the AI is working on stays marked as the User writes before it", () => {
    const note = noteWith([
      { type: "paragraph", attrs: { id: "p1" }, content: [text("Spring tides are high.")] },
    ]);
    note.apply((tr) =>
      tr.setMeta(proposalReviewKey, {
        target: { from: note.after("Spring "), to: note.after("high") },
      }),
    );
    note.apply((tr) => tr.insertText("Big ", 1));

    const target = editTarget(note.state);
    expect(note.state.doc.textBetween(target?.from ?? 0, target?.to ?? 0)).toBe("tides are high");
    const [wash] = proposalReviewKey.getState(note.state)?.decorations.find() ?? [];
    expect(wash).toMatchObject({ from: target?.from, to: target?.to });
    expect(classOf(wash)).toBe("proposal-target");
  });
});

describe("The text Edit with AI changes", () => {
  test("is the selection in paragraphs and headings, with a Citation as one character", () => {
    const note = noteWith([
      { type: "heading", attrs: { id: "h1", level: 2 }, content: [text("Tides")] },
      {
        type: "paragraph",
        attrs: { id: "p1" },
        content: [text("Spring tides "), citation("Spring tides"), text(" are high.")],
      },
    ]);

    expect(
      selectionForEdit(note.state, { from: note.after("Ti"), to: note.after(" are") }),
    ).toMatchObject({
      from: { blockId: "h1", offset: 2 },
      to: { blockId: "p1", offset: "Spring tides ￼ are".length },
      text: "des\nSpring tides ￼ are",
    });
  });

  test("isn't a Question, code, nothing but spaces, or an Answer being written", () => {
    const note = noteWith([
      { type: "question", attrs: { id: "q1" }, content: [text("When are tides high?")] },
      { type: "codeBlock", attrs: { id: "c1" }, content: [text("tides()")] },
      { type: "paragraph", attrs: { id: "p1" }, content: [text("Spring   waters")] },
      {
        type: "answer",
        attrs: { id: "a1", questionId: "q1", status: "streaming" },
        content: [{ type: "paragraph", attrs: { id: "p2" }, content: [text("Being written")] }],
      },
    ]);
    const select = (from: string, to: string) =>
      selectionForEdit(note.state, { from: note.after(from) - from.length, to: note.after(to) });

    expect(select("When", "high")).toBeNull();
    expect(select("tides(", ")")).toBeNull();
    expect(
      selectionForEdit(note.state, { from: note.after("Spring"), to: note.after("Spring  ") }),
    ).toBeNull();
    expect(select("Being", "written")).toBeNull();
    expect(select("Spring", "waters")).toMatchObject({ text: "Spring   waters" });
  });
});
