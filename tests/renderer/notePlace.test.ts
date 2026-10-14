import { Editor, type JSONContent } from "@tiptap/core";
import { NodeSelection, type Selection, TextSelection } from "@tiptap/pm/state";
import { describe, expect, onTestFinished, test } from "vitest";
import { ANSWER_BLOCK, QUESTION_BLOCK } from "../../src/core";
import {
  cursorBelowAnswer,
  notePlace,
  whenAnswerShown,
} from "../../src/renderer/src/editor/notePlace";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

const paragraph = (text: string): JSONContent => ({
  type: "paragraph",
  content: text ? [{ type: "text", text }] : [],
});
const question = (id: string, text: string): JSONContent => ({
  type: QUESTION_BLOCK,
  attrs: { id },
  content: [{ type: "text", text }],
});
const answer = (questionId: string, text: string): JSONContent => ({
  type: ANSWER_BLOCK,
  attrs: { id: `answer-of-${questionId}`, questionId },
  content: [paragraph(text)],
});

/** A headless editor holding these top-level Blocks. */
function editorWith(blocks: JSONContent[]): Editor {
  const editor = new Editor({
    element: null,
    extensions: noteExtensions(),
    content: { type: "doc", content: blocks },
  });
  onTestFinished(() => editor.destroy());
  return editor;
}

/** Each top-level Block's type and text. */
const blocks = (editor: Editor) => {
  const found: string[] = [];
  editor.state.doc.forEach((node) => {
    found.push(`${node.type.name}: ${node.textBetween(0, node.content.size, " | ", "⏎")}`);
  });
  return found;
};

/** Where the top-level Block at this index starts. */
function posOf(editor: Editor, index: number): number {
  let at = -1;
  editor.state.doc.forEach((_node, pos, i) => {
    if (i === index) at = pos;
  });
  return at;
}

/** The cursor at `offset` characters into the top-level Block at `index`. */
function cursorIn(editor: Editor, index: number, offset = 0) {
  return TextSelection.create(editor.state.doc, posOf(editor, index) + 1 + offset);
}

/** Adds a Note "Added" with the cursor at `selection` (null: not in the Mind), and returns the Blocks. */
function add(editor: Editor, selection: Selection | null) {
  const place = notePlace(editor.state.doc, selection);
  editor.commands.command(({ tr, state }) => {
    tr.replaceWith(place.from, place.to, state.schema.nodeFromJSON(paragraph("Added")));
    return true;
  });
  return blocks(editor);
}

describe("where Add to note puts an Answer", () => {
  test("with the cursor not in the Mind: on its empty last line, or after its last Block", () => {
    const ending = editorWith([paragraph("Notes"), paragraph("")]);
    expect(add(ending, null)).toEqual(["paragraph: Notes", "paragraph: Added"]);
    const full = editorWith([paragraph("Notes"), paragraph("More notes")]);
    expect(add(full, null)).toEqual([
      "paragraph: Notes",
      "paragraph: More notes",
      "paragraph: Added",
    ]);
  });

  test("on the empty line the cursor is on, which it replaces", () => {
    const editor = editorWith([paragraph("First"), paragraph(""), paragraph("Last")]);
    expect(add(editor, cursorIn(editor, 1))).toEqual([
      "paragraph: First",
      "paragraph: Added",
      "paragraph: Last",
    ]);
  });

  test("after the Block the cursor is in, never splitting its text", () => {
    const editor = editorWith([paragraph("First thought"), paragraph("Last")]);
    expect(add(editor, cursorIn(editor, 0, 5))).toEqual([
      "paragraph: First thought",
      "paragraph: Added",
      "paragraph: Last",
    ]);
  });

  test("after an Answer the cursor is in, and after a Question's Answer when the cursor is in the Question", () => {
    const blocksBefore = [
      question("q1", "Earlier?"),
      answer("q1", "Earlier answer."),
      paragraph("Last"),
    ];
    const inAnswer = editorWith(blocksBefore);
    // Inside the Answer's paragraph: two levels down.
    const insideAnswer = TextSelection.create(inAnswer.state.doc, posOf(inAnswer, 1) + 2);
    expect(add(inAnswer, insideAnswer)).toEqual([
      "question: Earlier?",
      "answer: Earlier answer.",
      "paragraph: Added",
      "paragraph: Last",
    ]);
    const inQuestion = editorWith(blocksBefore);
    expect(add(inQuestion, cursorIn(inQuestion, 0, 3))).toEqual([
      "question: Earlier?",
      "answer: Earlier answer.",
      "paragraph: Added",
      "paragraph: Last",
    ]);
  });

  test("after a whole Block selected", () => {
    const editor = editorWith([paragraph("First"), { type: "horizontalRule" }, paragraph("Last")]);
    const rule = NodeSelection.create(editor.state.doc, posOf(editor, 1));
    expect(add(editor, rule)).toEqual([
      "paragraph: First",
      "horizontalRule: ",
      "paragraph: Added",
      "paragraph: Last",
    ]);
  });
});

describe("a Question asked again in the note", () => {
  test("puts the cursor on a new empty line under its Answer, or the one already there", () => {
    const middle = editorWith([
      question("q1", "Why?"),
      answer("q1", "Because."),
      paragraph("Next"),
    ]);
    expect(cursorBelowAnswer(middle, "answer-of-q1")).toBe(true);
    expect(blocks(middle)).toEqual([
      "question: Why?",
      "answer: Because.",
      "paragraph: ",
      "paragraph: Next",
    ]);
    expect(middle.state.selection.$from.index(0)).toBe(2);

    const end = editorWith([question("q1", "Why?"), answer("q1", "Because."), paragraph("")]);
    expect(cursorBelowAnswer(end, "answer-of-q1")).toBe(true);
    expect(blocks(end)).toHaveLength(3);
    expect(end.state.selection.$from.index(0)).toBe(2);
    expect(cursorBelowAnswer(end, "no-such-answer")).toBe(false);
  });

  test("waits for its Answer when the change reaches the editor after the reply to the ask", async () => {
    const editor = editorWith([question("q1", "Why?"), paragraph("")]);
    const shown = whenAnswerShown(editor, "answer-of-q1", 2_000);
    editor.commands.insertContentAt(posOf(editor, 1), answer("q1", "Because."));
    await expect(shown).resolves.toBe(true);
    await expect(whenAnswerShown(editor, "answer-of-q1")).resolves.toBe(true);
    await expect(whenAnswerShown(editor, "never-written", 50)).resolves.toBe(false);
  });
});
