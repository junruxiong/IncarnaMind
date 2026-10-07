import { Editor, type JSONContent } from "@tiptap/core";
import { describe, expect, onTestFinished, test } from "vitest";
import { ANSWER_BLOCK, QUESTION_BLOCK } from "../../src/core";
import { BlockCommands, questionWithAnswer } from "../../src/renderer/src/editor/blockCommands";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

const paragraph = (text: string): JSONContent => ({
  type: "paragraph",
  content: [{ type: "text", text }],
});
const question = (id: string, text: string): JSONContent => ({
  type: QUESTION_BLOCK,
  attrs: { id },
  content: [{ type: "text", text }],
});
const answer = (questionId: string, text: string): JSONContent => ({
  type: ANSWER_BLOCK,
  attrs: { questionId },
  content: [paragraph(text)],
});

/** A headless editor holding these top-level Blocks, with the Block commands. */
function editorWith(blocks: JSONContent[]): Editor {
  const content = { type: "doc", content: blocks };
  const editor = new Editor({
    element: null,
    extensions: [...noteExtensions(), BlockCommands],
    content,
  });
  onTestFinished(() => editor.destroy());
  return editor;
}

/** Each top-level Block's type and text. */
const blocks = (editor: Editor) => {
  const found: string[] = [];
  editor.state.doc.forEach((node) => {
    found.push(`${node.type.name}: ${node.textContent}`);
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

describe("a Question and its Answer", () => {
  test("deleting a Question deletes its Answer, wherever it is", () => {
    const editor = editorWith([
      paragraph("Before"),
      answer("q2", "An Answer above its Question"),
      question("q1", "First?"),
      answer("q1", "First answer"),
      question("q2", "Second?"),
      paragraph("After"),
    ]);
    editor.commands.deleteBlock(posOf(editor, 2));
    expect(blocks(editor)).toEqual([
      "paragraph: Before",
      `${ANSWER_BLOCK}: An Answer above its Question`,
      `${QUESTION_BLOCK}: Second?`,
      "paragraph: After",
    ]);
    editor.commands.deleteBlock(posOf(editor, 2));
    expect(blocks(editor)).toEqual(["paragraph: Before", "paragraph: After"]);
  });

  test("deleting an Answer leaves its Question", () => {
    const editor = editorWith([question("q1", "First?"), answer("q1", "Gone"), paragraph("After")]);
    editor.commands.deleteBlock(posOf(editor, 1));
    expect(blocks(editor)).toEqual([`${QUESTION_BLOCK}: First?`, "paragraph: After"]);
  });

  test("are dragged as one from either of them, when the Answer comes right after", () => {
    const editor = editorWith([
      paragraph("Before"),
      question("q1", "First?"),
      answer("q1", "First answer"),
      question("q2", "Alone?"),
    ]);
    const { doc } = editor.state;
    const from = posOf(editor, 1);
    const to = posOf(editor, 3);
    expect(questionWithAnswer(doc, posOf(editor, 1))).toEqual({ from, to });
    expect(questionWithAnswer(doc, posOf(editor, 2))).toEqual({ from, to });
    expect(questionWithAnswer(doc, posOf(editor, 0))).toBeNull();
    expect(questionWithAnswer(doc, posOf(editor, 3))).toBeNull();
  });
});
