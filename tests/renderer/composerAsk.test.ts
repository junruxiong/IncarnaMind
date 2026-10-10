import { Editor, type JSONContent } from "@tiptap/core";
import { NodeSelection, type Selection, TextSelection } from "@tiptap/pm/state";
import { describe, expect, onTestFinished, test } from "vitest";
import { ANSWER_BLOCK, QUESTION_BLOCK } from "../../src/core";
import {
  cursorBelowAnswer,
  placeQuestion,
  type QuestionToAsk,
  questionAttributes,
  questionContent,
  questionPlace,
  takeBackQuestion,
} from "../../src/renderer/src/editor/composerAsk";
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

const asked = (text: string, more: Partial<QuestionToAsk> = {}): QuestionToAsk => ({
  id: "new-question",
  text,
  model: null,
  scope: { folderIds: [], tagIds: [], documentIds: [] },
  skill: null,
  ...more,
});

/** Asks `text` with the cursor at `selection` (null: not in the Mind), and returns the Blocks. */
function ask(editor: Editor, selection: Selection | null, text = "Why?") {
  const placed = placeQuestion(editor, asked(text), questionPlace(editor.state.doc, selection));
  expect(placed).not.toBeNull();
  return blocks(editor);
}

describe("where a Question asked from the composer goes", () => {
  test("with the cursor not in the Mind: on its empty last line, or after its last Block", () => {
    const ending = editorWith([paragraph("Notes"), paragraph("")]);
    expect(ask(ending, null)).toEqual(["paragraph: Notes", "question: Why?"]);
    const full = editorWith([paragraph("Notes"), paragraph("More notes")]);
    expect(ask(full, null)).toEqual([
      "paragraph: Notes",
      "paragraph: More notes",
      "question: Why?",
    ]);
  });

  test("on the empty line the cursor is on, which it replaces", () => {
    const editor = editorWith([paragraph("First"), paragraph(""), paragraph("Last")]);
    expect(ask(editor, cursorIn(editor, 1))).toEqual([
      "paragraph: First",
      "question: Why?",
      "paragraph: Last",
    ]);
  });

  test("after the Block the cursor is in, never splitting its text", () => {
    const editor = editorWith([paragraph("First thought"), paragraph("Last")]);
    expect(ask(editor, cursorIn(editor, 0, 5))).toEqual([
      "paragraph: First thought",
      "question: Why?",
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
    expect(ask(inAnswer, insideAnswer)).toEqual([
      "question: Earlier?",
      "answer: Earlier answer.",
      "question: Why?",
      "paragraph: Last",
    ]);
    const inQuestion = editorWith(blocksBefore);
    expect(ask(inQuestion, cursorIn(inQuestion, 0, 3))).toEqual([
      "question: Earlier?",
      "answer: Earlier answer.",
      "question: Why?",
      "paragraph: Last",
    ]);
  });

  test("after a whole Block selected", () => {
    const editor = editorWith([paragraph("First"), { type: "horizontalRule" }, paragraph("Last")]);
    const rule = NodeSelection.create(editor.state.doc, posOf(editor, 1));
    expect(ask(editor, rule)).toEqual([
      "paragraph: First",
      "horizontalRule: ",
      "question: Why?",
      "paragraph: Last",
    ]);
  });
});

describe("the Question put into the note", () => {
  test("keeps its lines as line breaks", () => {
    expect(questionContent("Which comes first?\nAnd why?")).toEqual([
      { type: "text", text: "Which comes first?" },
      { type: "hardBreak" },
      { type: "text", text: "And why?" },
    ]);
    const editor = editorWith([paragraph("")]);
    expect(ask(editor, null, "Which comes first?\nAnd why?")).toEqual([
      "question: Which comes first?⏎And why?",
    ]);
  });

  test("carries the Mind's model, the Search scope and the Skill, as Questions always stored them", () => {
    const scope = { folderIds: ["f1"], tagIds: [], documentIds: ["d1", "d2"] };
    const toAsk = asked("Why?", {
      model: { providerId: "p1", modelId: "m1" },
      scope,
      skill: "tide-tables",
    });
    expect(questionAttributes(toAsk)).toEqual({
      id: "new-question",
      providerId: "p1",
      modelId: "m1",
      scopeFolderIds: ["f1"],
      scopeTagIds: null,
      scopeDocumentIds: ["d1", "d2"],
      forcedSkill: "tide-tables",
    });
    const editor = editorWith([paragraph("")]);
    placeQuestion(editor, toAsk, questionPlace(editor.state.doc, null));
    expect(editor.state.doc.firstChild?.attrs).toEqual(questionAttributes(toAsk));
    // With the default model and no scope, nothing of either is stored.
    expect(questionAttributes(asked("Why?"))).toMatchObject({
      providerId: null,
      modelId: null,
      scopeFolderIds: null,
      scopeTagIds: null,
      scopeDocumentIds: null,
      forcedSkill: null,
    });
  });

  test("taken back when it couldn't be asked: the empty line it replaced comes back, and nothing else", () => {
    const onLine = editorWith([paragraph("First"), paragraph(""), paragraph("Last")]);
    const placed = placeQuestion(
      onLine,
      asked("Why?"),
      questionPlace(onLine.state.doc, cursorIn(onLine, 1)),
    );
    if (!placed) throw new Error("Not placed.");
    takeBackQuestion(onLine, placed, "Why?");
    expect(blocks(onLine)).toEqual(["paragraph: First", "paragraph: ", "paragraph: Last"]);

    const after = editorWith([paragraph("First"), paragraph("Last")]);
    const added = placeQuestion(
      after,
      asked("Why?"),
      questionPlace(after.state.doc, cursorIn(after, 0)),
    );
    if (!added) throw new Error("Not placed.");
    takeBackQuestion(after, added, "Why?");
    expect(blocks(after)).toEqual(["paragraph: First", "paragraph: Last"]);
  });

  test("not taken back once the User changed it", () => {
    const editor = editorWith([paragraph("")]);
    const placed = placeQuestion(editor, asked("Why?"), questionPlace(editor.state.doc, null));
    if (!placed) throw new Error("Not placed.");
    editor.commands.command(({ tr }) => {
      tr.insertText("Not ", 1);
      return true;
    });
    takeBackQuestion(editor, placed, "Why?");
    expect(blocks(editor)).toEqual(["question: Not Why?"]);
  });
});

describe("after asking, the Mind's cursor", () => {
  test("goes on a new empty line under the Answer, or the one already there", () => {
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
});
