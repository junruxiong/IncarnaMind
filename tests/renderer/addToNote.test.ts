import { Editor, type JSONContent } from "@tiptap/core";
import { TextSelection } from "@tiptap/pm/state";
import { describe, expect, onTestFinished, test } from "vitest";
import { addAtPlace, placeInNote, removeFromNote } from "../../src/renderer/src/editor/addToNote";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import type { BlockJSON } from "../../src/shared/noteBlocks";

const paragraph = (text: string, id: string = crypto.randomUUID()): JSONContent => ({
  type: "paragraph",
  attrs: { id },
  content: text ? [{ type: "text", text }] : [],
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

const texts = (editor: Editor) => {
  const found: string[] = [];
  editor.state.doc.forEach((node) => {
    found.push(`${node.type.name}: ${node.textContent}`);
  });
  return found;
};

const cited: JSONContent = {
  type: "paragraph",
  content: [
    { type: "text", text: "Spring tides come at new moon." },
    { type: "citation", attrs: { documentId: "doc-1", quote: "new moon", check: "found" } },
  ],
};

describe("Add to note, in the open Mind (#201)", () => {
  test("at the cursor, an Answer takes the empty line's place; Undo puts the line back", () => {
    const editor = editorWith([paragraph("Intro"), paragraph("", "empty-line"), paragraph("End")]);
    // The cursor on the empty line.
    let at = -1;
    editor.state.doc.forEach((node, pos) => {
      if (node.attrs.id === "empty-line") at = pos + 1;
    });
    editor.view.dispatch(editor.state.tr.setSelection(TextSelection.create(editor.state.doc, at)));
    const blocks = [cited, paragraph("That is all.")] as BlockJSON[];
    const added = addAtPlace(editor, blocks, placeInNote(editor, true));
    expect(added?.blockIds).toHaveLength(2);
    expect(added?.emptyLine).toBe("empty-line");
    expect(texts(editor)).toEqual([
      "paragraph: Intro",
      "paragraph: Spring tides come at new moon.",
      "paragraph: That is all.",
      "paragraph: End",
    ]);
    // The Citation came along.
    let citations = 0;
    editor.state.doc.descendants((node) => {
      if (node.type.name === "citation") citations++;
    });
    expect(citations).toBe(1);

    removeFromNote(editor, added ?? { blockIds: [], emptyLine: null });
    expect(texts(editor)).toEqual(["paragraph: Intro", "paragraph: ", "paragraph: End"]);
    let line: string | null = null;
    editor.state.doc.forEach((node, _pos, index) => {
      if (index === 1) line = node.attrs.id;
    });
    expect(line).toBe("empty-line");
  });

  test("with the cursor elsewhere, at the end; Undo takes just those Blocks out", () => {
    const editor = editorWith([paragraph("Intro"), paragraph("End")]);
    const added = addAtPlace(editor, [cited] as BlockJSON[], placeInNote(editor, false));
    expect(added?.emptyLine).toBeNull();
    expect(texts(editor)).toEqual([
      "paragraph: Intro",
      "paragraph: End",
      "paragraph: Spring tides come at new moon.",
    ]);
    removeFromNote(editor, added ?? { blockIds: [], emptyLine: null });
    expect(texts(editor)).toEqual(["paragraph: Intro", "paragraph: End"]);
  });
});
