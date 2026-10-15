import { Editor, type JSONContent } from "@tiptap/core";
import { describe, expect, onTestFinished, test } from "vitest";
import { PROPOSAL_ATTRIBUTE } from "../../src/core";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { typeText } from "../helpers/editor";

/*
 * A Block the assistant proposes whole (#93, #202): Enter at its end makes a
 * Block of the User's own, so what they type there stays when the change is
 * rejected; Enter inside it keeps both halves the assistant's.
 */

const proposal = { id: "change-1", request: "request-1", kind: "insert" };
const proposed = (text: string): JSONContent => ({
  type: "paragraph",
  attrs: { [PROPOSAL_ATTRIBUTE]: proposal },
  content: [{ type: "text", text }],
});

function editorWith(blocks: JSONContent[]): Editor {
  const editor = new Editor({
    element: null,
    extensions: noteExtensions(),
    content: { type: "doc", content: blocks },
  });
  editor.view.updateState(editor.state.reconfigure({ plugins: editor.extensionManager.plugins }));
  onTestFinished(() => editor.destroy());
  return editor;
}

const kinds = (editor: Editor) => {
  const blocks: [string, string | null][] = [];
  editor.state.doc.forEach((block) => {
    const value = block.attrs[PROPOSAL_ATTRIBUTE] as { kind?: string } | null;
    blocks.push([block.textContent, value?.kind ?? null]);
  });
  return blocks;
};

describe("A Block the assistant proposes, as the User types", () => {
  // protects=a line the User starts with Enter at the end of a proposed Block is theirs, so rejecting the change keeps it; fails_when=the new Block carries the Proposal and goes with it; no_claim=typing inside the proposed Block, which stays part of it.
  test("Enter at its end makes a Block of the User's own", () => {
    const editor = editorWith([proposed("Written by the assistant.")]);
    editor.commands.setTextSelection(editor.state.doc.content.size - 1);
    editor.commands.splitBlock();
    typeText(editor, "My own line.");

    expect(kinds(editor)).toEqual([
      ["Written by the assistant.", "insert"],
      ["My own line.", null],
    ]);
  });

  test("Enter inside it keeps both halves the assistant's", () => {
    const editor = editorWith([proposed("Written by the assistant.")]);
    editor.commands.setTextSelection(1 + "Written by".length);
    editor.commands.splitBlock();

    expect(kinds(editor)).toEqual([
      ["Written by", "insert"],
      [" the assistant.", "insert"],
    ]);
  });
});
