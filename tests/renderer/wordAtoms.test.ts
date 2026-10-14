import { Editor, type JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { EditorState } from "@tiptap/pm/state";
import { describe, expect, onTestFinished, test } from "vitest";
import { IMAGE_NODE, WORD_ATOM_NODE, WORD_BLOCK_NODE } from "../../src/core";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

const atom = (kind: string, label = ""): JSONContent => ({
  type: WORD_ATOM_NODE,
  attrs: { kind, label, text: "", source: `w:${kind}#0` },
});

/**
 * A headless editor of a Word import's paragraphs, and a state with the
 * editor's plugins, as the editor has once it is shown (a headless editor runs none).
 */
function editorWith(blocks: JSONContent[]) {
  const editor = new Editor({
    element: null,
    extensions: noteExtensions(),
    content: { type: "doc", content: blocks },
  });
  onTestFinished(() => editor.destroy());
  const shown = EditorState.create({
    doc: editor.state.doc,
    plugins: editor.extensionManager.plugins,
  });
  return { editor, shown };
}

const sectionBreaks = (doc: ProseMirrorNode) => {
  let count = 0;
  doc.descendants((node) => {
    if (node.type.name === WORD_ATOM_NODE && node.attrs.kind === "sectionBreak") count++;
  });
  return count;
};

const twoSections = (): JSONContent[] => [
  {
    type: "paragraph",
    content: [{ type: "text", text: "End of section one." }, atom("sectionBreak")],
  },
  { type: "paragraph", content: [{ type: "text", text: "Section two." }] },
];

describe("what a Word import keeps as it is, in the editor", () => {
  test("atoms, kept Blocks and pictures can't be edited inside", () => {
    const { editor } = editorWith([{ type: "paragraph" }]);
    for (const name of [WORD_ATOM_NODE, WORD_BLOCK_NODE, IMAGE_NODE]) {
      const type = editor.schema.nodes[name];
      expect(type?.isAtom).toBe(true);
      expect(type?.isLeaf).toBe(true);
    }
    expect(editor.schema.nodes[WORD_BLOCK_NODE]?.isBlock).toBe(true);
    expect(editor.schema.nodes[WORD_ATOM_NODE]?.isInline).toBe(true);
  });

  test("deleting a section break is an ordinary change, made at once", () => {
    const { shown } = editorWith(twoSections());
    // Select the whole first paragraph, its section break too, and delete it.
    const deleting = shown.tr.delete(0, shown.doc.child(0).nodeSize);
    const { state } = shown.applyTransaction(deleting);
    expect(sectionBreaks(state.doc)).toBe(0);
    expect(state.doc.textContent).toBe("Section two.");
  });

  test("an atom reads as its text, a deletion and the invisible marks as nothing", () => {
    const { editor } = editorWith([
      {
        type: "paragraph",
        content: [
          { type: "text", text: "Dated " },
          atom("field", "9 October 2026"),
          atom("bookmark"),
          { type: WORD_ATOM_NODE, attrs: { kind: "deletion", label: " gone", text: "" } },
          { type: "text", text: "." },
        ],
      },
    ]);
    expect(editor.getText()).toBe("Dated 9 October 2026.");
  });
});
