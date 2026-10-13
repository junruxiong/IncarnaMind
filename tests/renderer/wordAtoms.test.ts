import { Editor, type JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { EditorState } from "@tiptap/pm/state";
import { ySyncPluginKey } from "@tiptap/y-tiptap";
import { describe, expect, onTestFinished, test } from "vitest";
import { IMAGE_NODE, WORD_ATOM_NODE, WORD_BLOCK_NODE } from "../../src/core";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

const atom = (kind: string, label = ""): JSONContent => ({
  type: WORD_ATOM_NODE,
  attrs: { kind, label, text: "", source: `w:${kind}#0` },
});

/**
 * A headless editor of a Word import's paragraphs, recording each time it asks
 * before deleting a section break, and a state with the editor's plugins, as
 * the editor has once it is shown (a headless editor runs none).
 */
function editorWith(blocks: JSONContent[]) {
  const asked: (() => void)[] = [];
  const editor = new Editor({
    element: null,
    extensions: noteExtensions({ onSectionBreakDelete: (retry) => asked.push(retry) }),
    content: { type: "doc", content: blocks },
  });
  onTestFinished(() => editor.destroy());
  const shown = EditorState.create({
    doc: editor.state.doc,
    plugins: editor.extensionManager.plugins,
  });
  return { editor, asked, shown };
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

  test("deleting a section break asks first; the change is made only when the User says so", () => {
    const { editor, asked, shown } = editorWith(twoSections());
    // Select the whole first paragraph, its section break too, and delete it.
    const deleting = shown.tr.delete(0, shown.doc.child(0).nodeSize);
    expect(shown.applyTransaction(deleting).state.doc.eq(shown.doc)).toBe(true);
    expect(asked).toHaveLength(1);

    asked[0]?.();
    expect(sectionBreaks(editor.state.doc)).toBe(0);
    expect(editor.state.doc.textContent).toBe("Section two.");
  });

  test("typing and deleting elsewhere don't ask, nor does moving the paragraph that holds one", () => {
    const { asked, shown } = editorWith(twoSections());
    const first = shown.doc.child(0);
    const typed = shown.applyTransaction(shown.tr.insertText("Now: ", 1)).state;
    const second = typed.doc.child(0).nodeSize;
    const deleted = typed.applyTransaction(typed.tr.delete(second + 1, second + 9)).state;
    const moved = deleted.applyTransaction(
      deleted.tr
        .delete(0, deleted.doc.child(0).nodeSize)
        .insert(deleted.doc.child(1).nodeSize, first),
    ).state;
    expect(asked).toEqual([]);
    expect(sectionBreaks(moved.doc)).toBe(1);
    expect(moved.doc.child(1).textContent).toBe("End of section one.");
    expect(moved.doc.child(0).textContent).toBe("two.");
  });

  test("a change from Yjs (another device, or undo) is never held back", () => {
    const { asked, shown } = editorWith(twoSections());
    const remote = shown.tr
      .delete(0, shown.doc.child(0).nodeSize)
      .setMeta(ySyncPluginKey, { isChangeOrigin: true });
    expect(sectionBreaks(shown.applyTransaction(remote).state.doc)).toBe(0);
    expect(asked).toEqual([]);
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
