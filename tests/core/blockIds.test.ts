import { Editor, type JSONContent } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { updateYFragment, yXmlFragmentToProseMirrorRootNode } from "@tiptap/y-tiptap";
import { describe, expect, onTestFinished, test } from "vitest";
import * as Y from "yjs";
import { BLOCK_ID_ATTRIBUTE, noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind, type MindClient } from "../helpers/mindClient";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

/**
 * A Tiptap editor on the Note schema with no view: the renderer's editor minus
 * the screen. Its UniqueID plugin assigns Block IDs exactly as on screen.
 */
function headlessEditor(
  content: JSONContent = { type: "doc", content: [{ type: "paragraph" }] },
): Editor {
  const editor = new Editor({ element: null, extensions: noteExtensions(), content });
  // Tiptap adds the extensions' plugins (UniqueID's among them) when it mounts a view.
  editor.view.updateState(editor.state.reconfigure({ plugins: editor.extensionManager.plugins }));
  onTestFinished(() => editor.destroy());
  return editor;
}

/**
 * Writes every change the editor makes into the Mind's Yjs document, the way
 * the editor's Yjs binding does on screen.
 */
function bindToMind(editor: Editor, client: MindClient): void {
  const meta: Parameters<typeof updateYFragment>[3] = { mapping: new Map(), isOMark: new Map() };
  const write = () => {
    client.doc.transact(() => updateYFragment(client.doc, client.blocks, editor.state.doc, meta));
  };
  write();
  editor.on("transaction", write);
}

/** The top-level Blocks of a document: their type, ID and text. */
function blocksOf(doc: ProseMirrorNode) {
  const blocks: { type: string; id: unknown; text: string }[] = [];
  doc.forEach((node) => {
    blocks.push({
      type: node.type.name,
      id: node.attrs[BLOCK_ID_ATTRIBUTE],
      text: node.textContent || node.attrs.latex || "",
    });
  });
  return blocks;
}

/** The position just before the top-level Block at `index`. */
function positionOfBlock(doc: ProseMirrorNode, index: number): number {
  let pos = 0;
  for (let i = 0; i < index; i++) pos += doc.child(i).nodeSize;
  return pos;
}

/** Moves the top-level Block at `from` to sit before the Block at `to`, as dropping a dragged Block does. */
function dragBlock(editor: Editor, from: number, to: number): void {
  const { doc, tr } = editor.state;
  const block = doc.child(from);
  const start = positionOfBlock(doc, from);
  tr.delete(start, start + block.nodeSize);
  tr.insert(tr.mapping.map(positionOfBlock(doc, to)), block);
  editor.view.dispatch(tr);
}

const NOTE: JSONContent[] = [
  { type: "heading", attrs: { level: 1 }, content: [{ type: "text", text: "Results" }] },
  { type: "paragraph", content: [{ type: "text", text: "Effect sizes were small." }] },
  {
    type: "codeBlock",
    attrs: { language: "python" },
    content: [{ type: "text", text: "print(effect)" }],
  },
  { type: "blockMath", attrs: { latex: "d = \\frac{\\mu_1 - \\mu_2}{\\sigma}" } },
  {
    type: "bulletList",
    content: [
      {
        type: "listItem",
        content: [{ type: "paragraph", content: [{ type: "text", text: "A" }] }],
      },
    ],
  },
];

describe("Block IDs", () => {
  test("every new top-level Block gets a UUID, kept as it is edited, turned into a heading or dragged", () => {
    const editor = headlessEditor();
    editor.commands.insertContent(NOTE);

    const written = blocksOf(editor.state.doc);
    expect(written.map((block) => block.type)).toEqual([
      "heading",
      "paragraph",
      "codeBlock",
      "blockMath",
      "bulletList",
    ]);
    for (const block of written) expect(block.id).toMatch(UUID);
    expect(new Set(written.map((block) => block.id)).size).toBe(written.length);
    const [heading, paragraph, code, math, list] = written.map((block) => block.id);

    // Typing into a Block and turning it into a heading keep its ID.
    const paragraphStart = positionOfBlock(editor.state.doc, 1) + 1;
    editor
      .chain()
      .insertContentAt(paragraphStart, { type: "text", text: "Overall: " })
      .setNode("heading", { level: 2 })
      .run();
    expect(blocksOf(editor.state.doc)[1]).toEqual({
      type: "heading",
      id: paragraph,
      text: "Overall: Effect sizes were small.",
    });

    // Splitting a Block (Enter) keeps the ID on the first half and gives the second a new one.
    editor
      .chain()
      .setTextSelection(paragraphStart + "Overall:".length)
      .splitBlock()
      .run();
    const afterSplit = blocksOf(editor.state.doc);
    expect(afterSplit[1]).toMatchObject({ id: paragraph, text: "Overall:" });
    expect(afterSplit[2]?.id).toMatch(UUID);
    expect([heading, paragraph, code, math, list]).not.toContain(afterSplit[2]?.id);

    // Dragging the formula above the heading keeps every ID.
    dragBlock(editor, 4, 0);
    expect(blocksOf(editor.state.doc).map((block) => block.id)).toEqual([
      math,
      heading,
      paragraph,
      afterSplit[2]?.id,
      code,
      list,
    ]);
  });

  test("a Note's Block IDs are stored in the Mind's Yjs document and are the same after a restart", async () => {
    const dataDir = await createTempDataFolder();
    const firstRun = startCore(dataDir);
    const mind = await firstRun.createMind({ title: "Results" });
    const writer = await connectToMind(firstRun, mind.id);
    const editor = headlessEditor();
    bindToMind(editor, writer);

    editor.commands.insertContent(NOTE);
    dragBlock(editor, 3, 1);
    editor.commands.deleteRange({
      from: positionOfBlock(editor.state.doc, 4),
      to: positionOfBlock(editor.state.doc, 5),
    });
    await writer.settled();
    const written = blocksOf(editor.state.doc);
    expect(written.map((block) => block.type)).toEqual([
      "heading",
      "blockMath",
      "paragraph",
      "codeBlock",
    ]);
    for (const block of written) expect(block.id).toMatch(UUID);

    // The IDs are attributes of the top-level elements of the Yjs document.
    const stored = writer.blocks.toArray().map((element) => {
      if (!(element instanceof Y.XmlElement))
        throw new Error("A top-level Block isn't an element.");
      return element.getAttribute(BLOCK_ID_ATTRIBUTE);
    });
    expect(stored).toEqual(written.map((block) => block.id));
    firstRun.close();

    // After a restart, the Mind reads back with the same Blocks, IDs and order.
    const secondRun = startCore(dataDir);
    const reader = await connectToMind(secondRun, mind.id);
    const restored = yXmlFragmentToProseMirrorRootNode(reader.blocks, editor.schema);
    expect(blocksOf(restored)).toEqual(written);

    // Editing it again keeps them, and only the new Block gets a new ID.
    const reopened = headlessEditor(restored.toJSON());
    expect(blocksOf(reopened.state.doc)).toEqual(written);
    reopened.commands.insertContentAt(reopened.state.doc.content.size, {
      type: "paragraph",
      content: [{ type: "text", text: "Next steps" }],
    });
    const edited = blocksOf(reopened.state.doc);
    expect(edited.slice(0, written.length)).toEqual(written);
    expect(edited.at(-1)?.id).toMatch(UUID);
    expect(written.map((block) => block.id)).not.toContain(edited.at(-1)?.id);
  });
});
