import { Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { Selection } from "@tiptap/pm/state";
import { BLOCK_ID_ATTRIBUTE } from "./noteSchema";

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    blocks: {
      /**
       * Deletes the top-level Block starting at `pos`, or, without `pos`, every
       * top-level Block the selection touches.
       */
      deleteBlock: (pos?: number) => ReturnType;
    };
  }
}

/** Deletes the Block the cursor is in. Shown in the Block menu. */
const DELETE_BLOCK_SHORTCUT = "Mod-Shift-Backspace";

interface Range {
  from: number;
  to: number;
}

/** The top-level Block with this ID, and the position just before it. */
export function findBlock(
  doc: ProseMirrorNode,
  id: string,
): { node: ProseMirrorNode; pos: number } | null {
  let found: { node: ProseMirrorNode; pos: number } | null = null;
  doc.forEach((node, pos) => {
    if (!found && node.attrs[BLOCK_ID_ATTRIBUTE] === id) found = { node, pos };
  });
  return found;
}

function blockAt(doc: ProseMirrorNode, pos: number): Range | null {
  if (pos < 0 || pos >= doc.content.size || doc.resolve(pos).depth !== 0) return null;
  const node = doc.nodeAt(pos);
  return node ? { from: pos, to: pos + node.nodeSize } : null;
}

function selectedBlocks({ $from, $to }: Selection): Range | null {
  const from = $from.depth > 0 ? $from.before(1) : $from.pos;
  const to = $to.depth > 0 ? $to.after(1) : $to.pos;
  return from < to ? { from, to } : null;
}

/** Deleting Blocks as a whole: `deleteBlock`, and Mod-Shift-Backspace for the Block the cursor is in. */
export const BlockCommands = Extension.create({
  name: "blockCommands",

  addCommands() {
    return {
      deleteBlock:
        (pos) =>
        ({ state, tr, dispatch }) => {
          const range =
            pos === undefined ? selectedBlocks(state.selection) : blockAt(state.doc, pos);
          if (!range) return false;
          if (dispatch) {
            const empty = state.schema.topNodeType.createAndFill();
            if (range.from === 0 && range.to === tr.doc.content.size && empty) {
              // A Mind always holds a Block: deleting them all leaves an empty paragraph.
              tr.replaceWith(0, tr.doc.content.size, empty.content);
            } else {
              tr.delete(range.from, range.to);
            }
            const near = Math.min(range.from, tr.doc.content.size);
            tr.setSelection(Selection.near(tr.doc.resolve(near))).scrollIntoView();
          }
          return true;
        },
    };
  },

  addKeyboardShortcuts() {
    return {
      [DELETE_BLOCK_SHORTCUT]: () => this.editor.commands.deleteBlock(),
    };
  },
});
