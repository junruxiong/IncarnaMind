import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { EditorState } from "@tiptap/pm/state";
import type { MindFocus } from "../../../core/api";

/** A line break as a line of its own; any other inline node that isn't text reads as nothing. */
const leafText = (leaf: ProseMirrorNode) => (leaf.type.name === "hardBreak" ? "\n" : "");

/**
 * Where the User is in the Mind when they ask in the assistant (#544): the
 * text they selected, and the top-level Blocks it is in; with nothing
 * selected, the Block the cursor is in.
 */
export function mindFocusOf(state: EditorState): MindFocus {
  const { selection, doc } = state;
  const blockIds: string[] = [];
  const add = (node: ProseMirrorNode | null | undefined) => {
    const id: unknown = node?.attrs.id;
    if (typeof id === "string" && id !== "") blockIds.push(id);
  };
  if (selection.empty) {
    const { $from } = selection;
    add($from.depth > 0 ? $from.node(1) : $from.nodeAfter);
    return { selection: "", blockIds };
  }
  doc.nodesBetween(selection.from, selection.to, (node) => {
    add(node);
    return false;
  });
  return { selection: doc.textBetween(selection.from, selection.to, "\n\n", leafText), blockIds };
}
