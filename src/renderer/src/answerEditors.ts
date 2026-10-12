/**
 * The Answers shown in the assistant (#201), each drawn by a read-only editor
 * of its own, by Answer id: what is selected in one, for Add to note, and
 * Copy, which copies the Answer (or what is selected in it) as the editor
 * does, so it pastes back as Notes with their Citations.
 */
import { type Editor, getTextBetween, getTextSerializersFromSchema } from "@tiptap/core";
import type { Node as ProseMirrorNode, Slice } from "@tiptap/pm/model";
import { NOTE_BLOCK_TYPES } from "../../core/api";
import type { NodeJSON } from "../../shared/blockJson";
import { useAppStore } from "./store";

const editors = new Map<string, Editor>();

/** Lends an Answer's editor for as long as it is shown; returns how to take it back. */
export function lendAnswerEditor(answerId: string, editor: Editor): () => void {
  editors.set(answerId, editor);
  return () => {
    if (editors.get(answerId) === editor) editors.delete(answerId);
  };
}

/** The part of the Answer the User selected in it, as the editor's positions; null for none. */
function selectedRange(editor: Editor): { from: number; to: number } | null {
  const { view } = editor;
  const selection = view.dom.ownerDocument.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  if (!view.dom.contains(range.commonAncestorContainer)) return null;
  try {
    const from = view.posAtDOM(range.startContainer, range.startOffset);
    const to = view.posAtDOM(range.endContainer, range.endOffset);
    return from < to ? { from, to } : null;
  } catch {
    // A selection ProseMirror can't place (inside a formula's rendering, say): the whole Answer.
    return null;
  }
}

/** The slice's Notes: its top-level Blocks with something in them (a rule or a formula counts). */
function notesOf(slice: Slice): NodeJSON[] {
  const nodes: ProseMirrorNode[] = [];
  slice.content.forEach((node) => {
    if ((NOTE_BLOCK_TYPES as readonly string[]).includes(node.type.name)) nodes.push(node);
  });
  return nodes
    .filter((node) => node.isLeaf || node.content.size > 0)
    .map((node) => node.toJSON() as NodeJSON);
}

/**
 * The Notes selected in an Answer (from the top, so the paragraph or list
 * around a selection inside it comes too), or null when nothing in it is
 * selected: then the whole Answer is meant.
 */
export function selectedNotes(answerId: string): NodeJSON[] | null {
  const editor = editors.get(answerId);
  if (!editor || editor.isDestroyed) return null;
  const range = selectedRange(editor);
  if (!range) return null;
  const notes = notesOf(editor.state.doc.slice(range.from, range.to, true));
  return notes.length > 0 ? notes : null;
}

/**
 * Copies an Answer, or what is selected in it: HTML that pastes back as Notes
 * with their Citations, and plain text that names each Citation's source
 * ("[Aldermoor MSA, p. 14]"). Resolves to whether it was copied.
 */
export async function copyAnswer(answerId: string): Promise<boolean> {
  const editor = editors.get(answerId);
  if (!editor || editor.isDestroyed) return false;
  const { doc } = editor.state;
  const range = selectedRange(editor) ?? { from: 0, to: doc.content.size };
  const { dom } = editor.view.serializeForClipboard(doc.slice(range.from, range.to));
  const text = getTextBetween(doc, range, {
    blockSeparator: "\n\n",
    textSerializers: getTextSerializersFromSchema(editor.schema),
  }).trim();
  try {
    await navigator.clipboard.write([
      new ClipboardItem({
        "text/html": new Blob([dom.innerHTML], { type: "text/html" }),
        "text/plain": new Blob([text], { type: "text/plain" }),
      }),
    ]);
    return true;
  } catch (error) {
    useAppStore.getState().reportError(error);
    return false;
  }
}
