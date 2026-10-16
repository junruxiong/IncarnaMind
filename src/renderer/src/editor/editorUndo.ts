import type { Editor } from "@tiptap/core";
import { yUndoPluginKey } from "@tiptap/y-tiptap";
import type * as Y from "yjs";
import { type Undoable, useUndo } from "../undo";

type StackEvent = "stack-item-added" | "stack-item-popped";
type DeleteSet = ReturnType<typeof Y.createDeleteSet>;

/** One step of Undo: what it inserted and deleted, by Yjs item. */
export interface UndoStep {
  insertions: DeleteSet;
  deletions: DeleteSet;
}

export interface UndoStack {
  undoStack: UndoStep[];
  /** A change soon after the last one joins its step. */
  lastChange: number;
  stopCapturing(): void;
  addTrackedOrigin(origin: unknown): void;
  on(event: StackEvent, listener: (event: { stackItem: UndoStep }) => void): void;
  off(event: StackEvent, listener: (event: { stackItem: UndoStep }) => void): void;
}

export const undoStackOf = (editor: Editor): UndoStack | undefined =>
  yUndoPluginKey.getState(editor.state)?.undoManager;

/**
 * Makes what `change` does to the note one step of its Undo, apart from the
 * typing before and after it, so ⌘Z in the note takes back that and nothing else.
 */
export function asOneStep<T>(editor: Editor, change: () => T): T {
  const stack = undoStackOf(editor);
  stack?.stopCapturing();
  try {
    return change();
  } finally {
    stack?.stopCapturing();
  }
}

/**
 * Puts the note's last step on the app's undo stack, with its line ("Accepted
 * 2 changes · Undo", or under an Answer with `shownAt`). Once the User edits
 * again, takes it back in the note, or closes the note, it leaves the stack.
 */
export function offerEditorUndo(editor: Editor, message: string, shownAt?: string): void {
  const manager = undoStackOf(editor);
  const step = manager?.undoStack.at(-1);
  if (!manager || step === undefined) return;
  const change: Undoable = {
    message,
    undo: async () => {
      if (!editor.isDestroyed && manager.undoStack.at(-1) === step) editor.commands.undo();
    },
    ...(shownAt && { shownAt }),
  };
  const stale = () => {
    manager.off("stack-item-added", stale);
    manager.off("stack-item-popped", stale);
    editor.off("destroy", stale);
    useUndo.getState().withdraw(change);
  };
  manager.on("stack-item-added", stale);
  manager.on("stack-item-popped", stale);
  editor.on("destroy", stale);
  useUndo.getState().push(change);
}
