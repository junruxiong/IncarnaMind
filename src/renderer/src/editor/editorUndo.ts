import type { Editor } from "@tiptap/core";
import { yUndoPluginKey } from "@tiptap/y-tiptap";
import { type Undoable, useUndo } from "../undo";

interface UndoStack {
  undoStack: unknown[];
  stopCapturing(): void;
  on(event: "stack-item-added" | "stack-item-popped", listener: () => void): void;
  off(event: "stack-item-added" | "stack-item-popped", listener: () => void): void;
}

const undoStackOf = (editor: Editor): UndoStack | undefined =>
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
