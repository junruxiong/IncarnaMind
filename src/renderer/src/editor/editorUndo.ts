import type { Editor } from "@tiptap/core";
import { yUndoPluginKey } from "@tiptap/y-tiptap";
import { useUndo } from "../undo";

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
 * Offers Undo for the note's last step, on the undo line ("Accepted 2 changes
 * · Undo", or under an Answer with `shownAt`), while nothing has been done in
 * the note since: once the User edits again, or takes it back with ⌘Z there,
 * the line has nothing left to undo and goes.
 */
export function offerEditorUndo(editor: Editor, message: string, shownAt?: string): void {
  const manager = undoStackOf(editor);
  const step = manager?.undoStack.at(-1);
  if (!manager || step === undefined) return;
  useUndo.getState().offer({
    message,
    undo: async () => {
      if (!editor.isDestroyed && manager.undoStack.at(-1) === step) editor.commands.undo();
    },
    ...(shownAt && { shownAt }),
  });
  const offered = useUndo.getState().last?.id;
  const stale = () => {
    manager.off("stack-item-added", stale);
    manager.off("stack-item-popped", stale);
    stopWatching();
    if (useUndo.getState().last?.id === offered) useUndo.getState().dismiss();
  };
  const stopWatching = useUndo.subscribe((state) => {
    if (state.last?.id === offered) return;
    manager.off("stack-item-added", stale);
    manager.off("stack-item-popped", stale);
    stopWatching();
  });
  manager.on("stack-item-added", stale);
  manager.on("stack-item-popped", stale);
}
