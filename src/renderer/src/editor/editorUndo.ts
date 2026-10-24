import type { Editor } from "@tiptap/core";
import { yUndoPluginKey } from "@tiptap/y-tiptap";
import type * as Y from "yjs";
import type { WordsInLine } from "../lineNames";
import { type Undoable, useUndo } from "../undo";

type StackEvent = "stack-item-added" | "stack-item-popped";
type StackListener = (event: { stackItem: UndoStep; type: "undo" | "redo" }) => void;
type DeleteSet = ReturnType<typeof Y.createDeleteSet>;

/** One step of Undo: what it inserted and deleted, by Yjs item. */
export interface UndoStep {
  insertions: DeleteSet;
  deletions: DeleteSet;
}

export interface UndoStack {
  undoStack: UndoStep[];
  redoStack: UndoStep[];
  /** A change soon after the last one joins its step. */
  lastChange: number;
  stopCapturing(): void;
  addTrackedOrigin(origin: unknown): void;
  on(event: StackEvent, listener: StackListener): void;
  on(event: "stack-cleared", listener: () => void): void;
  off(event: StackEvent, listener: StackListener): void;
  off(event: "stack-cleared", listener: () => void): void;
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

interface Offered {
  steps: Set<UndoStep>;
  stepping: boolean;
}

const offeredIn = new WeakMap<UndoStack, Offered>();

function offeredOf(manager: UndoStack): Offered {
  let offered = offeredIn.get(manager);
  if (!offered) {
    offered = { steps: new Set(), stepping: false };
    offeredIn.set(manager, offered);
  }
  return offered;
}

const onlyOfferedAbove = (
  stack: readonly UndoStep[],
  step: UndoStep,
  offered: ReadonlySet<UndoStep>,
): boolean => {
  const at = stack.lastIndexOf(step);
  return at >= 0 && stack.slice(at + 1).every((each) => offered.has(each));
};

/**
 * Puts the note's last step on the app's undo stack, with its line ("Accepted
 * 2 changes · Undo", or under an Answer with `shownAt`), to be taken back and
 * done again from anywhere. Once the User edits again, takes it back in the
 * note, or closes the note, it leaves the stack. A later step offered the same
 * way keeps it: each is taken back in turn (#697: decisions one after another
 * in the reviewing pane).
 */
export function offerEditorUndo(editor: Editor, said: WordsInLine, shownAt?: string): void {
  const manager = undoStackOf(editor);
  const first = manager?.undoStack.at(-1);
  if (!manager || first === undefined) return;
  const offered = offeredOf(manager);
  offered.steps.add(first);
  let place: { stack: "undoStack" | "redoStack"; step: UndoStep } = {
    stack: "undoStack",
    step: first,
  };
  const move = (run: () => void, to: "undoStack" | "redoStack") => {
    offered.stepping = true;
    try {
      run();
    } finally {
      offered.stepping = false;
    }
    offered.steps.delete(place.step);
    const step = manager[to].at(-1);
    if (step === undefined) return;
    place = { stack: to, step };
    offered.steps.add(step);
  };
  const onTop = (stack: "undoStack" | "redoStack") =>
    !editor.isDestroyed && place.stack === stack && manager[stack].at(-1) === place.step;
  const change: Undoable = {
    said,
    undo: async () => {
      if (onTop("undoStack")) move(() => editor.commands.undo(), "redoStack");
    },
    redo: async () => {
      if (onTop("redoStack")) move(() => editor.commands.redo(), "undoStack");
    },
    ...(shownAt && { shownAt }),
  };
  const stale = () => {
    manager.off("stack-item-added", changed);
    manager.off("stack-item-popped", changed);
    manager.off("stack-cleared", stale);
    editor.off("destroy", stale);
    offered.steps.delete(place.step);
    useUndo.getState().withdraw(change);
  };
  const changed = () => {
    // Its own steps and those of its neighbours on the stack move only steps on the stack.
    if (offered.stepping) return;
    // Once whatever added a step has had the chance to offer it too.
    queueMicrotask(() => {
      if (!onlyOfferedAbove(manager[place.stack], place.step, offered.steps)) stale();
    });
  };
  manager.on("stack-item-added", changed);
  manager.on("stack-item-popped", changed);
  manager.on("stack-cleared", stale);
  editor.on("destroy", stale);
  useUndo.getState().push(change);
}

/**
 * Steps of a note's Undo, followed as ⌘Z takes them back and ⌘⇧Z does them
 * again, each time as a new step: whether they are in the note ("done"),
 * taken back ("undone"), or no longer in its history ("gone", as when it closed).
 */
export interface FollowedSteps {
  state(): "done" | "undone" | "gone";
  /** Done, and nothing done in the note since: Undo takes back these steps alone. */
  onTop(): boolean;
  add(step: UndoStep): void;
  undo(): void;
  observe(listener: () => void): () => void;
}

export function followSteps(editor: Editor, first: readonly UndoStep[]): FollowedSteps | null {
  const manager = undoStackOf(editor);
  if (!manager) return null;
  const done = new Set(first);
  const undone = new Set<UndoStep>();
  const listeners = new Set<() => void>();
  const changed = () => {
    for (const listener of listeners) listener();
  };
  // Undo and Redo add the step that reverses the one they pop, just before they pop it.
  let added: UndoStep | null = null;
  const onAdded: StackListener = ({ stackItem }) => {
    added = stackItem;
    changed();
  };
  const onPopped: StackListener = ({ stackItem, type }) => {
    const from = type === "undo" ? done : undone;
    const to = type === "undo" ? undone : done;
    if (from.delete(stackItem) && added) to.add(added);
    changed();
  };
  const closed = () => {
    manager.off("stack-item-added", onAdded);
    manager.off("stack-item-popped", onPopped);
    manager.off("stack-cleared", changed);
    editor.off("destroy", closed);
    changed();
  };
  manager.on("stack-item-added", onAdded);
  manager.on("stack-item-popped", onPopped);
  manager.on("stack-cleared", changed);
  editor.on("destroy", closed);
  const inStack = () => manager.undoStack.filter((step) => done.has(step));
  const followed: FollowedSteps = {
    state() {
      if (editor.isDestroyed) return "gone";
      if (inStack().length > 0) return "done";
      return undone.size > 0 ? "undone" : "gone";
    },
    onTop() {
      const ours = inStack().length;
      return (
        !editor.isDestroyed &&
        ours > 0 &&
        manager.undoStack.slice(-ours).every((step) => done.has(step))
      );
    },
    add(step) {
      done.add(step);
      changed();
    },
    undo() {
      let top = manager.undoStack.at(-1);
      while (!editor.isDestroyed && top !== undefined && done.has(top)) {
        editor.commands.undo();
        if (manager.undoStack.at(-1) === top) break;
        top = manager.undoStack.at(-1);
      }
    },
    observe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return followed;
}
