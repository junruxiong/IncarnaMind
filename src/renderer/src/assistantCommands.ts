/**
 * The changes the assistant's commands made (#1110) are one step of Undo per Run, in the window
 * whose Question asked for them, as the User's own changes to the Library are: once the Answer
 * has ended, the undo line says what changed ("Renamed 3 Documents · Undo"), and ⌘Z, that line
 * or the Answer's card takes all of it back.
 */
import { create } from "zustand";
import type { CommandDone } from "../../core/api";
import { commandRuns, runText, undoable } from "./commandFacts";
import { core } from "./core";
import { useAppStore } from "./store";
import { type Undoable, useUndo } from "./undo";

interface CommandRunsState {
  /** The Runs whose changes this window can take back. */
  here: Readonly<Record<string, true>>;
  /** By Run: whether its changes were taken back. */
  undone: Readonly<Record<string, boolean>>;
}

export const useCommandRuns = create<CommandRunsState>()(() => ({ here: {}, undone: {} }));

const setUndone = (run: string, undone: boolean) =>
  useCommandRuns.setState((state) => ({ undone: { ...state.undone, [run]: undone } }));

const steps = new Map<string, Undoable>();
const askedHere = new Set<string>();

function stepOf(run: string, done: readonly CommandDone[]): Undoable {
  return {
    said: (t) => runText(t, done),
    undo: async () => {
      await core.undoCommands(run);
      setUndone(run, true);
    },
    redo: async () => {
      await core.redoCommands(run);
      setUndone(run, false);
    },
  };
}

/** An Answer this window asks for, or asks for again: the commands it runs can be undone here. */
export function watchCommands(answerId: string): void {
  askedHere.add(answerId);
}

/** The card's Undo: the same step ⌘Z takes, which then leaves the stack. */
export async function undoRun(run: string): Promise<void> {
  const step = steps.get(run);
  if (!step) return;
  const undo = useUndo.getState();
  if (undo.done.at(-1) === step) {
    await undo.undo();
    return;
  }
  try {
    await step.undo();
    undo.withdraw(step);
  } catch (error) {
    useAppStore.getState().reportError(error);
  }
}

core.on("conversation.message", (message) => {
  if (message.kind !== "answer" || !askedHere.has(message.id) || message.status === "streaming") {
    return;
  }
  askedHere.delete(message.id);
  for (const { run, done } of commandRuns(message)) {
    if (steps.has(run) || !undoable(done)) continue;
    const step = stepOf(run, done);
    steps.set(run, step);
    useCommandRuns.setState((state) => ({ here: { ...state.here, [run]: true } }));
    useUndo.getState().push(step);
  }
});
