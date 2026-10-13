import { create } from "zustand";
import type { ErrorWording } from "../../core/errors";
import { describeError } from "./errors";

/**
 * Undo instead of asking (docs/agents/interaction.md, rule 4): a change the User
 * can take back happens at once and goes on this session's stack, offered as
 * the undo line ("Moved “Shortlist” to Finance · Undo", see `UndoLine`) and as
 * ⌘Z / Ctrl+Z outside text, which ⇧⌘Z / Ctrl+Shift+Z (or Ctrl+Y) does again.
 */
export interface Undoable {
  /** What happened, said plainly. */
  message: string;
  /** What was done, for the Edit menu: "Move 3 Documents" makes "Undo Move 3 Documents". */
  name?: string;
  /** Takes it back. Throws `UndoRefused` when a fix would let it. */
  undo(): Promise<void>;
  /**
   * Where its line shows, when that is where the User acted rather than the
   * sidebar's foot: under an Answer in the assistant (`answerUndoPlace`).
   */
  shownAt?: string;
  /** Does it again once taken back. Without it, taking it back ends what can be redone. */
  redo?(): Promise<void>;
}

/** Where the line of what was done with an Answer in the assistant shows: under that Answer. */
export const answerUndoPlace = (answerId: string) => `answer:${answerId}`;

/** The one action that lets a refused undo or redo happen. */
export interface UndoFix {
  label: string;
  run(): Promise<void>;
}

/** An undo or a redo the core can't do yet, with the fix that lets it: it stays on the stack. */
export class UndoRefused extends Error {
  constructor(
    message: string,
    readonly fix: UndoFix,
  ) {
    super(message);
    this.name = "UndoRefused";
  }
}

/** Why taking a change back, or doing it again, didn't happen. */
export interface UndoProblem {
  step: "undo" | "redo";
  reason: string;
  /** The reason as a key of the dictionary, when the core gave one. */
  wording?: ErrorWording;
  fix: UndoFix | null;
}

export type UndoLineState =
  | { id: number; change: Undoable; problem?: undefined }
  | { id: number; problem: UndoProblem; change?: undefined };

interface UndoState {
  /** The changes ⌘Z takes back, the last one last. */
  done: readonly Undoable[];
  /** The changes ⇧⌘Z does again, the last one taken back last. */
  undone: readonly Undoable[];
  /** What the undo line shows: the change just made or done again, or why one didn't happen. */
  line: UndoLineState | null;
  /** Puts a change just made on the stack and shows its line; nothing more can be redone. */
  push(change: Undoable): void;
  undo(): Promise<void>;
  redo(): Promise<void>;
  /** Takes a change off the stack, once it can't be taken back here any more. */
  withdraw(change: Undoable): void;
  /** Hides the line; the stack stays. */
  dismiss(): void;
}

/** How long the line shows before it goes by itself. */
export const UNDO_SHOWN_MS = 10_000;

let lines = 0;
const lineOf = (change: Undoable): UndoLineState => ({ id: ++lines, change });

/** Undo and redo run one at a time, in the order asked, so a held ⌘Z steps back change by change. */
let queue: Promise<void> = Promise.resolve();
const inTurn = (step: () => Promise<void>): Promise<void> => {
  const turn = queue.then(step);
  queue = turn.catch(() => undefined);
  return turn;
};

export const useUndo = create<UndoState>()((set, get) => {
  /**
   * A refused step: one with a fix stays where it was, to be tried again after the
   * fix; one without can never happen, so it leaves the stack.
   */
  const refused = (step: UndoProblem["step"], change: Undoable, error: unknown) => {
    const fix = error instanceof UndoRefused ? error.fix : null;
    const stack = step === "undo" ? "done" : "undone";
    const { message, wording } = describeError(error);
    set((state) => ({
      ...(fix &&
        (step === "undo"
          ? { done: [...state.done, change] }
          : { undone: [...state.undone, change] })),
      line: {
        id: ++lines,
        problem: {
          step,
          reason: message,
          ...(wording && { wording }),
          fix: fix && {
            label: fix.label,
            run: async () => {
              await fix.run();
              if (get()[stack].at(-1) === change) await get()[step]();
            },
          },
        },
      },
    }));
  };

  return {
    done: [],
    undone: [],
    line: null,
    push(change) {
      // A command offers the change its action already offered.
      if (get().done.at(-1) === change) return;
      set((state) => ({ done: [...state.done, change], undone: [], line: lineOf(change) }));
    },
    undo: () =>
      inTurn(async () => {
        const change = get().done.at(-1);
        if (!change) return;
        set((state) => ({ done: state.done.slice(0, -1), line: null }));
        try {
          await change.undo();
        } catch (error) {
          refused("undo", change, error);
          return;
        }
        set((state) => ({ undone: change.redo ? [...state.undone, change] : [] }));
      }),
    redo: () =>
      inTurn(async () => {
        const change = get().undone.at(-1);
        if (!change?.redo) return;
        set((state) => ({ undone: state.undone.slice(0, -1), line: null }));
        try {
          await change.redo();
        } catch (error) {
          refused("redo", change, error);
          return;
        }
        set((state) => ({ done: [...state.done, change], line: lineOf(change) }));
      }),
    withdraw(change) {
      set((state) => ({
        done: state.done.filter((each) => each !== change),
        undone: state.undone.filter((each) => each !== change),
        ...(state.line?.change === change && { line: null }),
      }));
    },
    dismiss() {
      set({ line: null });
    },
  };
});

/**
 * Whether a key goes to something the User types in, whose ⌘Z is its own: a
 * field that takes text, or editable text such as the note. A checkbox or a
 * button does not.
 */
export function typesText(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target instanceof HTMLInputElement) return !NOT_TEXT.has(target.type);
  return target instanceof HTMLTextAreaElement || target.isContentEditable;
}

const NOT_TEXT = new Set([
  "button",
  "checkbox",
  "color",
  "file",
  "hidden",
  "image",
  "radio",
  "range",
  "reset",
  "submit",
]);
