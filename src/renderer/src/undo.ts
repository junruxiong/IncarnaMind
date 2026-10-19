import { create } from "zustand";
import { translate } from "../../shared/i18n";
import { describeError } from "./errors";
import type { Words, WordsInLine } from "./lineNames";

/**
 * Undo instead of asking (docs/agents/interaction.md, rule 4): a change the User
 * can take back happens at once and goes on this session's stack, offered as
 * the undo line ("Moved “Shortlist” to Finance · Undo", see `UndoLine`) and as
 * ⌘Z / Ctrl+Z outside text, which ⇧⌘Z / Ctrl+Shift+Z (or Ctrl+Y) does again.
 */
export interface Undoable {
  /**
   * What happened, said plainly, in the words of the interface language when it is shown, so a
   * switch of language reaches a line already showing. A long name in it is cut in the middle.
   */
  said: WordsInLine;
  /** What was done, for the Edit menu: "Move 3 Documents" makes "Undo Move 3 Documents". */
  name?: Words;
  /** Takes it back. Throws `UndoRefused` when a fix would let it. */
  undo(): Promise<void>;
  /**
   * What the line says once it is taken back, when it couldn't take back all of it as it was:
   * a Folder it kept for what the User put in it since. Null: nothing.
   */
  undoneNote?(): Words | null;
  /**
   * Where its line shows, when that is where the User acted rather than the
   * sidebar's foot: under an Answer in the assistant (`answerUndoPlace`).
   */
  shownAt?: string;
  /** Does it again once taken back. Without it, taking it back ends what can be redone. */
  redo?(): Promise<void>;
  /** An action before Undo, for a change made for the User rather than by them: "Change". */
  also?: { label: Words; testId: string; run(): void };
}

/** Where the line of what was done with an Answer in the assistant shows: under that Answer. */
export const answerUndoPlace = (answerId: string) => `answer:${answerId}`;

/** The one action that lets a refused undo or redo happen. */
export interface UndoFix {
  label: Words;
  run(): Promise<void>;
}

/**
 * An undo or a redo the core can't do, said in the interface's words: with the fix that lets it,
 * it stays on the stack; with none, it never can and leaves it.
 */
export class UndoRefused extends Error {
  constructor(
    readonly said: Words,
    readonly fix: UndoFix | null,
  ) {
    super(said((key, params) => translate("en", key, params)));
    this.name = "UndoRefused";
  }
}

/** Why taking a change back, or doing it again, didn't happen. */
export interface UndoProblem {
  step: "undo" | "redo";
  reason: Words;
  fix: UndoFix | null;
}

export type UndoLineState =
  | { id: number; change: Undoable; problem?: undefined; note?: undefined }
  | { id: number; problem: UndoProblem; change?: undefined; note?: undefined }
  | { id: number; note: Words; change?: undefined; problem?: undefined };

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
  /**
   * A change already made has more to it, made just now: a request's later commands. It goes
   * back on top, past what the User did meanwhile, so ⌘Z takes it back first, and shows its line.
   */
  renew(change: Undoable): void;
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
    const reason: Words =
      error instanceof UndoRefused
        ? error.said
        : (t) => (wording ? t(wording.key, wording.values) : message);
    set((state) => ({
      ...(fix &&
        (step === "undo"
          ? { done: [...state.done, change] }
          : { undone: [...state.undone, change] })),
      line: {
        id: ++lines,
        problem: {
          step,
          reason,
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
        const note = change.undoneNote?.() ?? null;
        set((state) => ({
          undone: change.redo ? [...state.undone, change] : [],
          ...(note && { line: { id: ++lines, note } }),
        }));
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
    renew(change) {
      set((state) => ({
        done: [...state.done.filter((each) => each !== change), change],
        undone: [],
        line: lineOf(change),
      }));
    },
  };
});

/**
 * Runs changes that each go on the stack as they are made, then leaves them there as one step:
 * Undo takes them back last first, Redo does them again in order.
 */
export async function asOneStep(
  words: Pick<Undoable, "said" | "name">,
  run: () => Promise<void>,
): Promise<void> {
  const before = useUndo.getState().done.length;
  await run();
  const steps = useUndo.getState().done.slice(before);
  if (steps.length < 2) return;
  for (const step of steps) useUndo.getState().withdraw(step);
  useUndo.getState().push({
    ...words,
    undo: async () => {
      for (const step of [...steps].reverse()) await step.undo();
    },
    redo: async () => {
      for (const step of steps) await step.redo?.();
    },
  });
}

export const useAnswerLine = (answerId: string) =>
  useUndo((state) =>
    state.line?.change?.shownAt === answerUndoPlace(answerId) ? state.line : null,
  );

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
