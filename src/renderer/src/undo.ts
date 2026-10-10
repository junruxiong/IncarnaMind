import { create } from "zustand";

/**
 * Undo instead of asking (docs/agents/interaction.md, rule 4): an action the
 * User can take back happens at once, then offers Undo here, as one short
 * line ("Moved “Shortlist” to Finance · Undo", see `UndoLine`) and as ⌘Z /
 * Ctrl+Z outside a text field. One at a time: a newer one replaces it.
 */
export interface Undoable {
  /** What happened, said plainly. */
  message: string;
  /** Takes it back. A failure is the caller's to report. */
  undo(): Promise<void>;
}

interface UndoState {
  /** The last action that can be taken back, while its line shows. */
  last: (Undoable & { id: number }) | null;
  /** Offers Undo for an action just done. */
  offer(action: Undoable): void;
  /** Takes the last action back, once. */
  undo(): Promise<void>;
  /** Hides the line; the action stays done. */
  dismiss(): void;
}

/** How long the line shows before it goes by itself. */
export const UNDO_SHOWN_MS = 10_000;

let next = 0;

export const useUndo = create<UndoState>()((set, get) => ({
  last: null,
  offer(action) {
    next += 1;
    set({ last: { ...action, id: next } });
  },
  async undo() {
    const { last } = get();
    if (!last) return;
    set({ last: null });
    await last.undo();
  },
  dismiss() {
    set({ last: null });
  },
}));

/** Whether a key goes to something the User types in, whose ⌘Z is its own. */
export function typesText(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target.isContentEditable
  );
}
