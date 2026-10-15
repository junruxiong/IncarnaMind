import { create } from "zustand";
import type { MindLook } from "../../shared/wordLook";

export interface ShownWordLook {
  shown: MindLook;
  choose(look: MindLook): void;
  /** The Document the original is, when the Library has it. */
  originalId: string | null;
}

/** The Minds open in the window that have a Word file's look, for the Look menu and ⌘K. */
export const useWordLooks = create<Record<string, ShownWordLook>>(() => ({}));

const holders = new Map<string, number>();

/** Until the returned function is called; a Mind open in two panes is held by both. */
export function holdWordLook(mindId: string, entry: ShownWordLook): () => void {
  holders.set(mindId, (holders.get(mindId) ?? 0) + 1);
  useWordLooks.setState({ [mindId]: entry });
  return () => {
    const left = (holders.get(mindId) ?? 1) - 1;
    if (left > 0) {
      holders.set(mindId, left);
      return;
    }
    holders.delete(mindId);
    useWordLooks.setState((state) => {
      const { [mindId]: _gone, ...rest } = state;
      return rest;
    }, true);
  };
}
