import { create } from "zustand";
import type { MindLook } from "../../shared/wordLook";

export interface ShownWordLook {
  shown: MindLook;
  choose(look: MindLook): void;
  openOriginal: (() => void) | null;
}

/** The Minds open in the window that have a Word file's look, for the Look menu and ⌘K. */
export const useWordLooks = create<Record<string, ShownWordLook>>(() => ({}));

const holders = new Map<string, ShownWordLook[]>();

/** Until the returned function is called; a Mind open in two panes is held by both, the latest shown. */
export function holdWordLook(mindId: string, entry: ShownWordLook): () => void {
  holders.set(mindId, [...(holders.get(mindId) ?? []), entry]);
  useWordLooks.setState({ [mindId]: entry });
  return () => {
    const left = (holders.get(mindId) ?? []).filter((each) => each !== entry);
    const latest = left.at(-1);
    if (latest) {
      holders.set(mindId, left);
      useWordLooks.setState({ [mindId]: latest });
      return;
    }
    holders.delete(mindId);
    useWordLooks.setState((state) => {
      const { [mindId]: _gone, ...rest } = state;
      return rest;
    }, true);
  };
}
