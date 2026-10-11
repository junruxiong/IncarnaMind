import { createContext, useContext } from "react";

/** The Mind an editor shows, for its node views (Questions and Answers ask the core about it). */
export const MindIdContext = createContext<string | null>(null);

/**
 * Where the open Mind's composer is drawn: the dock at the foot of the Mind's
 * column, outside the part that scrolls (see `MindTab`). Null until it is there.
 */
export const ComposerDockContext = createContext<HTMLElement | null>(null);

/**
 * Whether a node view is drawn in a footnote's text (`FootnoteText`), where a
 * Citation is part of the footnote and has no number of its own.
 */
export const InFootnoteContext = createContext(false);

export function useMindId(): string {
  const mindId = useContext(MindIdContext);
  if (!mindId) throw new Error("A node view is drawn outside a Mind's editor.");
  return mindId;
}
