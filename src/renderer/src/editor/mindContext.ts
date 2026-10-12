import { createContext, useContext } from "react";

/** The Mind an editor shows, for its node views (Questions and Answers ask the core about it). */
export const MindIdContext = createContext<string | null>(null);

/**
 * Writes the Answer an editor shows again: an Answer in the assistant (#201),
 * whose Citations offer it when a quote isn't found. Null in a Mind, whose
 * Answers are Blocks the Citation finds around it.
 */
export const RegenerateAnswerContext = createContext<(() => void) | null>(null);

/**
 * Where a Mind's composer is drawn: the assistant's composer slot, given only
 * to the Mind the assistant reads (see `MindTab`, `assistantTabOf`). Null for
 * every other Mind on screen, and until the slot is there.
 */
export const ComposerDockContext = createContext<HTMLElement | null>(null);

/**
 * Where the open Mind's word count is drawn: right under its title, above any
 * notice (see `MindPane`). Null until it is there.
 */
export const CountSlotContext = createContext<HTMLElement | null>(null);

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
