import type { TabItem } from "../../core/api";

let pinned: { mindId: string; place: string } | null = null;

/** Whether the tab in front is one the Citation's card is still waited for from: its Document, or its Mind. */
export const staysPinned = (tab: TabItem | undefined, mindId: string): boolean =>
  tab === undefined || tab.kind === "document" || (tab.kind === "mind" && tab.mindId === mindId);

/**
 * A Citation whose card was open when its Document came to the front of a one-group window,
 * `place` being where it is in the Mind (or in a footnote, by the footnote's id). Going on to
 * another Mind or page instead lets it go (see `CitationView`).
 */
export function pinCitation(mindId: string, place: string): void {
  pinned = { mindId, place };
}

export const pinnedMind = (): string | null => pinned?.mindId ?? null;

export function unpinCitation(): void {
  pinned = null;
}

/** Whether the Citation at `place` had its card open when the User left; true once, as the Mind comes back. */
export function takePinnedCitation(mindId: string, place: string): boolean {
  const same = pinned?.mindId === mindId && pinned.place === place;
  if (same) pinned = null;
  return same;
}
