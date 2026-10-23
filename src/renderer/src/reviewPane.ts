import { useEffect, useState } from "react";
import { create } from "zustand";
import { heldByMind } from "./heldByMind";

export const REVIEW_PANE_WIDTH = 296;
/** Narrower than this beside the pane, the Mind folds the pane to its button (DESIGN.md, the reviewing pane). */
export const NARROWEST_MIND = 760;

export type ReviewFilter = "all" | "unresolved";

export interface ReviewPaneState {
  /** Over the Mind's right edge, while the Mind is too narrow to have it beside; else folded to its button. */
  over: boolean;
  /** Null until the User picks: Unresolved once anything has been decided, else All. */
  filter: ReviewFilter | null;
  /** The author shown (`authorKey`); null for everyone. */
  who: string | null;
  selected: string | null;
  /** Counts the times it was asked to open, so it takes the focus each time. */
  opened: number;
}

interface ReviewPanes {
  panes: Readonly<Record<string, ReviewPaneState>>;
  open(mindId: string): void;
  close(mindId: string): void;
  update(mindId: string, patch: Partial<ReviewPaneState>): void;
}

const CLOSED: ReviewPaneState = { over: false, filter: null, who: null, selected: null, opened: 0 };

/** Which Minds show their reviewing pane (#697), and what it shows. */
export const useReviewPanes = create<ReviewPanes>((set) => ({
  panes: {},
  open: (mindId) =>
    set((state) => {
      const pane = state.panes[mindId];
      return {
        panes: {
          ...state.panes,
          [mindId]: { ...(pane ?? CLOSED), over: true, opened: (pane?.opened ?? 0) + 1 },
        },
      };
    }),
  close: (mindId) =>
    set((state) => {
      const { [mindId]: _closed, ...rest } = state.panes;
      return { panes: rest };
    }),
  update: (mindId, patch) =>
    set((state) => {
      const pane = state.panes[mindId];
      return pane ? { panes: { ...state.panes, [mindId]: { ...pane, ...patch } } } : state;
    }),
}));

/** The key an author is filtered by: the name, or "" for someone the file doesn't name. */
export const authorKey = (author: string | null): string => author ?? "";

/** What each open Mind has to review, for ⌘K and the menus. */
const reviewables = heldByMind<{ has(): boolean; open(): void }>();
export const useReviewables = reviewables.use;
export const holdReviewable = reviewables.hold;

/**
 * Whether the Mind's tab has room for the pane beside its page: else it folds
 * to "Changes ({n})" and opens over the page's right edge. `row` holds both.
 */
export function useRoomBeside(row: HTMLElement | null): boolean {
  const [beside, setBeside] = useState(true);
  useEffect(() => {
    if (!row) return;
    const measure = () => setBeside(row.clientWidth - REVIEW_PANE_WIDTH >= NARROWEST_MIND);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(row);
    return () => observer.disconnect();
  }, [row]);
  return beside;
}
