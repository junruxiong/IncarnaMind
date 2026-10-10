import { create } from "zustand";
import type { SearchScope } from "../../core/api";

/** What the composer holds for one Mind until it is asked: the text, the Search scope and a Skill. */
export interface Draft {
  text: string;
  /** What the Mind's next Questions search: kept after asking. Empty: every Document. */
  scope: SearchScope;
  /** The Skill chosen with "/" for the next Question only. */
  skill: string | null;
}

export const NO_SCOPE: SearchScope = { folderIds: [], tagIds: [], documentIds: [] };

const EMPTY_DRAFT: Draft = { text: "", scope: NO_SCOPE, skill: null };

interface ComposerState {
  /** By Mind id: switching tabs keeps what was typed in each Mind's composer. */
  drafts: Readonly<Record<string, Draft>>;
  /**
   * Bumped to ask the open Mind's composer to take the focus, e.g. by a Skill
   * chosen in the note's slash menu.
   */
  focusRequests: number;
  draft(mindId: string): Draft;
  update(mindId: string, change: Partial<Draft>): void;
  requestFocus(): void;
}

/** The composer's drafts, in this window (DESIGN.md, Composer). */
export const useComposer = create<ComposerState>()((set, get) => ({
  drafts: {},
  focusRequests: 0,
  draft: (mindId) => get().drafts[mindId] ?? EMPTY_DRAFT,
  update(mindId, change) {
    set((state) => ({
      drafts: {
        ...state.drafts,
        [mindId]: { ...(state.drafts[mindId] ?? EMPTY_DRAFT), ...change },
      },
    }));
  },
  requestFocus: () => set((state) => ({ focusRequests: state.focusRequests + 1 })),
}));

/** The key under which a problem asking from a Mind's composer is kept (`useAnswers.blocked`). */
export const composerKey = (mindId: string) => `composer:${mindId}`;
