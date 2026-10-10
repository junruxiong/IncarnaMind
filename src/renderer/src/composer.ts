import { create } from "zustand";
import type { SearchScope } from "../../core/api";
import { hasSearchScope } from "../../shared/searchScope";

/** What the composer holds for one Mind until it is asked: the text, the Search scope and a Skill. */
export interface Draft {
  text: string;
  /**
   * What the Mind's next Questions search, once the User changed it (a chip
   * removed, or one added with "@"): kept after asking. Empty: every Document.
   * Null until then: the Mind's own, its Folder (see `composerScope`).
   */
  scope: SearchScope | null;
  /** The Skill chosen with "/" for the next Question only. */
  skill: string | null;
  /** Long pastes, each a chip above the text until it is put back, saved as a Document, or asked with. */
  pastes: Paste[];
}

/** Text pasted into the composer that was too long to leave in it (DESIGN.md, Composer › Height). */
export interface Paste {
  id: string;
  text: string;
}

/** A paste over this many characters, or lines, becomes a chip. */
export const LONG_PASTE_CHARS = 2000;
export const LONG_PASTE_LINES = 30;

export const linesOf = (text: string) => text.split("\n").length;
export const isLongPaste = (text: string) =>
  text.length > LONG_PASTE_CHARS || linesOf(text) > LONG_PASTE_LINES;

export const NO_SCOPE: SearchScope = { folderIds: [], tagIds: [], documentIds: [] };

const EMPTY_DRAFT: Draft = { text: "", scope: null, skill: null, pastes: [] };

/** A Mind's own Search scope: its Folder, or every Document for a Mind Not in a Folder. */
export const folderScope = (folderId: string | null): SearchScope =>
  folderId === null ? NO_SCOPE : { folderIds: [folderId], tagIds: [], documentIds: [] };

/**
 * What the composer's next Question searches: the scope the User chose, or
 * else the Mind's own (`folderScope`). The Folder is resolved when the
 * Question is asked, so Documents filed in or out since count as they are then.
 */
export const composerScope = (draft: Draft, folderId: string | null): SearchScope =>
  draft.scope ?? folderScope(folderId);

/**
 * The scope to keep for a Mind once the User changes it: null (the Mind's
 * own again) when it is the Mind's own, so it follows the Mind to another Folder.
 */
export function chosenScope(scope: SearchScope, folderId: string | null): SearchScope | null {
  const own = folderScope(folderId);
  const same = (a: readonly string[], b: readonly string[]) =>
    a.length === b.length && a.every((id, index) => id === b[index]);
  const isOwn =
    same(scope.folderIds, own.folderIds) &&
    same(scope.tagIds, own.tagIds) &&
    same(scope.documentIds, own.documentIds);
  // A Folder's Mind with no scope left searches everything, by the User's choice: kept.
  return isOwn ? null : hasSearchScope(scope) ? scope : NO_SCOPE;
}

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
