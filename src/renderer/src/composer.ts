import type { Editor } from "@tiptap/core";
import type * as Y from "yjs";
import { create } from "zustand";
import type { ChatModelChoice, SearchScope } from "../../core/api";
import { hasSearchScope } from "../../shared/searchScope";

/** What the composer holds for one Mind until it is asked: the text, the Search scope and a Skill. */
export interface Draft {
  text: string;
  /**
   * The Mind's Search scope once the User changed it (a chip removed, or one
   * added with "@"): kept after asking. Empty: every Document. Null until
   * then: the Mind's own, which `mindSearchScope` turns into what is searched.
   */
  scope: SearchScope | null;
  /** The Skill chosen with "/" for the next Question only. */
  skill: string | null;
  /** "Research…" was chosen: what is typed next is a topic to plan research for (#130). */
  research?: boolean;
  /** Tell it…: the Task what is typed goes to, by id, which reads it at its next step (#130). */
  tell?: string;
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

export const folderScope = (folderId: string | null): SearchScope =>
  folderId === null ? NO_SCOPE : { folderIds: [folderId], tagIds: [], documentIds: [] };

/** The core decides what this searches (`mindSearchScope`), when the Question is asked. */
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

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isStrings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((each) => typeof each === "string");

function isDraft(value: unknown): value is Draft {
  if (!isObject(value)) return false;
  const { text, scope, skill, pastes } = value;
  const validScope =
    scope === null ||
    (isObject(scope) &&
      isStrings(scope.folderIds) &&
      isStrings(scope.tagIds) &&
      isStrings(scope.documentIds));
  return (
    typeof text === "string" &&
    validScope &&
    (skill === null || typeof skill === "string") &&
    Array.isArray(pastes) &&
    pastes.every(
      (paste) => isObject(paste) && typeof paste.id === "string" && typeof paste.text === "string",
    )
  );
}

/** The drafts a reload left, as far as they read: one that doesn't is dropped, never fixed. */
export function draftsFrom(value: unknown): Record<string, Draft> {
  const drafts: Record<string, Draft> = {};
  if (!isObject(value)) return drafts;
  for (const [mindId, draft] of Object.entries(value)) {
    if (isDraft(draft)) drafts[mindId] = draft;
  }
  return drafts;
}

/**
 * The drafts as they are after a reload put back what it left: what the app already put in a
 * composer at its start (a Question that waited for Continue) wins over what the reload left.
 */
export function withReloaded(
  now: Readonly<Record<string, Draft>>,
  left: unknown,
): Record<string, Draft> {
  return { ...draftsFrom(left), ...now };
}

interface ComposerState {
  /** By Mind id: switching tabs keeps what was typed in each Mind's composer. */
  drafts: Readonly<Record<string, Draft>>;
  /**
   * The composer is to take the focus, e.g. for ⌘J or a Skill chosen in the
   * note's slash menu: it does once it is on screen (the assistant unfolds for
   * it), and says so with `focusTaken`.
   */
  focusWanted: boolean;
  /** The model chosen in the composer with no Mind in focus; null: the default. Lost on quitting. */
  modelWithoutMind: ChatModelChoice | null;
  /**
   * The Mind the assistant reads, once its editor is open: what the one composer
   * needs of it (see `Assistant`). Null before then and with no Mind in focus.
   */
  seat: { editor: Editor; doc: Y.Doc; mindId: string } | null;
  /** Text to ask at once, as typed and sent in the composer: ⌘K's "Ask about…". */
  askWanted: string | null;
  draft(mindId: string): Draft;
  update(mindId: string, change: Partial<Draft>): void;
  requestFocus(): void;
  focusTaken(): void;
  requestAsk(text: string): void;
  askTaken(): void;
}

/** The composer's drafts, in this window (DESIGN.md, Composer). */
export const useComposer = create<ComposerState>()((set, get) => ({
  drafts: {},
  focusWanted: false,
  modelWithoutMind: null,
  seat: null,
  draft: (mindId) => get().drafts[mindId] ?? EMPTY_DRAFT,
  update(mindId, change) {
    set((state) => ({
      drafts: {
        ...state.drafts,
        [mindId]: { ...(state.drafts[mindId] ?? EMPTY_DRAFT), ...change },
      },
    }));
  },
  requestFocus: () => set({ focusWanted: true }),
  focusTaken: () => set({ focusWanted: false }),
  askWanted: null,
  requestAsk: (text) => set({ askWanted: text }),
  askTaken: () => set({ askWanted: null }),
}));

/** The draft's key when no Mind is in focus (the Library, Home or a Document): not a Mind's id. */
export const NO_MIND = "assistant";

/** The key under which a problem asking from a Mind's composer is kept (`useAnswers.blocked`). */
export const composerKey = (mindId: string | null) => `composer:${mindId ?? NO_MIND}`;
