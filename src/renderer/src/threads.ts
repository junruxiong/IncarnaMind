import type * as Y from "yjs";
import { create } from "zustand";
import { chatBlocks, listChats } from "../../shared/mindChats";

/**
 * How the thread above a Mind's composer shows (#201, DESIGN.md D124): which
 * of the Mind's chats, and whether it is up over the Mind or folded to its
 * header. Per Mind, in this window; the chats themselves are in the Mind.
 */
export interface ThreadView {
  /** The chat shown; null for the newest (none yet: the next Question starts one). */
  chatId: string | null;
  /** Up over the lower part of the Mind; otherwise folded to its header. */
  open: boolean;
  /** "New chat" was chosen: the next Question starts a chat, and none is shown till then. */
  fresh: boolean;
  /** An Answer to bring into view and focus once the thread shows it (a link back from a Note). */
  focusAnswer: string | null;
}

/** What one "Add to note" did, so the Answer can say so and offer Undo. */
export interface AddedToNote {
  /** The Mind whose chat holds the Answer. */
  mindId: string;
  answerId: string;
  /** Where it went: this Mind's note (through its editor), or another Mind's (through the core). */
  target: { mindId: string; title: string; here: boolean; created?: boolean };
  /** The top-level Blocks it added, in order. */
  blockIds: string[];
  /** Added here: the empty line they took the place of, which Undo puts back. */
  emptyLine?: string | null;
}

interface ThreadsState {
  views: Readonly<Record<string, ThreadView>>;
  /** The last "Add to note" from each Mind's thread, by that Mind: Undo takes it back. */
  added: Readonly<Record<string, AddedToNote | undefined>>;
  view(mindId: string): ThreadView;
  update(mindId: string, change: Partial<ThreadView>): void;
  /** Brings the thread up, on a chat if given (null: the newest). */
  open(mindId: string, chatId?: string | null): void;
  /** Folds it to its header (Esc). */
  fold(mindId: string): void;
  /** "New chat": the next Question starts one; the thread waits up, empty. */
  startNew(mindId: string): void;
  setAdded(added: AddedToNote | null, mindId: string): void;
}

const CLOSED: ThreadView = { chatId: null, open: false, fresh: false, focusAnswer: null };

/**
 * The chat the next Question goes into: the one the thread shows, or the
 * Mind's newest; null for a new one ("New chat", or none yet).
 */
export function chatToAskIn(view: ThreadView, doc: Y.Doc): string | null {
  if (view.fresh) return null;
  if (view.chatId && chatBlocks(doc, view.chatId)) return view.chatId;
  return listChats(doc)[0]?.id ?? null;
}

/** Whether the Mind's thread is up over it, so Esc folds it. */
export function threadUp(mindId: string, doc: Y.Doc): boolean {
  const view = useThreads.getState().view(mindId);
  return view.open && (view.fresh || listChats(doc).length > 0);
}

/** The thread of each Mind, in this window. */
export const useThreads = create<ThreadsState>()((set, get) => ({
  views: {},
  added: {},
  view: (mindId) => get().views[mindId] ?? CLOSED,
  update(mindId, change) {
    set((state) => ({
      views: { ...state.views, [mindId]: { ...(state.views[mindId] ?? CLOSED), ...change } },
    }));
  },
  open(mindId, chatId) {
    get().update(mindId, {
      open: true,
      ...(chatId !== undefined && { chatId, fresh: false }),
    });
  },
  fold(mindId) {
    get().update(mindId, { open: false, focusAnswer: null });
  },
  startNew(mindId) {
    get().update(mindId, { open: true, fresh: true, chatId: null, focusAnswer: null });
  },
  setAdded(added, mindId) {
    set((state) => ({ added: { ...state.added, [mindId]: added ?? undefined } }));
  },
}));
