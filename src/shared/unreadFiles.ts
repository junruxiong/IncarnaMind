/**
 * The files in a scope that couldn't be read, and why (#740): what each
 * reason says, and what a check's scope held when it started, kept beside the
 * Mind's Blocks (`MIND_CHECK_UNREAD_FIELD`) by the request that made the
 * checks, so the line beside them and the AI use statement say what was
 * unread then, not now. Shared by the core, the renderer and the tests.
 */
import type * as Y from "yjs";
import { MIND_CHECK_UNREAD_FIELD, type UnreadFile, type UnreadReason } from "../core/api";
import { isRecord } from "../core/errors";
import type { MessageKey } from "./i18n";

/** What each reason says, and what to do about it. */
export const UNREAD_REASON_KEYS: Record<UnreadReason, MessageKey> = {
  format: "unread.reason.format",
  scan: "unread.reason.scan",
  password: "unread.reason.password",
  damaged: "unread.reason.damaged",
  "too-large": "unread.reason.too-large",
  pages: "unread.reason.pages",
};

const REASONS = Object.keys(UNREAD_REASON_KEYS) as UnreadReason[];

const unreadOf = (doc: Y.Doc): Y.Map<unknown> => doc.getMap(MIND_CHECK_UNREAD_FIELD);

const text = (value: unknown): string | null => (typeof value === "string" ? value : null);

function keptFiles(value: unknown): Omit<UnreadFile, "id">[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((each) => {
    if (!isRecord(each)) return [];
    const name = text(each.name);
    const reason = REASONS.find((known) => known === each.reason);
    if (!name || !reason) return [];
    return [
      {
        name,
        reason,
        extension: text(each.extension),
        documentId: text(each.documentId),
        pages: Array.isArray(each.pages)
          ? each.pages.filter((page): page is number => Number.isInteger(page) && page >= 1)
          : [],
      },
    ];
  });
}

/**
 * Records what a check request's scope couldn't read as it starts, and drops
 * what earlier requests recorded once no check's mark names them (`requests`,
 * the requests the Mind's marks name now), in one Yjs transaction.
 */
export function recordCheckUnread(
  doc: Y.Doc,
  request: string,
  files: readonly UnreadFile[],
  requests: ReadonlySet<string>,
): void {
  const unread = unreadOf(doc);
  doc.transact(() => {
    for (const key of [...unread.keys()]) {
      if (key !== request && !requests.has(key)) unread.delete(key);
    }
    if (files.length === 0) return;
    // Without a file's path, which is this computer's, not the Mind's.
    unread.set(
      request,
      files.map(({ name, reason, extension, documentId, pages }) => ({
        name,
        reason,
        extension,
        documentId,
        pages: [...pages],
      })),
    );
  });
}

/**
 * The files that couldn't be read when these requests' checks started, each
 * once: a Document by its id, any other file by its name (two of one name in
 * one request stay two).
 */
export function checkUnreadOf(doc: Y.Doc, requests: Iterable<string>): UnreadFile[] {
  const unread = unreadOf(doc);
  const found = new Map<string, UnreadFile>();
  for (const request of new Set(requests)) {
    const seen = new Map<string, number>();
    for (const file of keptFiles(unread.get(request))) {
      const key = file.documentId ?? file.name;
      const nth = seen.get(key) ?? 0;
      seen.set(key, nth + 1);
      const id = nth === 0 ? key : `${key}#${nth}`;
      if (!found.has(id)) found.set(id, { id, ...file });
    }
  }
  return [...found.values()];
}

/** Calls `listener` whenever what the Mind's checks couldn't read may have changed. Returns how to stop. */
export function observeCheckUnread(doc: Y.Doc, listener: () => void): () => void {
  const unread = unreadOf(doc);
  unread.observe(listener);
  return () => unread.unobserve(listener);
}
