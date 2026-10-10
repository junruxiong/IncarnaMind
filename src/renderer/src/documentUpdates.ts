/**
 * The core's per-Document events, gathered and applied together (#156).
 * Organize, or indexing a big Linked folder, sends thousands of them a
 * minute; applied one by one, each copied the whole Document list and drew
 * every row again. Here they wait for a short interval, the latest copy of
 * each Document winning, and then change the store once, in one pass over
 * the list. Pure: no store, no bridge, so the tests import it.
 */

import type { Document, DocumentGroupAssignment, LibrarySnapshot } from "../../core/api";

/** What changed since the last time the store was updated. */
export interface DocumentChanges {
  /** Each Document added or changed: its latest copy, in the order each was first seen. */
  readonly upserted: ReadonlyMap<string, Document>;
  /**
   * Documents removed. One added again after its removal is in `upserted`
   * too: it leaves its place in the list and comes back first, as a new one.
   */
  readonly removed: ReadonlySet<string>;
  /** Each Document's Library assignment that changed: the latest. */
  readonly assignments: ReadonlyMap<string, DocumentGroupAssignment>;
}

/**
 * The list with the changes made, as making them one at a time would have:
 * a changed Document in its place, a new one first (the newest first), a
 * removed one gone. Every other Document stays the same object; with nothing
 * to change, the list is the same array.
 */
export function applyDocumentChanges(
  documents: readonly Document[],
  upserted: ReadonlyMap<string, Document>,
  removed: ReadonlySet<string>,
): Document[] {
  if (upserted.size === 0 && removed.size === 0) return documents as Document[];
  const placed = new Set<string>();
  const kept: Document[] = [];
  for (const each of documents) {
    if (removed.has(each.id)) continue;
    const changed = upserted.get(each.id);
    if (changed) placed.add(each.id);
    kept.push(changed ?? each);
  }
  const added = [...upserted.values()].filter((each) => !placed.has(each.id)).reverse();
  return added.length > 0 ? [...added, ...kept] : kept;
}

/**
 * The Library with these Documents' assignments in place of theirs, and
 * those it didn't have yet at the end. The snapshot, and every assignment
 * that didn't change, stay the same objects when nothing changes.
 */
export function applyAssignments(
  library: LibrarySnapshot,
  assignments: ReadonlyMap<string, DocumentGroupAssignment>,
): LibrarySnapshot {
  if (assignments.size === 0) return library;
  const placed = new Set<string>();
  const next = library.assignments.map((each) => {
    const changed = assignments.get(each.documentId);
    if (!changed) return each;
    placed.add(each.documentId);
    return changed;
  });
  for (const [documentId, changed] of assignments) {
    if (!placed.has(documentId)) next.push(changed);
  }
  return { ...library, assignments: next };
}

/** When to update the store: a timer, so the tests can drive it. */
export interface UpdateClock {
  now(): number;
  setTimeout(run: () => void, ms: number): unknown;
  clearTimeout(timer: unknown): void;
}

const systemClock: UpdateClock = {
  now: () => performance.now(),
  setTimeout: (run, ms) => setTimeout(run, ms),
  clearTimeout: (timer) => clearTimeout(timer as ReturnType<typeof setTimeout>),
};

/**
 * Gathers the core's per-Document events and hands them to `apply` at most
 * once per `interval` ms: the first after a quiet spell goes at once (on the
 * next task, so one change of the User's shows without waiting), and those
 * coming fast are gathered until the interval since the last is over.
 * `flush` hands over what is waiting now: a store action that changes the
 * Documents itself calls it first, so nothing older lands on top of it later.
 */
export function createDocumentUpdates(
  apply: (changes: DocumentChanges) => void,
  { interval = 100, clock = systemClock }: { interval?: number; clock?: UpdateClock } = {},
) {
  let upserted = new Map<string, Document>();
  let removed = new Set<string>();
  let assignments = new Map<string, DocumentGroupAssignment>();
  let timer: unknown = null;
  let last = Number.NEGATIVE_INFINITY;

  const flush = () => {
    if (timer !== null) {
      clock.clearTimeout(timer);
      timer = null;
    }
    if (upserted.size === 0 && removed.size === 0 && assignments.size === 0) return;
    const changes: DocumentChanges = { upserted, removed, assignments };
    upserted = new Map();
    removed = new Set();
    assignments = new Map();
    last = clock.now();
    apply(changes);
  };

  const schedule = () => {
    if (timer !== null) return;
    timer = clock.setTimeout(flush, Math.max(0, last + interval - clock.now()));
  };

  return {
    /** Documents added or changed, each carried whole. */
    upsert(documents: readonly Document[]) {
      for (const each of documents) upserted.set(each.id, each);
      if (documents.length > 0) schedule();
    },
    /** Documents removed, by id. */
    remove(ids: readonly string[]) {
      for (const id of ids) {
        upserted.delete(id);
        removed.add(id);
      }
      if (ids.length > 0) schedule();
    },
    /** Documents' Library assignments, each carried whole. */
    assign(changed: readonly DocumentGroupAssignment[]) {
      for (const each of changed) assignments.set(each.documentId, each);
      if (changed.length > 0) schedule();
    },
    flush,
  };
}
