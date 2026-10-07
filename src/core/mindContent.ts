import { randomUUID } from "node:crypto";
import * as Y from "yjs";
import { MIND_CONTENT_FIELD } from "./api";
import { InvalidInputError } from "./errors";
import type { Database } from "./storage";

/** The origin of the core's own changes to a Mind's document. */
const CORE_ORIGIN = Symbol("the core");

export type MindContent = ReturnType<typeof createMindContent>;

/**
 * Once a Mind has this many stored updates, they are merged into one. The same
 * default as y-leveldb's: compaction rewrites the whole document, so it
 * shouldn't run every few keystrokes. Loading or closing a Mind also compacts it.
 */
export const COMPACT_AFTER_UPDATES = 500;

interface LoadedMind {
  doc: Y.Doc;
  /** Rows stored for this Mind, a compacted one included. */
  storedRows: number;
  /** Updates the document emitted that aren't stored yet. */
  outbox: Uint8Array[];
}

/**
 * The authoritative Yjs document of each Mind being edited (ADR-0003).
 *
 * Every change to a document, whoever made it, is stored as one row and then
 * reported through `onStored`. A document is loaded on first use and stays in
 * memory until `close`, `remove` or `closeAll`. Callers check that the Mind exists.
 */
export function createMindContent(
  db: Database,
  now: () => string,
  onStored: (mindId: string, update: Uint8Array) => void,
) {
  const loaded = new Map<string, LoadedMind>();

  const insert = (mindId: string, data: Uint8Array, at: string) => {
    db.run(
      "INSERT INTO mind_updates (id, mind_id, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
      [randomUUID(), mindId, data, at, at],
    );
  };

  /** Replaces the Mind's stored rows with one row holding the document's whole state. */
  const compact = (mindId: string, mind: LoadedMind) => {
    if (mind.storedRows <= 1) return;
    const state = Y.encodeStateAsUpdate(mind.doc);
    db.transaction(() => {
      db.run("DELETE FROM mind_updates WHERE mind_id = ? AND deleted_at IS NULL", [mindId]);
      insert(mindId, state, now());
    });
    mind.storedRows = 1;
  };

  const unload = (mindId: string) => {
    loaded.get(mindId)?.doc.destroy();
    loaded.delete(mindId);
  };

  /** A new document holding what is stored for the Mind, and how many rows that took. */
  const fromStorage = (mindId: string) => {
    const rows = db.all<{ data: Uint8Array }>(
      "SELECT data FROM mind_updates WHERE mind_id = ? AND deleted_at IS NULL ORDER BY rowid",
      [mindId],
    );
    const doc = new Y.Doc();
    doc.transact(() => {
      for (const row of rows) Y.applyUpdate(doc, row.data);
    });
    return { doc, rows: rows.length };
  };

  const load = (mindId: string): LoadedMind => {
    const existing = loaded.get(mindId);
    if (existing) return existing;

    const { doc, rows } = fromStorage(mindId);
    const mind: LoadedMind = { doc, storedRows: rows, outbox: [] };
    // Listening only after loading, so the stored rows aren't stored again. The
    // listener must not throw: Yjs calls it while finishing a transaction.
    doc.on("update", (update: Uint8Array) => {
      mind.outbox.push(update);
    });
    loaded.set(mindId, mind);
    compact(mindId, mind);
    return mind;
  };

  /** Stores and reports what the document emitted since the last flush. */
  const flush = (mindId: string, mind: LoadedMind) => {
    if (mind.outbox.length === 0) return;
    const updates = mind.outbox.splice(0);
    try {
      const at = now();
      db.transaction(() => {
        for (const update of updates) insert(mindId, update, at);
      });
    } catch (error) {
      // Memory is now ahead of storage: drop the document, so the next use reloads what was stored.
      unload(mindId);
      throw error;
    }
    mind.storedRows += updates.length;
    for (const update of updates) onStored(mindId, update);
    if (mind.storedRows >= COMPACT_AFTER_UPDATES) compact(mindId, mind);
  };

  return {
    /** The Mind's whole document, encoded as one update. */
    state(mindId: string): Uint8Array {
      return Y.encodeStateAsUpdate(load(mindId).doc);
    },

    apply(mindId: string, update: unknown): void {
      if (!(update instanceof Uint8Array) || update.byteLength === 0) {
        throw new InvalidInputError("A Mind update must be a non-empty Uint8Array.");
      }
      const mind = load(mindId);
      let failure: unknown;
      try {
        Y.applyUpdate(mind.doc, update);
      } catch (error) {
        failure = error;
      }
      // Whatever Yjs took in is in the document now, so it is stored even if the rest was malformed.
      flush(mindId, mind);
      if (failure !== undefined) {
        throw new InvalidInputError("That isn't a valid Yjs update.", { cause: failure });
      }
    },

    /**
     * Reads the Mind's Blocks. `look` must not change them, nor keep them: the
     * document can be dropped from memory, and loaded again, between calls.
     */
    read<T>(mindId: string, look: (blocks: Y.XmlFragment) => T): T {
      return look(load(mindId).doc.getXmlFragment(MIND_CONTENT_FIELD));
    },

    /**
     * Reads the Mind's Blocks as `read` does, but a Mind that isn't loaded is
     * read from what is stored and left unloaded, writing nothing.
     */
    peek<T>(mindId: string, look: (blocks: Y.XmlFragment) => T): T {
      const mind = loaded.get(mindId);
      if (mind) return look(mind.doc.getXmlFragment(MIND_CONTENT_FIELD));
      const { doc } = fromStorage(mindId);
      try {
        return look(doc.getXmlFragment(MIND_CONTENT_FIELD));
      } finally {
        doc.destroy();
      }
    },

    /**
     * Changes the Mind's Blocks in one Yjs transaction (the core's own edits,
     * e.g. writing an Answer), then stores and reports the change like any
     * client's. `change` must not keep the Blocks, as for `read`.
     */
    edit<T>(mindId: string, change: (blocks: Y.XmlFragment) => T): T {
      const mind = load(mindId);
      let result: T | undefined;
      try {
        mind.doc.transact(() => {
          result = change(mind.doc.getXmlFragment(MIND_CONTENT_FIELD));
        }, CORE_ORIGIN);
      } finally {
        flush(mindId, mind);
      }
      return result as T;
    },

    /** Compacts the Mind and drops its document from memory. Does nothing if it isn't loaded. */
    close(mindId: string): void {
      const mind = loaded.get(mindId);
      if (!mind) return;
      compact(mindId, mind);
      unload(mindId);
    },

    /** Marks the Mind's content deleted. Run it in the same transaction as deleting the Mind. */
    remove(mindId: string, at: string): void {
      unload(mindId);
      db.run(
        "UPDATE mind_updates SET deleted_at = ?, updated_at = ? WHERE mind_id = ? AND deleted_at IS NULL",
        [at, at, mindId],
      );
    },

    /** Drops every document from memory. Everything is stored already. */
    closeAll(): void {
      for (const mindId of [...loaded.keys()]) unload(mindId);
    },
  };
}
