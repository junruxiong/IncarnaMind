import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { CoreAdapters } from "./adapters";
import type { CoreApi, CoreEventSource, Unsubscribe } from "./api";
import { createDocuments } from "./documents";
import { type AnyEventListener, createEventHub } from "./events";
import { createMinds } from "./minds";
import { createSettings } from "./settings";
import { migrate, openDatabase } from "./storage";

export const DATABASE_FILE = "incarnamind.db";

/** The core as its host sees it: the public interface (methods and events) plus host-only hooks. */
export interface Core extends CoreApi, CoreEventSource {
  /** Every event the core emits, for the host to forward to the UI. */
  onAnyEvent(listener: AnyEventListener): Unsubscribe;
  /** Closes the database and drops all listeners. The core can't be used afterwards. Safe to call twice. */
  close(): void;
}

/**
 * Opens (or creates) the data folder and its database, brings the schema up to
 * date, and returns the core. The host passes in every platform capability.
 */
export function createCore(adapters: CoreAdapters): Core {
  const { dataDir } = adapters.paths;
  mkdirSync(dataDir, { recursive: true });

  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    migrate(db);
  } catch (error) {
    db.close();
    throw error;
  }

  const now = () => (adapters.now?.() ?? new Date()).toISOString();
  const events = createEventHub();
  const minds = createMinds(db, now);
  const settings = createSettings(db, now, adapters.systemLanguages);
  let documents: ReturnType<typeof createDocuments>;
  try {
    documents = createDocuments({
      db,
      dataDir,
      now,
      emitStatus: (document) => events.emit("document.status", document),
    });
  } catch (error) {
    db.close();
    throw error;
  }

  // Async on purpose: the renderer reaches these over IPC, and a future hosted core may be remote.
  return {
    createMind: async (input) => minds.create(input),
    listMinds: async () => minds.list(),
    getSettings: async () => settings.get(),
    updateSettings: async (patch) => {
      const updated = settings.update(patch);
      events.emit("settings.changed", updated);
      return updated;
    },
    addDocuments: (paths) => documents.add(paths),
    listDocuments: async () => documents.list(),
    renameDocument: async (id, name) => documents.rename(id, name),
    deleteDocument: (id) => documents.delete(id),
    searchPassages: async (query, limit) => documents.search(query, limit),
    on: (event, listener) => events.on(event, listener),
    onAnyEvent: (listener) => events.onAny(listener),
    close: () => {
      documents.close();
      events.clear();
      db.close();
    },
  };
}
