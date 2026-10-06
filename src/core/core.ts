import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { CoreAdapters } from "./adapters";
import type { CoreApi } from "./api";
import { createMinds } from "./minds";
import { createSettings } from "./settings";
import { migrate, openDatabase } from "./storage";

export const DATABASE_FILE = "incarnamind.db";

/** The core as its host sees it: the public interface plus lifecycle. */
export interface Core extends CoreApi {
  /** Closes the database. The core can't be used afterwards. Safe to call twice. */
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
  const minds = createMinds(db, now);
  const settings = createSettings(db, now, adapters.systemLanguages);

  // Async on purpose: the renderer reaches these over IPC, and a future hosted core may be remote.
  return {
    createMind: async (input) => minds.create(input),
    listMinds: async () => minds.list(),
    getSettings: async () => settings.get(),
    updateSettings: async (patch) => settings.update(patch),
    close: () => db.close(),
  };
}
