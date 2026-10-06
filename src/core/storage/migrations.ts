import type { Database } from "./database";

/**
 * Schema migrations, applied in order. `PRAGMA user_version` records the last
 * one applied. Never edit a migration that has shipped; add a new one.
 *
 * Sync-ready rules (ADR-0003), for every table:
 * - `id` is a random UUID generated on the device;
 * - `created_at` and `updated_at` are ISO 8601 UTC timestamps;
 * - `deleted_at` marks a soft delete, and queries ignore rows where it is set.
 */
export interface Migration {
  version: number;
  description: string;
  sql: string;
}

export const migrations: readonly Migration[] = [
  {
    version: 1,
    description: "Minds, per-User settings and per-device settings",
    sql: `
      CREATE TABLE minds (
        id TEXT PRIMARY KEY NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE INDEX minds_by_updated_at ON minds (updated_at) WHERE deleted_at IS NULL;

      -- Per-User settings: will sync across the User's devices.
      CREATE TABLE user_settings (
        id TEXT PRIMARY KEY NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL, -- JSON
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX user_settings_by_key ON user_settings (key) WHERE deleted_at IS NULL;

      -- Per-device settings: never sync.
      CREATE TABLE device_settings (
        id TEXT PRIMARY KEY NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL, -- JSON
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX device_settings_by_key ON device_settings (key) WHERE deleted_at IS NULL;
    `,
  },
];

/** Brings the database up to the latest schema. Each migration runs in its own transaction. */
export function migrate(db: Database, list: readonly Migration[] = migrations): void {
  const current = db.get<{ user_version: number }>("PRAGMA user_version")?.user_version ?? 0;
  const latest = list.at(-1)?.version ?? 0;
  if (current > latest) {
    throw new Error(
      `The database was written by a newer version of IncarnaMind (schema ${current}; this version supports up to ${latest}).`,
    );
  }
  for (const migration of list) {
    if (migration.version <= current) continue;
    db.transaction(() => {
      db.exec(migration.sql);
      db.exec(`PRAGMA user_version = ${migration.version}`);
    });
  }
}
