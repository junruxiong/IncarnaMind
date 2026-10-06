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
  {
    version: 2,
    description: "Mind content, stored as Yjs updates",
    sql: `
      -- A Mind's content is one Yjs document (ADR-0003), stored as the updates
      -- that built it, in insertion order. Compaction replaces a Mind's rows with
      -- one row holding the merged state; that row carries the same content, so
      -- the merged rows are removed outright. Deleting a Mind marks its rows
      -- deleted. No foreign key: a later sync may deliver rows before their Mind.
      CREATE TABLE mind_updates (
        id TEXT PRIMARY KEY NOT NULL,
        mind_id TEXT NOT NULL,
        data BLOB NOT NULL, -- a Yjs update, v1 encoding
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE INDEX mind_updates_by_mind ON mind_updates (mind_id) WHERE deleted_at IS NULL;
    `,
  },
  {
    version: 3,
    description: "Documents, their Passages and a keyword index over Passage text",
    sql: `
      -- Files the User added. The copy in the data folder is named by content_hash.
      -- status: queued | extracting | ready | failed | no-text (checked in code, so later tickets can add states).
      CREATE TABLE documents (
        id TEXT PRIMARY KEY NOT NULL,
        content_hash TEXT NOT NULL, -- SHA-256 of the file, hex
        name TEXT NOT NULL,
        kind TEXT NOT NULL, -- pdf | text | markdown
        size INTEGER NOT NULL,
        page_count INTEGER,
        status TEXT NOT NULL,
        failure_reason TEXT,
        failure_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      -- The same file is one Document (ADR-0003).
      CREATE UNIQUE INDEX documents_by_content_hash ON documents (content_hash) WHERE deleted_at IS NULL;
      CREATE INDEX documents_by_created_at ON documents (created_at) WHERE deleted_at IS NULL;

      -- Spans of a Document's text that can be retrieved and cited. Never edited:
      -- processing a Document again would replace them. Pages are 1-based and NULL
      -- for Documents without pages; window_from and window_to are the positions of
      -- the first and last Passage in this Passage's sliding window.
      CREATE TABLE passages (
        seq INTEGER PRIMARY KEY, -- local key for the keyword index only: rowids not declared like this can change on VACUUM
        id TEXT NOT NULL UNIQUE,
        document_id TEXT NOT NULL REFERENCES documents (id),
        position INTEGER NOT NULL,
        page_from INTEGER,
        page_to INTEGER,
        window_from INTEGER NOT NULL,
        window_to INTEGER NOT NULL,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE INDEX passages_by_document ON passages (document_id, position) WHERE deleted_at IS NULL;

      -- PROVISIONAL (ticket #21 decides): the trigram tokenizer matches any substring
      -- of 3 or more characters, so Chinese works without word segmentation.
      -- The index holds live Passages only: soft-deleting one removes it.
      CREATE VIRTUAL TABLE passages_fts USING fts5 (
        text,
        content = 'passages',
        content_rowid = 'seq',
        tokenize = 'trigram remove_diacritics 1'
      );
      CREATE TRIGGER passages_fts_insert AFTER INSERT ON passages WHEN new.deleted_at IS NULL BEGIN
        INSERT INTO passages_fts (rowid, text) VALUES (new.seq, new.text);
      END;
      CREATE TRIGGER passages_fts_soft_delete AFTER UPDATE OF deleted_at ON passages
      WHEN old.deleted_at IS NULL AND new.deleted_at IS NOT NULL BEGIN
        INSERT INTO passages_fts (passages_fts, rowid, text) VALUES ('delete', old.seq, old.text);
      END;
      CREATE TRIGGER passages_fts_delete AFTER DELETE ON passages WHEN old.deleted_at IS NULL BEGIN
        INSERT INTO passages_fts (passages_fts, rowid, text) VALUES ('delete', old.seq, old.text);
      END;
    `,
  },
  // Versions 4 and 5 are reserved for other tickets: the runner allows gaps.
  {
    version: 6,
    description: "Folders, and the Folder each Document is filed in",
    sql: `
      -- Folders the User files Documents in by hand. They nest with no depth
      -- limit: parent_id is NULL at the top level. No foreign key: a later sync
      -- may deliver a Folder before its parent. Deleting a Folder marks it and
      -- its sub-Folders deleted.
      CREATE TABLE folders (
        id TEXT PRIMARY KEY NOT NULL,
        parent_id TEXT,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE INDEX folders_by_parent ON folders (parent_id) WHERE deleted_at IS NULL;

      -- A Document is in at most one Folder, so the Folder is a column on the
      -- Document rather than a link table. NULL means unfiled. Filing a Document
      -- moves its updated_at, like any other change to it.
      ALTER TABLE documents ADD COLUMN folder_id TEXT;
      CREATE INDEX documents_by_folder ON documents (folder_id) WHERE deleted_at IS NULL;
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
