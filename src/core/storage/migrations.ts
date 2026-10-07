import type { Database } from "./database";

/**
 * Schema migrations. The `schema_migrations` table records each one applied, so
 * a migration with a lower number that lands after a higher one (tickets built
 * in parallel reserve numbers up front) still runs. Never edit a migration that
 * has shipped; add a new one.
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
  {
    version: 4,
    description: "Chat providers and data-flow consent",
    sql: `
      -- Chat provider settings without secrets: API keys live in the secrets file (ADR-0003).
      CREATE TABLE chat_providers (
        id TEXT PRIMARY KEY NOT NULL,
        kind TEXT NOT NULL,
        base_url TEXT, -- NULL for OpenAI, Anthropic and Google
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX chat_providers_by_server ON chat_providers (kind, coalesce(base_url, ''))
        WHERE deleted_at IS NULL;

      -- The User's decision on each external data flow, per service.
      CREATE TABLE data_flow_consents (
        id TEXT PRIMARY KEY NOT NULL,
        flow TEXT NOT NULL,
        service_id TEXT NOT NULL,
        service_name TEXT NOT NULL,
        decision TEXT NOT NULL CHECK (decision IN ('accepted', 'declined')),
        data_kinds TEXT NOT NULL, -- JSON array of the kinds of data accepted
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX data_flow_consents_by_flow ON data_flow_consents (flow, service_id)
        WHERE deleted_at IS NULL;
    `,
  },
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
  {
    version: 10,
    description: "Segmented keyword index, Passage embeddings and processing versions (ADR-0009)",
    sql: `
      -- The keyword index holds each live Passage's words as segmented by
      -- Intl.Segmenter (the Document's name, then the Passage's text), joined
      -- by spaces for the default unicode61 tokenizer. The core writes the rows,
      -- since the segmenting happens in JavaScript; rowid is passages.seq. Only
      -- the index is stored (content = ''), and the triggers remove a Passage's
      -- row when it is deleted. It replaces migration 3's provisional trigram index.
      DROP TRIGGER passages_fts_insert;
      DROP TRIGGER passages_fts_soft_delete;
      DROP TRIGGER passages_fts_delete;
      DROP TABLE passages_fts;
      CREATE VIRTUAL TABLE passages_fts USING fts5 (
        text,
        content = '',
        contentless_delete = 1,
        tokenize = 'unicode61 remove_diacritics 2'
      );
      CREATE TRIGGER passages_fts_soft_delete AFTER UPDATE OF deleted_at ON passages
      WHEN old.deleted_at IS NULL AND new.deleted_at IS NOT NULL BEGIN
        DELETE FROM passages_fts WHERE rowid = old.seq;
      END;
      CREATE TRIGGER passages_fts_delete AFTER DELETE ON passages WHEN old.deleted_at IS NULL BEGIN
        DELETE FROM passages_fts WHERE rowid = old.seq;
      END;

      -- The Passage's vector from the embedding model in documents.embedding_model:
      -- little-endian float32s, L2-normalised. NULL until it is embedded.
      ALTER TABLE passages ADD COLUMN embedding BLOB;

      -- Which version of the processing pipeline (text normalisation, Passage
      -- sizes, keyword indexing) built a Document's Passages. Documents from an
      -- older version are processed again at startup. Version 1 is #25's.
      ALTER TABLE documents ADD COLUMN processing_version INTEGER NOT NULL DEFAULT 1;
      -- The embedding model the Document's Passage vectors come from, once embedding starts.
      ALTER TABLE documents ADD COLUMN embedding_model TEXT;
    `,
  },
  {
    version: 11,
    description: "Each Document's page text, for the Citation check (#30)",
    sql: `
      -- The text of each page of a Document as its Passages were built from it:
      -- extracted, with running headers, footers and page numbers removed. The
      -- Citation check looks for a quote in the pages a Citation names, so it
      -- needs the pages themselves, not the Passages, which overlap and don't
      -- mark where a page ends. Stored rather than extracted again: the check
      -- runs when an Answer finishes, and must see exactly this text. page is
      -- 1-based, and NULL for the one row of a Document without pages (TXT,
      -- Markdown). Like Passages, rows are replaced when the Document is
      -- processed again, and deleted with it.
      CREATE TABLE document_pages (
        id TEXT PRIMARY KEY NOT NULL,
        document_id TEXT NOT NULL REFERENCES documents (id),
        page INTEGER,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE INDEX document_pages_by_document ON document_pages (document_id, page)
        WHERE deleted_at IS NULL;
    `,
  },
  {
    version: 12,
    description: "Tags, the Tags on each Document, and automatic tagging's state",
    sql: `
      -- Labels with a short description. preset is the key of the preset Tag a
      -- row was created as on first run (e.g. 'paper'), NULL for the User's own;
      -- it outlives edits, so a later sync can tell two devices' presets apart.
      -- Names are unique among live Tags, ignoring case: the core checks it,
      -- since a later sync may bring two.
      CREATE TABLE tags (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        preset TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE INDEX tags_by_name ON tags (name COLLATE NOCASE) WHERE deleted_at IS NULL;

      -- The Tags each Document carries: one live row per Document and Tag.
      -- source is 'automatic' (automatic tagging applied it, and may take it
      -- away again) or 'user' (the User added it). Taking a Tag off marks the
      -- row deleted; when the User does, the row's source becomes 'user'. So a
      -- Document and Tag with any 'user' row, live or deleted, are the User's:
      -- automatic tagging never adds, keeps or removes that Tag there again.
      -- confidence (0 to 1) and needs_review are set by taggers that give a
      -- probability; the chat model doesn't. No foreign keys: a later sync may
      -- deliver a link before its Document or Tag.
      CREATE TABLE document_tags (
        id TEXT PRIMARY KEY NOT NULL,
        document_id TEXT NOT NULL,
        tag_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK (source IN ('automatic', 'user')),
        confidence REAL,
        needs_review INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX document_tags_by_pair ON document_tags (document_id, tag_id)
        WHERE deleted_at IS NULL;
      CREATE INDEX document_tags_by_tag ON document_tags (tag_id) WHERE deleted_at IS NULL;
      CREATE INDEX document_tags_by_user ON document_tags (document_id, tag_id)
        WHERE source = 'user';

      -- Where automatic tagging is for each Document, separately from status
      -- (a Document is searchable once ready, tagged or not): pending |
      -- waiting-for-provider | tagging | tagged | failed, checked in code. A
      -- failure keeps the provider's error kind and message. Documents from
      -- before this migration start pending, so they get tagged too.
      ALTER TABLE documents ADD COLUMN tagging_status TEXT NOT NULL DEFAULT 'pending';
      ALTER TABLE documents ADD COLUMN tagging_error_kind TEXT;
      ALTER TABLE documents ADD COLUMN tagging_error_message TEXT;
    `,
  },
  {
    version: 17,
    description: "The size of each Document's vectors, beside their embedding model (#32)",
    sql: `
      -- With a choice of embedding providers (#32), documents.embedding_model
      -- names the provider and server as well as the model (the built-in
      -- model keeps its id), and this records how many float32s each of its
      -- Passages' vectors has: set with the first vector. Search compares a
      -- query only with vectors of the same model and size, so vectors from
      -- different models are never mixed. Every vector so far is the built-in
      -- model's, of 384.
      ALTER TABLE documents ADD COLUMN embedding_dimensions INTEGER;
      UPDATE documents SET embedding_dimensions = 384
        WHERE embedding_model = 'multilingual-e5-small-int8';
    `,
  },
];

/**
 * Brings the database up to the latest schema: every migration in the list that
 * isn't recorded as applied runs, in version order, each in its own transaction.
 * A database recording a migration this build doesn't know was written by a
 * newer version of IncarnaMind, and is refused.
 */
export function migrate(db: Database, list: readonly Migration[] = migrations): void {
  list.forEach((migration, index) => {
    const previous = list[index - 1];
    if (previous && migration.version <= previous.version) {
      throw new Error(
        `Migration ${migration.version} comes after ${previous.version}: versions must increase.`,
      );
    }
  });

  const applied = appliedVersions(db, list);
  const known = new Set(list.map((migration) => migration.version));
  const unknown = [...applied].filter((version) => !known.has(version));
  if (unknown.length > 0) {
    throw new Error(
      `The database was written by a newer version of IncarnaMind (it has schema migration ${Math.max(...unknown)}, which this version doesn't know).`,
    );
  }

  for (const migration of list) {
    if (applied.has(migration.version)) continue;
    db.transaction(() => {
      db.exec(migration.sql);
      db.run("INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", [
        migration.version,
        new Date().toISOString(),
      ]);
      // Kept for tools that read it: the highest migration applied.
      const highest = db.get<{ v: number }>("SELECT max(version) AS v FROM schema_migrations")?.v;
      db.exec(`PRAGMA user_version = ${highest ?? 0}`);
    });
  }
}

/**
 * The versions recorded as applied. Creates the bookkeeping table on first use.
 * It is local to this device and never synced, so the sync-ready rules don't apply.
 *
 * Databases from before the table existed recorded only their highest version in
 * `PRAGMA user_version`: every listed migration up to it counts as applied.
 */
function appliedVersions(db: Database, list: readonly Migration[]): Set<number> {
  const exists = db.get(
    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'",
  );
  if (!exists) {
    db.transaction(() => {
      db.exec(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY NOT NULL, applied_at TEXT NOT NULL) STRICT",
      );
      const legacy = db.get<{ user_version: number }>("PRAGMA user_version")?.user_version ?? 0;
      const at = new Date().toISOString();
      for (const migration of list) {
        if (migration.version > legacy) break;
        db.run("INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", [
          migration.version,
          at,
        ]);
      }
      // A legacy database at a version this build doesn't list was written by a newer build.
      if (legacy > (list.at(-1)?.version ?? 0)) {
        db.run("INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", [legacy, at]);
      }
    });
  }
  return new Set(
    db.all<{ version: number }>("SELECT version FROM schema_migrations").map((row) => row.version),
  );
}
