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
    version: 14,
    description: "Connectors",
    sql: `
      -- Connectors (MCP servers). transport: 'stdio' for a program on this
      -- computer (remote ones come later), checked in code. config is JSON
      -- without secrets: for stdio, { command, args, env }, where env lists
      -- only the names of the environment variables; their values are in the
      -- keychain, never here. Names are unique among live Connectors, ignoring case.
      CREATE TABLE connectors (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL,
        transport TEXT NOT NULL,
        config TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX connectors_by_name ON connectors (name COLLATE NOCASE)
        WHERE deleted_at IS NULL;
    `,
  },
  {
    version: 15,
    description: "Skills (#40)",
    sql: `
      -- Skills the User imported. Their files are in the data folder under
      -- skills/<id>/, and this row holds what the app shows and lists without
      -- reading them: the frontmatter fields, and files, a JSON array of
      -- { path, size, script } with SKILL.md first. Importing a Skill of the
      -- same name again updates the row in place, so it keeps its id. Removing
      -- one marks it deleted; its folder goes once nothing uses it. Names are
      -- unique among live Skills: the core checks it, since a later sync may
      -- bring two. enabled is 1 for on.
      CREATE TABLE skills (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        license TEXT,
        compatibility TEXT,
        files TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE INDEX skills_by_name ON skills (name) WHERE deleted_at IS NULL;
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
  {
    version: 19,
    description: "Built-in Skills (#42)",
    sql: `
      -- Skills that ship with the app. built_in is 1 for one the core
      -- installed from the app's copy at startup, 0 for one the User imported
      -- or duplicated. built_in_digest is the SHA-256 of the app's files it was
      -- installed from: a start with other files (a newer version of the app)
      -- updates it in place, keeping its id and whether it's on. Removing one
      -- marks the row deleted, like any Skill; a deleted built-in row of a name
      -- is how later starts know the User removed it, so it isn't installed again.
      ALTER TABLE skills ADD COLUMN built_in INTEGER NOT NULL DEFAULT 0;
      ALTER TABLE skills ADD COLUMN built_in_digest TEXT;
    `,
  },
  {
    version: 20,
    description: "Approval policies (#38)",
    sql: `
      -- What the User chose about asking before something runs, replacing the
      -- default. subject_kind: 'tool' (one Tool of one Connector) or
      -- 'skill-script' (the scripts of one Skill, #41), checked in code.
      -- subject_id: for 'tool', the Connector's id and the Tool's name as the
      -- Connector gives it, joined by ':' (a UUID has none); for
      -- 'skill-script', the Skill's id. policy: 'always' (always allow, or
      -- always run) or 'ask' (ask every time, even for a Tool its Connector
      -- says only reads), checked in code. One live row per subject: changing
      -- the policy updates it, revoking it marks it deleted. No foreign keys: a
      -- later sync may deliver a policy before its Connector or Skill.
      CREATE TABLE approval_policies (
        id TEXT PRIMARY KEY NOT NULL,
        subject_kind TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        policy TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX approval_policies_by_subject ON approval_policies (subject_kind, subject_id)
        WHERE deleted_at IS NULL;
    `,
  },
  {
    version: 21,
    description: "Documents indexed in place: Linked folders, file paths and versions (ADR-0010)",
    sql: `
      -- Folders on the User's computer that IncarnaMind keeps in sync, read
      -- only: every supported file in one, at any depth, is a Document. path is
      -- absolute, with symbolic links resolved. layout is how the sidebar shows
      -- it: 'tree' (its sub-folders as Folders) or 'flat' (one list, for
      -- Zotero-style folders of one file each), NULL until its first scan
      -- suggests one. paused is 1 while its indexing is paused.
      -- ignore_patterns is a JSON array of extra names or relative paths to
      -- leave out, on top of hidden files, .git and node_modules (no UI yet).
      CREATE TABLE linked_folders (
        id TEXT PRIMARY KEY NOT NULL,
        path TEXT NOT NULL,
        layout TEXT,
        paused INTEGER NOT NULL DEFAULT 0,
        ignore_patterns TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX linked_folders_by_path ON linked_folders (path) WHERE deleted_at IS NULL;

      -- Folders now mirror the folders inside Linked folders, as they are on
      -- disk; the in-app Folders the User filed Documents in by hand are gone.
      -- They are marked deleted, so a Search scope naming one shows it as a
      -- deleted Folder. A Folder's id is derived from its Linked folder and its
      -- relative_path ('' for the Linked folder itself, '/' between names), so
      -- it is the same at every scan.
      UPDATE folders SET deleted_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE deleted_at IS NULL;
      ALTER TABLE folders ADD COLUMN linked_folder_id TEXT;
      ALTER TABLE folders ADD COLUMN relative_path TEXT;
      CREATE INDEX folders_by_linked_folder ON folders (linked_folder_id) WHERE deleted_at IS NULL;

      -- A Document is a file at a path, indexed where the User keeps it. The
      -- same content at two paths is two Documents, so content_hash is no
      -- longer unique. content_hash is the version whose text is indexed (the
      -- SHA-256 of the file as it was read). path is absolute: NULL only for
      -- Documents copied into the data folder before this migration, whose
      -- path the core fills in at startup (their copy in documents/, until
      -- the User links the folder the original is in). linked_folder_id is the
      -- Linked folder the file is in, NULL for a file added on its own.
      -- file_status: 'available' | 'missing' (the file is gone) |
      -- 'unavailable' (its Linked folder, or a single file's folder, can't be
      -- reached, or the file can't be read), checked in code. size and
      -- file_mtime_ms are the file's as last seen, so a scan only reads files
      -- whose size or modified time changed. Documents filed in the old
      -- Folders become unfiled.
      UPDATE documents SET folder_id = NULL WHERE folder_id IS NOT NULL;
      DROP INDEX documents_by_content_hash;
      CREATE INDEX documents_by_content_hash ON documents (content_hash) WHERE deleted_at IS NULL;
      ALTER TABLE documents ADD COLUMN path TEXT;
      ALTER TABLE documents ADD COLUMN linked_folder_id TEXT;
      ALTER TABLE documents ADD COLUMN file_status TEXT NOT NULL DEFAULT 'available';
      ALTER TABLE documents ADD COLUMN file_mtime_ms REAL;
      CREATE UNIQUE INDEX documents_by_path ON documents (path) WHERE deleted_at IS NULL;
      CREATE INDEX documents_by_linked_folder ON documents (linked_folder_id)
        WHERE deleted_at IS NULL;

      -- Each Passage and each page of text belongs to one version of its
      -- Document: the content_hash it was built from. When the file changes,
      -- the new version's Passages replace the old ones in search, but the old
      -- version's pages stay for as long as a Citation quotes that version,
      -- so its check still reads the text it quoted.
      ALTER TABLE passages ADD COLUMN content_hash TEXT;
      UPDATE passages SET content_hash =
        (SELECT d.content_hash FROM documents d WHERE d.id = passages.document_id);
      ALTER TABLE document_pages ADD COLUMN content_hash TEXT;
      UPDATE document_pages SET content_hash =
        (SELECT d.content_hash FROM documents d WHERE d.id = document_pages.document_id);
      DROP INDEX document_pages_by_document;
      CREATE INDEX document_pages_by_version ON document_pages (document_id, content_hash, page)
        WHERE deleted_at IS NULL;
    `,
  },
  {
    version: 22,
    description: "Each Document's text as Units: pages, slides, sections, rows, lines (ADR-0011)",
    sql: `
      -- Each row of document_pages is now one Unit of a version's text: a
      -- PDF's page, a deck's slide (its speaker notes included), a section of
      -- a Word or Markdown file, a block of rows of one sheet of a spreadsheet
      -- or CSV, or a block of lines of a plain-text file. page is the Unit's
      -- number, from 1: a PDF's page, a deck's slide. kind says which:
      -- 'page' | 'slide' | 'section' | 'rows' | 'lines' | 'text', checked in
      -- code. label is JSON (UnitLabel in src/shared/units.ts: a section's
      -- heading path, a block's sheet and rows…), NULL for pages. anchors is
      -- a JSON array of { start, end, target } mapping spans of the text to a
      -- slide's shapes and notes or a section's paragraphs, NULL where none
      -- are stored (pages; rows work theirs out from the text). A TXT or
      -- Markdown file stored as one text, with page NULL, becomes Unit 1 of
      -- kind 'text': the current versions are processed again into sections
      -- and lines, and the old versions Citations quote keep their one text.
      ALTER TABLE document_pages ADD COLUMN kind TEXT NOT NULL DEFAULT 'page';
      ALTER TABLE document_pages ADD COLUMN label TEXT;
      ALTER TABLE document_pages ADD COLUMN anchors TEXT;
      UPDATE document_pages SET kind = 'text', page = 1 WHERE page IS NULL;
    `,
  },
  {
    // 23 was reserved for Topics, which Folders and Tags replaced (ADR-0012); it stays unused.
    version: 24,
    description: "Each Document's creation date, read without processing it again (#53)",
    sql: `
      -- When the Document itself was created, for the Library's year: the
      -- creation date in its file's metadata (a PDF's Info dictionary or
      -- XMP, the core properties of a Word, PowerPoint or Excel file), or
      -- else a year written in its first Unit; never a modification date.
      -- ISO 8601 at the precision the file gives, with the offset from UTC
      -- it gives, so the first four characters are the year:
      -- '2019-03-04T10:30:00+01:00', '2019-03-04', or '2019' from text. NULL
      -- when nothing gives one. metadata_version is the version of the
      -- metadata read that set it (METADATA_VERSION in
      -- src/core/documents/processing.ts), 0 for none: those Documents have
      -- their metadata read in the background, one at a time, without being
      -- processed or embedded again. A new version of the file is read as
      -- it is processed.
      ALTER TABLE documents ADD COLUMN creation_date TEXT;
      ALTER TABLE documents ADD COLUMN metadata_version INTEGER NOT NULL DEFAULT 0;
      CREATE INDEX documents_by_metadata_version ON documents (metadata_version)
        WHERE deleted_at IS NULL;
    `,
  },
  {
    version: 25,
    description: "User-defined Library groups and persistent primary classifications",
    sql: `
      CREATE TABLE library_groups (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE TABLE document_groups (
        id TEXT PRIMARY KEY NOT NULL,
        document_id TEXT NOT NULL REFERENCES documents(id),
        group_id TEXT REFERENCES library_groups(id),
        source TEXT NOT NULL CHECK (source IN ('automatic', 'user')),
        status TEXT NOT NULL,
        content_hash TEXT,
        request_id TEXT NOT NULL,
        error_kind TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
      ) STRICT;
      CREATE UNIQUE INDEX document_groups_live ON document_groups(document_id)
        WHERE deleted_at IS NULL;
    `,
  },
  {
    version: 26,
    description: "Record the model and routing reason for Library classifications",
    sql: `ALTER TABLE document_groups ADD COLUMN classification_model TEXT;`,
  },
  {
    version: 27,
    description: "Tag colours: presets their own, the User's Tags in turn through the palette",
    // The palette is src/shared/tagColours.ts, in order; the preset colours are
    // src/core/tags/presets.ts. Deleted Tags are coloured too, harmlessly.
    sql: `
      ALTER TABLE tags ADD COLUMN colour TEXT NOT NULL DEFAULT 'stone';
      UPDATE tags SET colour = CASE preset
        WHEN 'paper' THEN 'violet'
        WHEN 'report' THEN 'petrol'
        WHEN 'book' THEN 'brick'
        WHEN 'contract' THEN 'indigo'
        WHEN 'invoice' THEN 'rose'
        WHEN 'slides' THEN 'orchid'
        WHEN 'notes' THEN 'taupe'
        ELSE 'stone' END
      WHERE preset IS NOT NULL;
      UPDATE tags SET colour = (
        SELECT CASE turn.n % 8
          WHEN 0 THEN 'stone' WHEN 1 THEN 'taupe' WHEN 2 THEN 'brick' WHEN 3 THEN 'rose'
          WHEN 4 THEN 'orchid' WHEN 5 THEN 'violet' WHEN 6 THEN 'indigo' ELSE 'petrol' END
        FROM (
          SELECT id, ROW_NUMBER() OVER (ORDER BY created_at, rowid) - 1 AS n
          FROM tags WHERE preset IS NULL
        ) AS turn
        WHERE turn.id = tags.id
      )
      WHERE preset IS NULL;
    `,
  },
  {
    version: 28,
    description: "Tag colours: the bright, Finder-like palette",
    // A preset still on its first colour takes its new one (src/core/tags/presets.ts);
    // any other colour takes the nearest bright hue, each a different one, so Tags that
    // differed still do. The column's default, 'stone', is never used: every insert
    // names its colour.
    sql: `
      UPDATE tags SET colour = CASE
        WHEN preset = 'paper' AND colour = 'violet' THEN 'purple'
        WHEN preset = 'report' AND colour = 'petrol' THEN 'blue'
        WHEN preset = 'book' AND colour = 'brick' THEN 'orange'
        WHEN preset = 'contract' AND colour = 'indigo' THEN 'red'
        WHEN preset = 'invoice' AND colour = 'rose' THEN 'green'
        WHEN preset = 'slides' AND colour = 'orchid' THEN 'yellow'
        WHEN preset = 'notes' AND colour = 'taupe' THEN 'gray'
        WHEN colour = 'stone' THEN 'gray'
        WHEN colour = 'taupe' THEN 'yellow'
        WHEN colour = 'brick' THEN 'orange'
        WHEN colour = 'rose' THEN 'red'
        WHEN colour = 'orchid' THEN 'purple'
        WHEN colour = 'violet' THEN 'teal'
        WHEN colour = 'indigo' THEN 'blue'
        WHEN colour = 'petrol' THEN 'green'
        ELSE colour END;
    `,
  },
  {
    version: 29,
    description: "Chat providers name their provider in the catalog, and its endpoint",
    // The catalog is src/core/providers/catalog/providers.ts: each kind saved so far is one
    // of its providers, except the ChatGPT plan, which has none. A hosted provider is on its
    // one endpoint, "global"; a server keeps its URL in base_url. Keys stay in the keychain
    // under the row's id, which doesn't change.
    sql: `
      ALTER TABLE chat_providers ADD COLUMN catalog_id TEXT;
      ALTER TABLE chat_providers ADD COLUMN endpoint TEXT;
      UPDATE chat_providers SET catalog_id = kind
        WHERE kind IN ('openai', 'anthropic', 'google', 'openai-compatible', 'ollama');
      UPDATE chat_providers SET endpoint = 'global'
        WHERE kind IN ('openai', 'anthropic', 'google');
      -- A provider is one per catalog provider and server, as it was one per kind and server.
      DROP INDEX chat_providers_by_server;
      CREATE UNIQUE INDEX chat_providers_by_server
        ON chat_providers (coalesce(catalog_id, kind), coalesce(base_url, ''))
        WHERE deleted_at IS NULL;
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
