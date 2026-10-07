/**
 * Documents: adding files, processing them into Passages off the main thread
 * (extracting on a worker thread, embedding through the embedder), renaming,
 * filing in Folders, soft-deleting, and keyword, vector and hybrid search.
 */
import { randomUUID } from "node:crypto";
import { basename, extname, isAbsolute } from "node:path";
import type {
  AddDocumentsResult,
  Document,
  DocumentFailureReason,
  DocumentKind,
  DocumentStatus,
  PassageSearchResult,
  SearchMode,
  SkippedFile,
} from "../api";
import type { EmbeddingModel } from "../embedding";
import { EmbeddingModelNotReadyError, InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { Database } from "../storage";
import { createEmbeddingQueue } from "./embedding";
import { createDocumentFiles, kindOf } from "./files";
import { keywordText } from "./keywords";
import {
  PROCESSING_VERSION,
  type ProcessedPassage,
  type ProcessingJob,
  type ProcessingResult,
} from "./processing";
import { createProcessor } from "./processor";
import { fuseRankings, HYBRID_CANDIDATES, keywordSearch, passagesBySeq } from "./search";
import { createVectorIndex } from "./vectors";

const DEFAULT_SEARCH_LIMIT = 20;
const MAX_SEARCH_LIMIT = 200;
const MAX_NAME_LENGTH = 500;
const SEARCH_MODES: readonly SearchMode[] = ["hybrid", "keyword", "vector"];

interface DocumentRow {
  id: string;
  content_hash: string;
  name: string;
  kind: string;
  size: number;
  page_count: number | null;
  status: string;
  failure_reason: string | null;
  failure_message: string | null;
  folder_id: string | null;
  created_at: string;
  updated_at: string;
}

const COLUMNS = `id, content_hash, name, kind, size, page_count, status,
  failure_reason, failure_message, folder_id, created_at, updated_at`;

function parsePaths(input: unknown): string[] {
  if (!Array.isArray(input)) throw new InvalidInputError("addDocuments expects a list of paths.");
  for (const path of input) {
    if (typeof path !== "string" || !isAbsolute(path)) {
      throw new InvalidInputError("Each Document must be given as an absolute file path.");
    }
  }
  return input;
}

function parseId(id: unknown): string {
  if (typeof id !== "string" || id === "")
    throw new InvalidInputError("A Document id is required.");
  return id;
}

function parseName(name: unknown): string {
  if (typeof name !== "string") throw new InvalidInputError("A Document's name must be text.");
  const trimmed = name.trim();
  if (!trimmed) throw new InvalidInputError("A Document's name can't be empty.");
  if (trimmed.length > MAX_NAME_LENGTH) {
    throw new InvalidInputError(
      `A Document's name can't be longer than ${MAX_NAME_LENGTH} characters.`,
    );
  }
  return trimmed;
}

/** `listDocuments` options, checked. `folderId` undefined means every Document. */
export function parseListOptions(options: unknown): {
  folderId: string | undefined;
  includeSubfolders: boolean;
} {
  if (options === undefined) return { folderId: undefined, includeSubfolders: false };
  if (!isRecord(options)) throw new InvalidInputError("listDocuments expects an object.");
  const { folderId, includeSubfolders = false } = options;
  if (folderId !== undefined && (typeof folderId !== "string" || folderId === "")) {
    throw new InvalidInputError("A Folder id must be a non-empty string.");
  }
  if (typeof includeSubfolders !== "boolean") {
    throw new InvalidInputError("includeSubfolders must be true or false.");
  }
  return { folderId, includeSubfolders };
}

/** `searchPassages` options, checked and with their defaults. */
function parseSearchOptions(options: unknown): {
  mode: SearchMode;
  limit: number;
  documentIds: string[] | undefined;
} {
  if (options === undefined) {
    return { mode: "hybrid", limit: DEFAULT_SEARCH_LIMIT, documentIds: undefined };
  }
  if (!isRecord(options)) throw new InvalidInputError("searchPassages expects an object.");
  const { mode = "hybrid", limit = DEFAULT_SEARCH_LIMIT, documentIds } = options;
  if (!SEARCH_MODES.includes(mode as SearchMode)) {
    throw new InvalidInputError(`The search mode must be one of ${SEARCH_MODES.join(", ")}.`);
  }
  if (!Number.isInteger(limit) || (limit as number) < 1 || (limit as number) > MAX_SEARCH_LIMIT) {
    throw new InvalidInputError(`The limit must be a whole number from 1 to ${MAX_SEARCH_LIMIT}.`);
  }
  if (
    documentIds !== undefined &&
    (!Array.isArray(documentIds) || !documentIds.every((id) => typeof id === "string" && id !== ""))
  ) {
    throw new InvalidInputError("documentIds must be a list of Document ids.");
  }
  return {
    mode: mode as SearchMode,
    limit: limit as number,
    documentIds: documentIds as string[] | undefined,
  };
}

/** The file name without its extension, or the whole name if that leaves nothing. */
const nameFromPath = (path: string) => basename(path, extname(path)).trim() || basename(path);

/** What the keyword index holds for a Passage: its Document's name, then its own words. */
const indexedText = (nameKeywords: string, passageKeywords: string) =>
  nameKeywords ? `${nameKeywords} ${passageKeywords}` : passageKeywords;

/** A live Document's stored file, opened for reading. */
export interface DocumentFile {
  document: Document;
  /** The file's bytes. Cancel the stream if it isn't read to the end, so the file is closed. */
  stream: ReadableStream<Uint8Array>;
}

export interface DocumentsOptions {
  db: Database;
  dataDir: string;
  now: () => string;
  /** The built-in embedding model, which embeds Passages and search queries. */
  model: EmbeddingModel;
  /** Pushes the "document.status" event. */
  emitStatus(document: Document): void;
}

export function createDocuments({ db, dataDir, now, model, emitStatus }: DocumentsOptions) {
  const files = createDocumentFiles(dataDir);
  const vectors = createVectorIndex(db, model);

  /** The share of a Document's Passages embedded so far. */
  const progressOf = (id: string): number => {
    const counts = db.get<{ total: number; embedded: number }>(
      `SELECT count(*) AS total, count(embedding) AS embedded FROM passages
       WHERE document_id = ? AND deleted_at IS NULL`,
      [id],
    );
    return counts && counts.total > 0 ? counts.embedded / counts.total : 0;
  };

  const toDocument = (row: DocumentRow): Document => ({
    id: row.id,
    name: row.name,
    kind: row.kind as DocumentKind,
    contentHash: row.content_hash,
    size: row.size,
    pageCount: row.page_count,
    status: row.status as DocumentStatus,
    progress: row.status === "embedding" ? progressOf(row.id) : null,
    failure:
      row.status === "failed"
        ? {
            reason: (row.failure_reason ?? "processing-error") as DocumentFailureReason,
            message: row.failure_message ?? "",
          }
        : null,
    folderId: row.folder_id,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  });

  const find = (id: string) =>
    db.get<DocumentRow>(`SELECT ${COLUMNS} FROM documents WHERE id = ? AND deleted_at IS NULL`, [
      id,
    ]);

  const findByHash = (contentHash: string) =>
    db.get<DocumentRow>(
      `SELECT ${COLUMNS} FROM documents WHERE content_hash = ? AND deleted_at IS NULL`,
      [contentHash],
    );

  const announce = (id: string) => {
    const row = find(id);
    if (row) emitStatus(toDocument(row));
  };

  const jobFor = (row: Pick<DocumentRow, "id" | "content_hash" | "kind">): ProcessingJob => ({
    documentId: row.id,
    contentHash: row.content_hash,
    kind: row.kind as DocumentKind,
    file: files.pathFor(row.content_hash),
  });

  /** Removes a Document file once no live Document uses it. */
  const releaseFile = async (contentHash: string) => {
    if (!findByHash(contentHash)) await files.remove(contentHash);
  };

  const setStatus = (id: string, status: DocumentStatus) =>
    db.run("UPDATE documents SET status = ?, updated_at = ? WHERE id = ?", [status, now(), id]);

  const embedding = createEmbeddingQueue({ db, now, model, vectors, announce });

  /** Adds a Passage to the keyword index, under the `seq` it was stored with. */
  const index = (seq: number, text: string) =>
    db.run("INSERT INTO passages_fts (rowid, text) VALUES (?, ?)", [BigInt(seq), text]);

  /** Stores Passages and indexes them. Run in a transaction. */
  function insertPassages(documentId: string, name: string, passages: ProcessedPassage[]): void {
    const at = now();
    const nameKeywords = keywordText(name);
    for (const passage of passages) {
      const stored = db.get<{ seq: number }>(
        `INSERT INTO passages (id, document_id, position, page_from, page_to,
           window_from, window_to, text, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         RETURNING seq`,
        [
          randomUUID(),
          documentId,
          BigInt(passage.position),
          passage.pageFrom === null ? null : BigInt(passage.pageFrom),
          passage.pageTo === null ? null : BigInt(passage.pageTo),
          BigInt(passage.windowFrom),
          BigInt(passage.windowTo),
          passage.text,
          at,
          at,
        ],
      );
      if (!stored) throw new Error("The Passage wasn't saved.");
      index(stored.seq, indexedText(nameKeywords, passage.keywords));
    }
  }

  /**
   * Writes a job's result, replacing Passages from an earlier processing,
   * unless the Document was deleted meanwhile. Returns the status it wrote, if it did.
   */
  function record(job: ProcessingJob, result: ProcessingResult): DocumentStatus | undefined {
    return db.transaction(() => {
      const row = find(job.documentId);
      if (row?.status !== "extracting") return undefined;
      const at = now();
      db.run(
        `UPDATE passages SET deleted_at = ?, updated_at = ?, embedding = NULL
         WHERE document_id = ? AND deleted_at IS NULL`,
        [at, at, job.documentId],
      );
      let status: DocumentStatus;
      if (result.outcome === "ready") {
        insertPassages(job.documentId, row.name, result.passages);
        status = model.isReady() ? "embedding" : "waiting-for-model";
      } else {
        status = result.outcome;
      }
      db.run(
        `UPDATE documents SET status = ?, page_count = ?, failure_reason = ?, failure_message = ?,
           processing_version = ?, embedding_model = ?, updated_at = ?
         WHERE id = ?`,
        [
          status,
          result.outcome === "failed" || result.pageCount === null
            ? null
            : BigInt(result.pageCount),
          result.outcome === "failed" ? result.reason : null,
          result.outcome === "failed" ? result.message : null,
          BigInt(PROCESSING_VERSION),
          status === "embedding" ? model.id : null,
          at,
          job.documentId,
        ],
      );
      return status;
    });
  }

  const processor = createProcessor({
    onStart(job) {
      if (find(job.documentId)?.status !== "queued") return;
      setStatus(job.documentId, "extracting");
      announce(job.documentId);
    },
    onResult(job, result) {
      const status = record(job, result);
      if (status === undefined) {
        void releaseFile(job.contentHash); // deleted while processing
        return;
      }
      // Any vectors from an earlier processing went with the old Passages.
      vectors.removeDocument(job.documentId);
      announce(job.documentId);
      if (status === "embedding") embedding.enqueue(job.documentId);
      if (status === "waiting-for-model") model.ensure();
    },
  });

  // Startup: tidy the folder, then pick up work a quit interrupted.
  files.prepare((contentHash) => findByHash(contentHash) !== undefined);
  // Documents processed by an older pipeline are processed again, through the usual statuses.
  db.run(
    `UPDATE documents SET status = 'queued', updated_at = ?
     WHERE deleted_at IS NULL AND processing_version < ?
       AND status IN ('ready', 'embedding', 'waiting-for-model')`,
    [now(), BigInt(PROCESSING_VERSION)],
  );
  const unfinished = db.all<DocumentRow>(
    `SELECT ${COLUMNS} FROM documents
     WHERE deleted_at IS NULL AND status IN ('queued', 'extracting')
     ORDER BY created_at, rowid`,
  );
  for (const row of unfinished) {
    if (row.status === "extracting") setStatus(row.id, "queued");
    processor.enqueue(jobFor(row));
  }
  // Embedding carries on where it stopped, or waits for the model.
  const embeddingRows = db.all<{ id: string }>(
    `SELECT id FROM documents WHERE deleted_at IS NULL AND status = 'embedding'
     ORDER BY created_at, rowid`,
  );
  for (const { id } of embeddingRows) {
    if (model.isReady()) embedding.enqueue(id);
    else setStatus(id, "waiting-for-model");
  }
  model.onReady(() => embedding.resumeWaiting());
  if (model.isReady()) embedding.resumeWaiting();
  else if (
    db.get("SELECT 1 FROM documents WHERE deleted_at IS NULL AND status = 'waiting-for-model'")
  )
    model.ensure();

  /** Most recently added first. With `folderIds`, only Documents filed in one of those Folders. */
  const list = (folderIds?: readonly string[]): Document[] => {
    const rows =
      folderIds === undefined
        ? db.all<DocumentRow>(
            `SELECT ${COLUMNS} FROM documents WHERE deleted_at IS NULL
             ORDER BY created_at DESC, rowid DESC`,
          )
        : db.all<DocumentRow>(
            `SELECT ${COLUMNS} FROM documents
             WHERE deleted_at IS NULL AND folder_id IN (SELECT value FROM json_each(?))
             ORDER BY created_at DESC, rowid DESC`,
            [JSON.stringify(folderIds)],
          );
    return rows.map(toDocument);
  };

  /** Adds one file whose kind is known. Returns undefined if it can't be read. */
  async function addFile(path: string, kind: DocumentKind): Promise<Document | undefined> {
    let imported: Awaited<ReturnType<typeof files.import>>;
    try {
      imported = await files.import(path);
    } catch {
      return undefined;
    }
    const { contentHash, size } = imported;
    const added = db.transaction(() => {
      const existing = findByHash(contentHash);
      if (existing) return { row: existing, isNew: false };
      const id = randomUUID();
      const at = now();
      db.run(
        `INSERT INTO documents (id, content_hash, name, kind, size, status, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)`,
        [id, contentHash, nameFromPath(path), kind, size, at, at],
      );
      const row = find(id);
      if (!row) throw new Error("The new Document wasn't saved.");
      return { row, isNew: true };
    });
    const document = toDocument(added.row);
    if (added.isNew) {
      emitStatus(document);
      processor.enqueue(jobFor(added.row));
    }
    return document;
  }

  /**
   * The query's vector, or null when a hybrid search has to do without:
   * the model isn't ready, or fails. A vector search throws instead.
   */
  async function queryVector(query: string, mode: SearchMode): Promise<Float32Array | null> {
    const unavailable = () => {
      if (mode === "vector") throw new EmbeddingModelNotReadyError(model.status());
      return null;
    };
    if (!(await model.load())) return unavailable();
    try {
      return await model.embedQuery(query);
    } catch {
      // The process running the model may have stopped: start it again and retry once.
      if (!(await model.load())) return unavailable();
      try {
        return await model.embedQuery(query);
      } catch (error) {
        if (mode === "vector") throw error;
        console.error(error);
        return null;
      }
    }
  }

  return {
    async add(input: unknown): Promise<AddDocumentsResult> {
      const paths = parsePaths(input);
      const documents: Document[] = [];
      const skipped: SkippedFile[] = [];
      // One at a time: keeps the order, and doesn't copy many large files at once.
      for (const path of paths) {
        const kind = kindOf(path);
        const document = kind && (await addFile(path, kind));
        if (document) documents.push(document);
        else skipped.push({ path, reason: kind ? "unreadable" : "unsupported-type" });
      }
      return { documents, skipped };
    },

    list,

    /**
     * Files a Document in a Folder, or unfiles it with null. The caller checks
     * that the Folder exists. Returns the Document and whether it moved.
     */
    move(idInput: unknown, folderId: string | null): { document: Document; moved: boolean } {
      const id = parseId(idInput);
      return db.transaction(() => {
        const row = find(id);
        if (!row) throw new NotFoundError("There is no such Document.");
        if (row.folder_id === folderId) return { document: toDocument(row), moved: false };
        db.run("UPDATE documents SET folder_id = ?, updated_at = ? WHERE id = ?", [
          folderId,
          now(),
          id,
        ]);
        const moved = find(id);
        if (!moved) throw new NotFoundError("There is no such Document.");
        return { document: toDocument(moved), moved: true };
      });
    },

    /** Unfiles every Document filed in one of `folderIds`, at `at`. Returns them, unfiled. */
    unfile(folderIds: readonly string[], at: string): Document[] {
      return db.transaction(() => {
        const filed = list(folderIds);
        db.run(
          `UPDATE documents SET folder_id = NULL, updated_at = ?
           WHERE deleted_at IS NULL AND folder_id IN (SELECT value FROM json_each(?))`,
          [at, JSON.stringify(folderIds)],
        );
        return filed.map((document) => ({ ...document, folderId: null, updatedAt: at }));
      });
    },

    /**
     * Renames a Document, and re-indexes its Passages' keywords, which include
     * the name. Its vectors keep the name they were embedded with.
     */
    rename(idInput: unknown, nameInput: unknown): Document {
      const id = parseId(idInput);
      const name = parseName(nameInput);
      if (!find(id)) throw new NotFoundError("There is no such Document.");
      db.transaction(() => {
        db.run("UPDATE documents SET name = ?, updated_at = ? WHERE id = ?", [name, now(), id]);
        const nameKeywords = keywordText(name);
        const passages = db.all<{ seq: number; text: string }>(
          "SELECT seq, text FROM passages WHERE document_id = ? AND deleted_at IS NULL",
          [id],
        );
        for (const { seq, text } of passages) {
          db.run("DELETE FROM passages_fts WHERE rowid = ?", [BigInt(seq)]);
          index(seq, indexedText(nameKeywords, keywordText(text)));
        }
      });
      const row = find(id);
      if (!row) throw new NotFoundError("There is no such Document.");
      return toDocument(row);
    },

    async delete(idInput: unknown): Promise<void> {
      const id = parseId(idInput);
      const row = find(id);
      if (!row) throw new NotFoundError("There is no such Document.");
      const at = now();
      db.transaction(() => {
        db.run("UPDATE documents SET deleted_at = ?, updated_at = ? WHERE id = ?", [at, at, id]);
        db.run(
          `UPDATE passages SET deleted_at = ?, updated_at = ?, embedding = NULL
           WHERE document_id = ? AND deleted_at IS NULL`,
          [at, at, id],
        );
      });
      vectors.removeDocument(id);
      processor.cancel(id);
      await releaseFile(row.content_hash);
    },

    /** Opens a live Document's stored file. Deleted or unknown Documents, and missing files, are refused. */
    async openFile(idInput: unknown): Promise<DocumentFile> {
      const id = parseId(idInput);
      const row = find(id);
      if (!row) throw new NotFoundError("There is no such Document.");
      let stream: ReadableStream<Uint8Array>;
      try {
        stream = await files.open(row.content_hash);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") {
          throw new NotFoundError("The Document's file is missing from the data folder.");
        }
        throw error;
      }
      return { document: toDocument(row), stream };
    },

    async search(query: unknown, options: unknown): Promise<PassageSearchResult[]> {
      if (typeof query !== "string") throw new InvalidInputError("The search query must be text.");
      const { mode, limit, documentIds } = parseSearchOptions(options);
      if (query.trim() === "") return [];
      if (mode === "keyword")
        return passagesBySeq(db, keywordSearch(db, query, limit, documentIds));

      const vector = await queryVector(query, mode);
      if (mode === "vector") {
        const hits = vector ? vectors.search(vector, limit, documentIds) : [];
        return passagesBySeq(
          db,
          hits.map((hit) => hit.seq),
        );
      }
      const keyword = keywordSearch(db, query, HYBRID_CANDIDATES, documentIds);
      const similar = vector
        ? vectors.search(vector, HYBRID_CANDIDATES, documentIds).map((hit) => hit.seq)
        : [];
      return passagesBySeq(db, fuseRankings([keyword, similar], limit));
    },

    close(): void {
      embedding.close();
      processor.close();
    },
  };
}
