/**
 * Documents: adding files, processing them into Passages off the main thread
 * (extracting on a worker thread, embedding through the embedder), renaming,
 * filing in Folders, soft-deleting, and keyword, vector and hybrid search.
 */
import { randomUUID } from "node:crypto";
import { rm } from "node:fs/promises";
import { basename, extname, isAbsolute, join } from "node:path";
import type {
  AddDocumentsResult,
  Document,
  DocumentFailureReason,
  DocumentKind,
  DocumentStatus,
  PassageSearchResult,
  ProviderErrorKind,
  SearchMode,
  SkippedFile,
  TaggingState,
} from "../api";
import { EmbeddingUnavailableError, type SearchEmbedder } from "../embedding/active";
import { InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { Database } from "../storage";
import { tagsOfDocument } from "../tags";
import type { StoredTaggingState } from "../tags/tagger";
import { copyFileName, createOpenCopies } from "./copies";
import { createEmbeddingQueue } from "./embedding";
import { createDocumentFiles, kindOf } from "./files";
import { keywordText } from "./keywords";
import type { PageText } from "./passages";
import {
  PROCESSING_VERSION,
  type ProcessedPassage,
  type ProcessingJob,
  type ProcessingResult,
} from "./processing";
import { createProcessor } from "./processor";
import {
  fuseRankingScores,
  fuseRankings,
  HYBRID_CANDIDATES,
  keywordSearch,
  passagesBySeq,
  passagesInWindow,
  type WindowedPassage,
  windowedPassagesBySeq,
} from "./search";
import { type SearchCandidate, type SearchToolOptions, searchDocumentsTool } from "./searchTool";
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
  tagging_status: string;
  tagging_error_kind: string | null;
  tagging_error_message: string | null;
  embedding_model: string | null;
  created_at: string;
  updated_at: string;
}

const COLUMNS = `id, content_hash, name, kind, size, page_count, status,
  failure_reason, failure_message, folder_id, tagging_status, tagging_error_kind,
  tagging_error_message, embedding_model, created_at, updated_at`;

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

/** `listDocuments` options, checked. `folderId` and `tagId` undefined mean any. */
export function parseListOptions(options: unknown): {
  folderId: string | undefined;
  includeSubfolders: boolean;
  tagId: string | undefined;
} {
  if (options === undefined) {
    return { folderId: undefined, includeSubfolders: false, tagId: undefined };
  }
  if (!isRecord(options)) throw new InvalidInputError("listDocuments expects an object.");
  const { folderId, includeSubfolders = false, tagId } = options;
  if (folderId !== undefined && (typeof folderId !== "string" || folderId === "")) {
    throw new InvalidInputError("A Folder id must be a non-empty string.");
  }
  if (typeof includeSubfolders !== "boolean") {
    throw new InvalidInputError("includeSubfolders must be true or false.");
  }
  if (tagId !== undefined && (typeof tagId !== "string" || tagId === "")) {
    throw new InvalidInputError("A Tag id must be a non-empty string.");
  }
  return { folderId, includeSubfolders, tagId };
}

/** Where automatic tagging is, from what is stored and the Document's processing status. */
function taggingOf(status: string, stored: string): TaggingState {
  if (stored === "tagged") return "tagged"; // kept while the Document is processed again
  if (status === "ready") return stored as StoredTaggingState;
  if (status === "failed" || status === "no-text") return "skipped";
  return "pending";
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

/** A Passage a Citation names, with its Document: what the Citation check needs. */
export interface CitationSource {
  passageId: string;
  /** The Passage's text, as search returned it. */
  text: string;
  /** The pages the Passage covers; null for a Document without pages. */
  pageFrom: number | null;
  pageTo: number | null;
  documentId: string;
  documentName: string;
  documentKind: DocumentKind;
  contentHash: string;
  /** The Document has been deleted since. */
  documentDeleted: boolean;
}

/** A live Document's stored file, opened for reading. */
export interface DocumentFile {
  document: Document;
  /** The file's bytes. Cancel the stream if it isn't read to the end, so the file is closed. */
  stream: ReadableStream<Uint8Array>;
}

export interface DocumentsOptions {
  db: Database;
  dataDir: string;
  /** Where copies to open in another app go (see ./copies). */
  tempDir: string;
  now: () => string;
  /**
   * The embedding model in use (the built-in one unless the User chose
   * another), which embeds Passages and search queries.
   */
  model: SearchEmbedder;
  /** Pushes the "document.status" event. */
  emitStatus(document: Document): void;
  /** A Document just became ready, i.e. searchable: called before its status is pushed. */
  onReady?: (documentId: string) => void;
}

/** Which Documents `list` returns. Every condition given must match. */
export interface DocumentFilter {
  /** Only Documents filed in one of these Folders. */
  folderIds?: readonly string[];
  /** Only Documents carrying this Tag, if it isn't deleted. */
  tagId?: string;
  /** Only these Documents. */
  ids?: readonly string[];
}

export function createDocuments(options: DocumentsOptions) {
  const { db, dataDir, now, model, emitStatus } = options;
  const files = createDocumentFiles(dataDir);
  const openCopies = createOpenCopies(options.tempDir);
  const vectors = createVectorIndex(db, model);

  /**
   * The share of a Document's Passages embedded so far with the current model:
   * none while it waits its turn after a switch, holding another model's vectors.
   */
  const progressOf = (row: DocumentRow): number => {
    if (row.embedding_model !== model.id) return 0;
    const counts = db.get<{ total: number; embedded: number }>(
      `SELECT count(*) AS total, count(embedding) AS embedded FROM passages
       WHERE document_id = ? AND deleted_at IS NULL`,
      [row.id],
    );
    return counts && counts.total > 0 ? counts.embedded / counts.total : 0;
  };

  const toDocument = (row: DocumentRow): Document => {
    const tagging = taggingOf(row.status, row.tagging_status);
    return {
      id: row.id,
      name: row.name,
      kind: row.kind as DocumentKind,
      contentHash: row.content_hash,
      size: row.size,
      pageCount: row.page_count,
      status: row.status as DocumentStatus,
      progress: row.status === "embedding" ? progressOf(row) : null,
      failure:
        row.status === "failed"
          ? {
              reason: (row.failure_reason ?? "processing-error") as DocumentFailureReason,
              message: row.failure_message ?? "",
            }
          : null,
      folderId: row.folder_id,
      tags: tagsOfDocument(db, row.id),
      tagging,
      taggingError:
        tagging === "failed"
          ? {
              kind: (row.tagging_error_kind ?? "unknown") as ProviderErrorKind,
              message: row.tagging_error_message ?? "",
            }
          : null,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  };

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

  const embedding = createEmbeddingQueue({
    db,
    now,
    model,
    vectors,
    announce,
    onReady: options.onReady,
  });

  /** Adds a Passage to the keyword index, under the `seq` it was stored with. */
  const index = (seq: number, text: string) =>
    db.run("INSERT INTO passages_fts (rowid, text) VALUES (?, ?)", [BigInt(seq), text]);

  /** Stores the text of a Document's pages, for the Citation check. Run in a transaction. */
  function insertPages(documentId: string, pages: readonly PageText[]): void {
    const at = now();
    for (const { page, text } of pages) {
      db.run(
        `INSERT INTO document_pages (id, document_id, page, text, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
        [randomUUID(), documentId, page === null ? null : BigInt(page), text, at, at],
      );
    }
  }

  /** Soft-deletes a Document's stored page text. Run in a transaction. */
  const removePages = (documentId: string, at: string) =>
    db.run(
      `UPDATE document_pages SET deleted_at = ?, updated_at = ?
       WHERE document_id = ? AND deleted_at IS NULL`,
      [at, at, documentId],
    );

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
      removePages(job.documentId, at);
      let status: DocumentStatus;
      if (result.outcome === "ready") {
        insertPassages(job.documentId, row.name, result.passages);
        insertPages(job.documentId, result.pages);
        status = model.isReady() ? "embedding" : "waiting-for-model";
      } else {
        status = result.outcome;
      }
      db.run(
        `UPDATE documents SET status = ?, page_count = ?, failure_reason = ?, failure_message = ?,
           processing_version = ?, embedding_model = ?, embedding_dimensions = NULL, updated_at = ?
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
  void openCopies.tidy();
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
  /**
   * Puts every Document with Passages to search whose vectors aren't all from
   * the current model through embedding (again), or has it wait for the model.
   * Documents already embedded with it stay ready. Returns those whose status changed.
   */
  function embedWithCurrentModel(): string[] {
    const ready = model.isReady();
    const changed = db.transaction(() => {
      const rows = db.all<{ id: string; status: string }>(
        `SELECT id, status FROM documents
         WHERE deleted_at IS NULL AND status IN ('ready', 'embedding', 'waiting-for-model')
           AND (status <> 'ready' OR embedding_model IS NOT ?)
         ORDER BY created_at, rowid`,
        [model.id],
      );
      const target = ready ? "embedding" : "waiting-for-model";
      return rows.filter((row) => {
        if (row.status === target) return false;
        setStatus(row.id, target);
        return true;
      });
    });
    if (ready) {
      const embeddingRows = db.all<{ id: string }>(
        `SELECT id FROM documents WHERE deleted_at IS NULL AND status = 'embedding'
         ORDER BY created_at, rowid`,
      );
      for (const { id } of embeddingRows) embedding.enqueue(id);
    } else if (
      db.get("SELECT 1 FROM documents WHERE deleted_at IS NULL AND status = 'waiting-for-model'")
    ) {
      model.ensure();
    }
    return changed.map((row) => row.id);
  }

  // Embedding carries on where it stopped, or waits for the model; Documents
  // embedded with a model other than the current one are embedded again.
  embedWithCurrentModel();
  model.onReady(() => embedding.resumeWaiting());
  // The User switched model: what is under way stops, and every Document is embedded again.
  model.onSwitch(() => {
    embedding.restart();
    vectors.reset();
    for (const id of embedWithCurrentModel()) announce(id);
  });

  /** The live Documents a filter keeps, as SQL conditions on `documents`, with their parameters. */
  const filterWhere = ({ folderIds, tagId, ids }: DocumentFilter) => {
    const conditions = ["deleted_at IS NULL"];
    const params: string[] = [];
    if (folderIds !== undefined) {
      conditions.push("folder_id IN (SELECT value FROM json_each(?))");
      params.push(JSON.stringify(folderIds));
    }
    if (tagId !== undefined) {
      conditions.push(
        `id IN (SELECT l.document_id FROM document_tags l JOIN tags t ON t.id = l.tag_id
                WHERE l.tag_id = ? AND l.deleted_at IS NULL AND t.deleted_at IS NULL)`,
      );
      params.push(tagId);
    }
    if (ids !== undefined) {
      conditions.push("id IN (SELECT value FROM json_each(?))");
      params.push(JSON.stringify(ids));
    }
    return { where: conditions.join(" AND "), params };
  };

  /** Most recently added first, only those the filter keeps. */
  const list = (filter: DocumentFilter = {}): Document[] => {
    const { where, params } = filterWhere(filter);
    return db
      .all<DocumentRow>(
        `SELECT ${COLUMNS} FROM documents WHERE ${where} ORDER BY created_at DESC, rowid DESC`,
        params,
      )
      .map(toDocument);
  };

  /** A live Document, and the name a copy of its file gets (see ./copies). Throws NotFoundError otherwise. */
  const copyName = (idInput: unknown): { document: Document; fileName: string } => {
    const row = find(parseId(idInput));
    if (!row) throw new NotFoundError("There is no such Document.");
    const document = toDocument(row);
    return { document, fileName: copyFileName(document) };
  };

  /**
   * Copies a live Document's stored file to `destination`, an absolute path,
   * replacing any file there. Deleted or unknown Documents, and a missing
   * stored file, are refused (NotFoundError) before anything is written.
   */
  async function saveCopy(idInput: unknown, destination: unknown): Promise<void> {
    const id = parseId(idInput);
    if (typeof destination !== "string" || !isAbsolute(destination)) {
      throw new InvalidInputError("A copy must be saved to an absolute file path.");
    }
    const row = find(id);
    if (!row) throw new NotFoundError("There is no such Document.");
    try {
      await files.copy(row.content_hash, destination);
    } catch (error) {
      if (
        (error as NodeJS.ErrnoException).code === "ENOENT" &&
        !(await files.has(row.content_hash))
      ) {
        throw new NotFoundError("The Document's file is missing from the data folder.");
      }
      throw error;
    }
  }

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
      if (mode === "vector") throw model.notReadyError();
      return null;
    };
    if (!(await model.load())) return unavailable();
    try {
      return await model.embedQuery(query);
    } catch (error) {
      if (error instanceof EmbeddingUnavailableError) return unavailable();
      if (mode === "vector") throw error;
      console.error(error);
      return null;
    }
  }

  /** SQL for the live Documents `d` with Passages to search: of all of them, or only of these. */
  function searchable(documentIds: readonly string[] | undefined) {
    return {
      where: `d.deleted_at IS NULL
        AND EXISTS (SELECT 1 FROM passages p WHERE p.document_id = d.id AND p.deleted_at IS NULL)
        ${documentIds ? "AND d.id IN (SELECT value FROM json_each(?))" : ""}`,
      params: documentIds ? [JSON.stringify(documentIds)] : [],
    };
  }

  function searchableCount(documentIds?: readonly string[]): number {
    const { where, params } = searchable(documentIds);
    return (
      db.get<{ count: number }>(`SELECT count(*) AS count FROM documents d WHERE ${where}`, params)
        ?.count ?? 0
    );
  }

  /** Hybrid search for the document-search Tool: the best `limit` live Passages, with fused scores. */
  async function candidates(
    query: string,
    limit: number,
    documentIds: readonly string[] | undefined,
  ): Promise<SearchCandidate[]> {
    if (query.trim() === "") return [];
    const vector = await queryVector(query, "hybrid");
    const listed = Math.max(limit, HYBRID_CANDIDATES);
    const keyword = keywordSearch(db, query, listed, documentIds);
    const similar = vector ? vectors.search(vector, listed, documentIds).map((hit) => hit.seq) : [];
    const fused = fuseRankingScores([keyword, similar], limit);
    const scores = new Map(fused.map((hit) => [hit.seq, hit.score]));
    return windowedPassagesBySeq(
      db,
      fused.map((hit) => hit.seq),
    ).map((passage) => ({ ...passage, score: scores.get(passage.seq) ?? 0 }));
  }

  return {
    /**
     * The document-search Tool (see ./searchTool): hybrid search, grouped by
     * Document and clustered by sliding window.
     */
    searchTool(query: string, options?: SearchToolOptions): Promise<WindowedPassage[]> {
      return searchDocumentsTool(
        { candidates, window: (documentId, window) => passagesInWindow(db, documentId, window) },
        query,
        options,
      );
    },

    /**
     * How far embedding with the current model has got: the live Documents
     * with Passages to search, and of those, the ones whose Passages are all
     * embedded with it.
     */
    embeddingProgress(): { total: number; done: number } {
      const counts = db.get<{ total: number; done: number | null }>(
        `SELECT count(*) AS total,
           sum(status = 'ready' AND embedding_model IS ?) AS done
         FROM documents
         WHERE deleted_at IS NULL AND status IN ('ready', 'embedding', 'waiting-for-model')`,
        [model.id],
      );
      return { total: counts?.total ?? 0, done: counts?.done ?? 0 };
    },

    /** How many live Documents have Passages to search: of all of them, or only of these. */
    searchableCount,

    /**
     * How many live Documents have Passages to search (of all of them, or only
     * of these), and the names of the `limit` most recently added, newest first.
     */
    searchableNames(
      documentIds: readonly string[] | undefined,
      limit: number,
    ): { total: number; names: string[] } {
      const { where, params } = searchable(documentIds);
      const names = db
        .all<{ name: string }>(
          `SELECT d.name FROM documents d WHERE ${where}
           ORDER BY d.created_at DESC, d.rowid DESC LIMIT ?`,
          [...params, BigInt(limit)],
        )
        .map((row) => row.name);
      return { total: searchableCount(documentIds), names };
    },

    /**
     * A Passage and its Document, for a Citation. Deleted Passages are found
     * too (processing a Document again replaces its Passages), and a deleted
     * Document is reported as such. Null for an unknown Passage.
     */
    citationSource(passageId: string): CitationSource | null {
      const row = db.get<{
        id: string;
        text: string;
        page_from: number | null;
        page_to: number | null;
        document_id: string;
        name: string;
        kind: string;
        content_hash: string;
        deleted_at: string | null;
      }>(
        `SELECT p.id, p.text, p.page_from, p.page_to, p.document_id, d.name, d.kind,
           d.content_hash, d.deleted_at
         FROM passages p JOIN documents d ON d.id = p.document_id
         WHERE p.id = ?`,
        [passageId],
      );
      if (!row) return null;
      return {
        passageId: row.id,
        text: row.text,
        pageFrom: row.page_from,
        pageTo: row.page_to,
        documentId: row.document_id,
        documentName: row.name,
        documentKind: row.kind as DocumentKind,
        contentHash: row.content_hash,
        documentDeleted: row.deleted_at !== null,
      };
    },

    /**
     * The stored text of a live Document's pages from `from` to `to`, in
     * order, as its Passages were built from it. With both null, every row: the
     * one "page" of a Document without pages. Empty if the Document was
     * deleted, or hasn't been processed by this version yet.
     */
    pageTexts(documentId: string, from: number | null, to: number | null): PageText[] {
      const range = from === null || to === null ? "" : "AND dp.page BETWEEN ? AND ?";
      const params = from === null || to === null ? [] : [BigInt(from), BigInt(to)];
      return db
        .all<{ page: number | null; text: string }>(
          `SELECT dp.page, dp.text FROM document_pages dp
           JOIN documents d ON d.id = dp.document_id
           WHERE dp.document_id = ? AND dp.deleted_at IS NULL AND d.deleted_at IS NULL ${range}
           ORDER BY dp.page`,
          [documentId, ...params],
        )
        .map((row) => ({ page: row.page, text: row.text }));
    },

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

    /** The ids of the live Documents the filter keeps, in `list`'s order. */
    ids(filter: DocumentFilter = {}): string[] {
      const { where, params } = filterWhere(filter);
      return db
        .all<{ id: string }>(
          `SELECT id FROM documents WHERE ${where} ORDER BY created_at DESC, rowid DESC`,
          params,
        )
        .map((row) => row.id);
    },

    /** A live Document. Throws NotFoundError for an unknown or deleted one. */
    get(idInput: unknown): Document {
      const row = find(parseId(idInput));
      if (!row) throw new NotFoundError("There is no such Document.");
      return toDocument(row);
    },

    /** The live Documents among `ids`, in that order. */
    getMany(ids: readonly string[]): Document[] {
      return ids.flatMap((id) => {
        const row = find(id);
        return row ? [toDocument(row)] : [];
      });
    },

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
        const filed = list({ folderIds });
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
        db.run(
          `UPDATE document_tags SET deleted_at = ?, updated_at = ?
           WHERE document_id = ? AND deleted_at IS NULL`,
          [at, at, id],
        );
        removePages(id, at);
      });
      vectors.removeDocument(id);
      processor.cancel(id);
      await releaseFile(row.content_hash);
    },

    copyName,
    saveCopy,

    /**
     * Copies a live Document's stored file into a new folder of its own in the
     * temporary folder, named after the Document (see ./copies), to open in
     * another app. Resolves with the copy's path. Refuses what `saveCopy` does.
     */
    async openableCopy(idInput: unknown): Promise<string> {
      const { document, fileName } = copyName(idInput);
      const folder = await openCopies.folder();
      try {
        await saveCopy(document.id, join(folder, fileName));
      } catch (error) {
        await rm(folder, { recursive: true, force: true });
        throw error;
      }
      return join(folder, fileName);
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
