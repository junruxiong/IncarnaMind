/**
 * Documents: adding files, processing them into Passages off the main thread,
 * renaming, filing in Folders, soft-deleting and keyword search.
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
  SkippedFile,
} from "../api";
import { InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { Database } from "../storage";
import { createDocumentFiles, kindOf } from "./files";
import type { ProcessingJob, ProcessingResult } from "./processing";
import { createProcessor } from "./processor";
import { searchPassages } from "./search";

const DEFAULT_SEARCH_LIMIT = 20;
const MAX_SEARCH_LIMIT = 200;
const MAX_NAME_LENGTH = 500;

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

const toDocument = (row: DocumentRow): Document => ({
  id: row.id,
  name: row.name,
  kind: row.kind as DocumentKind,
  contentHash: row.content_hash,
  size: row.size,
  pageCount: row.page_count,
  status: row.status as DocumentStatus,
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

function parseLimit(limit: unknown): number {
  if (limit === undefined) return DEFAULT_SEARCH_LIMIT;
  if (!Number.isInteger(limit) || (limit as number) < 1 || (limit as number) > MAX_SEARCH_LIMIT) {
    throw new InvalidInputError(`The limit must be a whole number from 1 to ${MAX_SEARCH_LIMIT}.`);
  }
  return limit as number;
}

/** The file name without its extension, or the whole name if that leaves nothing. */
const nameFromPath = (path: string) => basename(path, extname(path)).trim() || basename(path);

export interface DocumentsOptions {
  db: Database;
  dataDir: string;
  now: () => string;
  /** Pushes the "document.status" event. */
  emitStatus(document: Document): void;
}

export function createDocuments({ db, dataDir, now, emitStatus }: DocumentsOptions) {
  const files = createDocumentFiles(dataDir);

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

  /** Writes a job's result, unless the Document was deleted meanwhile. Returns whether it did. */
  function record(job: ProcessingJob, result: ProcessingResult): boolean {
    return db.transaction(() => {
      const row = find(job.documentId);
      if (row?.status !== "extracting") return false;
      const at = now();
      if (result.outcome === "ready") {
        for (const passage of result.passages) {
          db.run(
            `INSERT INTO passages (id, document_id, position, page_from, page_to,
               window_from, window_to, text, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [
              randomUUID(),
              job.documentId,
              passage.position,
              passage.pageFrom,
              passage.pageTo,
              passage.windowFrom,
              passage.windowTo,
              passage.text,
              at,
              at,
            ],
          );
        }
      }
      db.run(
        `UPDATE documents SET status = ?, page_count = ?, failure_reason = ?, failure_message = ?,
           updated_at = ?
         WHERE id = ?`,
        result.outcome === "failed"
          ? ["failed", null, result.reason, result.message, at, job.documentId]
          : [result.outcome, result.pageCount, null, null, at, job.documentId],
      );
      return true;
    });
  }

  const processor = createProcessor({
    onStart(job) {
      if (find(job.documentId)?.status !== "queued") return;
      setStatus(job.documentId, "extracting");
      announce(job.documentId);
    },
    onResult(job, result) {
      if (record(job, result)) announce(job.documentId);
      else void releaseFile(job.contentHash); // deleted while processing
    },
  });

  // Startup: tidy the folder, then pick up work a quit interrupted.
  files.prepare((contentHash) => findByHash(contentHash) !== undefined);
  const unfinished = db.all<DocumentRow>(
    `SELECT ${COLUMNS} FROM documents
     WHERE deleted_at IS NULL AND status IN ('queued', 'extracting')
     ORDER BY created_at, rowid`,
  );
  for (const row of unfinished) {
    if (row.status === "extracting") setStatus(row.id, "queued");
    processor.enqueue(jobFor(row));
  }

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

    rename(idInput: unknown, nameInput: unknown): Document {
      const id = parseId(idInput);
      const name = parseName(nameInput);
      if (!find(id)) throw new NotFoundError("There is no such Document.");
      db.run("UPDATE documents SET name = ?, updated_at = ? WHERE id = ?", [name, now(), id]);
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
          `UPDATE passages SET deleted_at = ?, updated_at = ?
           WHERE document_id = ? AND deleted_at IS NULL`,
          [at, at, id],
        );
      });
      processor.cancel(id);
      await releaseFile(row.content_hash);
    },

    search(query: unknown, limit: unknown): PassageSearchResult[] {
      if (typeof query !== "string") throw new InvalidInputError("The search query must be text.");
      return searchPassages(db, query, parseLimit(limit));
    },

    close(): void {
      processor.close();
    },
  };
}
