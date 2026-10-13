/**
 * Importing references (#287): reads a CSL JSON, BibTeX or RIS export, matches
 * each reference to a Document, and writes it to that Document's reference
 * record, or to a record with no Document yet. The rules for which source wins
 * a field are in ../importStore. It reads one file the User chose and sends nothing.
 */
import { randomUUID } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { dirname, extname, isAbsolute, resolve } from "node:path";
import type {
  AddDocumentsResult,
  Document,
  ReferenceFields,
  ReferenceFilesResult,
  ReferenceImportEntry,
  ReferenceImportResult,
  ReferenceRecord,
} from "../../api";
import { InvalidInputError, NotFoundError } from "../../errors";
import type { Database } from "../../storage";
import {
  type ImportTarget,
  linkImported,
  type UndoStep,
  undoImported,
  unlinkedRecords,
  writeImported,
} from "../importStore";
import { type Identity, Index, identityOf, type MatchedBy } from "./match";
import { readReferenceFile } from "./read";

/** Larger than any library's export; refused rather than read into memory. */
const MAX_BYTES = 50 * 1024 * 1024;

/** Files an export may attach that IncarnaMind can add as Documents. */
const ADDABLE = new Set([".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md", ".markdown"]);

/** How many references are matched between pauses, so the window can show progress. */
const CHUNK = 50;

/** The imports whose Undo is still offered. */
const KEPT_IMPORTS = 5;

export interface ReferenceImportOptions {
  db: Database;
  now: () => string;
  /** The Documents that are not deleted, with their reference records. */
  documents(): Document[];
  /** The live Document; throws NotFoundError for an unknown or deleted one. */
  document(id: string): Document;
  addDocuments(paths: string[]): Promise<AddDocumentsResult>;
  /** Pushes "references.changed". */
  changed(documents: Document[]): void;
  /** Pushes "references.importProgress". */
  progress(done: number, total: number): void;
}

interface Session {
  steps: UndoStep[];
  /** Files to offer for each unmatched record, as absolute paths. */
  files: Map<string, string[]>;
}

/** One reference of the file after duplicates are merged. */
interface Item {
  fields: Partial<ReferenceFields>;
  files: string[];
  identity: Identity;
}

/** The first reference's fields, with the second's added where the first has none. */
function fill(into: Partial<ReferenceFields>, from: Partial<ReferenceFields>): void {
  for (const [field, value] of Object.entries(from)) {
    if (!(field in into)) (into as Record<string, unknown>)[field] = value;
  }
}

const isFile = async (path: string) => {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
};

/** The attached files that are on this computer, that IncarnaMind can read, as absolute paths. */
async function existingFiles(listed: readonly string[], folder: string): Promise<string[]> {
  const found: string[] = [];
  for (const entry of listed) {
    const path = isAbsolute(entry) ? entry : resolve(folder, entry);
    if (!ADDABLE.has(extname(path).toLowerCase()) || found.includes(path)) continue;
    if (await isFile(path)) found.push(path);
  }
  return found;
}

const pause = () => new Promise<void>((resolveLater) => setImmediate(resolveLater));

export function createReferenceImport(options: ReferenceImportOptions) {
  const { db, now } = options;
  const sessions = new Map<string, Session>();

  const keep = (importId: string, session: Session) => {
    sessions.set(importId, session);
    for (const old of [...sessions.keys()].slice(0, -KEPT_IMPORTS)) sessions.delete(old);
  };

  /** Pushes the Documents whose records changed, as they are now. */
  const announce = (ids: Iterable<string>) => {
    const documents: Document[] = [];
    for (const id of ids) {
      try {
        documents.push(options.document(id));
      } catch (error) {
        if (!(error instanceof NotFoundError)) throw error;
      }
    }
    if (documents.length > 0) options.changed(documents);
  };

  return {
    async run(pathInput: unknown): Promise<ReferenceImportResult> {
      if (typeof pathInput !== "string" || pathInput === "") {
        throw new InvalidInputError("The path of a reference file is needed.");
      }
      const path = pathInput;
      let bytes: Buffer;
      try {
        if ((await stat(path)).size > MAX_BYTES) {
          return { outcome: "failed", reason: "too-large", message: "The file is over 50 MB." };
        }
        bytes = await readFile(path);
      } catch (error) {
        return {
          outcome: "failed",
          reason: "unreadable",
          message: error instanceof Error ? error.message : "The file can't be read.",
        };
      }
      // UTF-16 is what some Windows tools save; everything else is UTF-8.
      const utf16 = bytes[0] === 0xff && bytes[1] === 0xfe;
      const read = readReferenceFile(new TextDecoder(utf16 ? "utf-16le" : "utf-8").decode(bytes));
      if (!read) {
        return {
          outcome: "failed",
          reason: "unknown-format",
          message: "The file is not CSL JSON, BibTeX or RIS.",
        };
      }
      if (read.references.length === 0 && read.problems.length === 0) {
        return { outcome: "failed", reason: "empty", message: "The file holds no references." };
      }

      // The same work twice in the file is one reference.
      const items: Item[] = [];
      const seen = new Index<Item>();
      let merged = 0;
      for (const reference of read.references) {
        const identity = identityOf(reference.fields);
        const twin = seen.find(identity);
        if (twin) {
          fill(twin.value.fields, reference.fields);
          twin.value.files.push(...reference.files);
          merged++;
          continue;
        }
        const item: Item = {
          fields: { ...reference.fields },
          files: [...reference.files],
          identity,
        };
        items.push(item);
        seen.add(identity, item);
      }

      const documents = new Index<Document>();
      for (const document of options.documents()) {
        const fields = document.reference?.fields;
        documents.add(identityOf(fields ?? {}, document.title ?? document.name), document);
      }
      const loose = new Index<ReferenceRecord>();
      for (const record of unlinkedRecords(db)) loose.add(identityOf(record.fields), record);

      // Matching, with a pause between chunks so progress shows.
      const claimed = new Set<string>();
      const plans: {
        item: Item;
        document: Document | null;
        by: MatchedBy | null;
        loose: ReferenceRecord | null;
        files: string[];
      }[] = [];
      const folder = dirname(path);
      for (const [index, item] of items.entries()) {
        if (index % CHUNK === 0) {
          options.progress(index, items.length);
          await pause();
        }
        let document: Document | null = null;
        let by: MatchedBy | null = null;
        const found = documents.find(item.identity);
        if (found && !claimed.has(found.value.id)) {
          document = found.value;
          by = found.by;
          claimed.add(document.id);
        }
        let record: ReferenceRecord | null = null;
        const looseFound = loose.find(item.identity);
        if (looseFound && !claimed.has(looseFound.value.id)) {
          record = looseFound.value;
          claimed.add(record.id);
        }
        const files = document ? [] : await existingFiles(item.files, folder);
        plans.push({ item, document, by, loose: record, files });
      }
      options.progress(items.length, items.length);

      const session: Session = { steps: [], files: new Map() };
      const entries: ReferenceImportEntry[] = [];
      const changedDocuments = new Set<string>();
      const at = now();
      db.transaction(() => {
        for (const plan of plans) {
          const { item, document } = plan;
          let target: ImportTarget = { unlinked: true };
          const steps: UndoStep[] = [];
          let linked = false;
          if (document) {
            target = { documentId: document.id };
            if (plan.loose) {
              const link = linkImported(db, plan.loose.id, document.id, at);
              if (link) {
                steps.push(...link.steps);
                linked = true;
              }
            }
          } else if (plan.loose) {
            target = { recordId: plan.loose.id };
          }
          const write = writeImported(db, target, item.fields, at);
          steps.push(...write.steps);
          session.steps.push(...steps);
          const created = write.steps.some((step) => step.kind === "fields" && step.created);
          const changed = write.changed || linked;
          if (document && changed) changedDocuments.add(document.id);
          if (!document && plan.files.length > 0) session.files.set(write.recordId, plan.files);
          entries.push({
            recordId: write.recordId,
            title: item.fields.title ?? "",
            documentId: document?.id ?? null,
            matchedBy: plan.by,
            change: created ? "added" : changed ? "updated" : "unchanged",
          });
        }
      });
      announce(changedDocuments);

      const importId = randomUUID();
      keep(importId, session);
      const matched = entries.filter((entry) => entry.documentId !== null).length;
      return {
        outcome: "imported",
        importId,
        format: read.format,
        total: items.length,
        matched,
        unmatched: entries.length - matched,
        changed: entries.filter((entry) => entry.change !== "unchanged").length,
        merged,
        problems: read.problems,
        entries,
        files: [...session.files.values()].reduce((count, list) => count + list.length, 0),
      };
    },

    undo(importId: unknown): Set<string> {
      const session = typeof importId === "string" ? sessions.get(importId) : undefined;
      if (!session) throw new NotFoundError({ key: "error.import.undoGone" });
      sessions.delete(importId as string);
      undoImported(db, session.steps, now());
      // The Documents whose records the import touched.
      const touched = new Set<string>();
      for (const step of session.steps) {
        const row = db.get<{ document_id: string | null }>(
          "SELECT document_id FROM reference_records WHERE id = ?",
          [step.recordId],
        );
        if (row?.document_id) touched.add(row.document_id);
        if (step.kind === "link") touched.add(step.documentId);
      }
      announce(touched);
      return touched;
    },

    async addFiles(importId: unknown): Promise<ReferenceFilesResult> {
      const session = typeof importId === "string" ? sessions.get(importId) : undefined;
      if (!session) throw new NotFoundError({ key: "error.import.filesGone" });
      const documents: Document[] = [];
      const skipped: ReferenceFilesResult["skipped"] = [];
      const wanted = [...session.files.entries()];
      session.files.clear();
      for (const [recordId, paths] of wanted) {
        const added = await options.addDocuments(paths);
        skipped.push(...added.skipped);
        const first = added.documents[0];
        if (!first) continue;
        documents.push(...added.documents);
        const link = db.transaction(() => linkImported(db, recordId, first.id, now()));
        if (link) session.steps.push(...link.steps);
      }
      announce(documents.map((document) => document.id));
      return { documents: documents.map((document) => options.document(document.id)), skipped };
    },

    unmatched: (): ReferenceRecord[] => unlinkedRecords(db),

    async addFile(recordId: unknown, path: unknown): Promise<Document> {
      if (typeof recordId !== "string" || typeof path !== "string") {
        throw new InvalidInputError("A reference and a file are needed.");
      }
      if (!unlinkedRecords(db).some((record) => record.id === recordId))
        throw new NotFoundError({ key: "error.import.noSuchReference" });
      const added = await options.addDocuments([path]);
      const document = added.documents[0];
      if (!document) throw new InvalidInputError({ key: "error.import.fileNotAdded" });
      db.transaction(() => linkImported(db, recordId, document.id, now()));
      announce([document.id]);
      return options.document(document.id);
    },
  };
}
