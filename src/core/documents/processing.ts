/**
 * What the processing worker does with each job. A processing job turns a
 * Document's file into Passages, and reads its creation date. A metadata job
 * reads only its creation date: no text is extracted and nothing is
 * embedded. Each reads the file where the User keeps it, once, and hashes
 * what it read, so what it gives is always of the version recorded with it,
 * even if the file changes meanwhile. It never touches the database; the
 * core writes the result.
 */
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import type { TextUnit } from "../../shared/units";
import type { DocumentFailureReason, DocumentKind } from "../api";
import { stripBoilerplate } from "./boilerplate";
import { creationDate, pdfDate, w3cDate } from "./creationDate";
import { ExtractionError, extractText, pdfCreationDates } from "./extract";
import { officeCreated } from "./formats/coreProperties";
import { packageSizeError } from "./formats/zip";
import { keywordText } from "./keywords";
import { type BuiltPassage, buildPassages } from "./passages";

/**
 * The version of this pipeline: text extraction, Passage building, text
 * normalisation and keyword indexing. Bump it whenever any of them changes the
 * Passages or the index: Documents processed by an older version are
 * processed again at startup, going through the usual statuses.
 * 1: #25 (400/200 Passages, trigram index). 2: ADR-0009 (500/200 Passages,
 * normalised text, segmented keyword index, embeddings). 3: #30 (running
 * headers, footers and page numbers removed; page text stored for the
 * Citation check). 4: #31 (Passages built from whole lines, as the retrieval
 * prototype built them). 5: ADR-0011 (text stored as Units: Markdown by
 * section, plain text by blocks of lines; Word, PowerPoint, Excel and CSV
 * read). 6: ADR-0009's amendment (the keyword index reads traditional
 * Chinese characters as simplified ones; see `HAN_CURRENT_SINCE`). 7: #76
 * (a Word file's comments are read into the section their mark is in).
 */
export const PROCESSING_VERSION = 7;

/**
 * Per kind, the oldest version whose Passages and Units are still what this
 * version would build: a Document processed by an older one is processed
 * again at startup. Version 5 changed nothing for PDFs, whose Passages
 * (and embeddings) stay as they are; version 6 changed no Passage or Unit;
 * version 7 changed only Word files', which gain their comments.
 */
export const CURRENT_SINCE: Readonly<Record<DocumentKind, number>> = {
  pdf: 4,
  text: 5,
  markdown: 5,
  docx: 7,
  pptx: 5,
  xlsx: 5,
  csv: 5,
};

/**
 * For a Document with Han characters in its text or name (Chinese, or
 * Japanese kanji), the oldest version whose keyword index is still what this
 * version would build: since version 6, the index reads traditional
 * characters as simplified ones (see ./keywords). Such a Document processed
 * by an older version, and current for its kind, is processed again at
 * startup. One without Han characters has nothing to fold: its version is
 * set to this one without its file being read, so its text is looked
 * through only once.
 */
export const HAN_CURRENT_SINCE = 6;

/**
 * The version of the metadata read: what a Document's creation date is
 * worked out from (see ./creationDate). A Document whose metadata was read
 * by an older version, or not at all (`metadata_version` 0, as every
 * Document from before migration 24 is), has it read again by a metadata
 * job, in the background, without being processed or embedded again. When
 * only the metadata read changes, bump this, not `PROCESSING_VERSION`.
 * 1: #53 (creation dates).
 */
export const METADATA_VERSION = 1;

/** Extracts a Document's text into Passages and Units, and reads its creation date. */
export interface ProcessingJob {
  task: "process";
  documentId: string;
  kind: DocumentKind;
  /** The Document's file, where the User keeps it. */
  file: string;
}

/** Reads only a Document's metadata (its creation date): nothing is extracted or embedded. */
export interface MetadataJob {
  task: "metadata";
  documentId: string;
  kind: DocumentKind;
  /** The Document's file, where the User keeps it. */
  file: string;
  /** The version indexed: the metadata is read only if the file still is that version. */
  contentHash: string;
  /** That version's first Unit, as stored: where a year is looked for if the metadata has no date. */
  firstUnit: string | null;
}

export type WorkerJob = ProcessingJob | MetadataJob;

/** A Passage as the worker hands it over. */
export interface ProcessedPassage extends BuiltPassage {
  /**
   * The Passage's words for the keyword index (see ./keywords), worked out here
   * so the core's thread doesn't have to. The Document's name isn't included:
   * the core adds it, so a rename only redoes that part.
   */
  keywords: string;
}

/** What became of the file's text. */
type Processed =
  | {
      outcome: "ready";
      pageCount: number | null;
      passages: ProcessedPassage[];
      /**
       * The text of each Unit (page, slide, section, block of rows or lines),
       * as the Passages were built from it (see ./boilerplate), with its label and anchors.
       */
      pages: TextUnit[];
    }
  | { outcome: "no-text"; pageCount: number | null }
  | { outcome: "failed"; reason: DocumentFailureReason; message: string };

/** The file couldn't be read: it is gone (`gone`), or can't be read now. */
interface FileUnreadable {
  outcome: "file-unreadable";
  gone: boolean;
  message: string;
}

/** The worker stopped mid-job (a crash, or out of memory). */
interface Crashed {
  outcome: "crashed";
  message: string;
}

export type ProcessingResult =
  | (Processed & {
      /** The version processed: the SHA-256 of the bytes read, hex. */
      contentHash: string;
      /** How many bytes were read. */
      size: number;
      /** The version's creation date (see ./creationDate), or null. */
      creationDate: string | null;
    })
  /** Nothing was processed. */
  | FileUnreadable
  | Crashed;

export type MetadataResult =
  /** The version's creation date (see ./creationDate), or null if nothing gives one. */
  | { outcome: "read"; creationDate: string | null }
  /** The file is no longer the version indexed: nothing was read. */
  | { outcome: "changed" }
  | FileUnreadable
  | Crashed;

export type WorkerResult = ProcessingResult | MetadataResult;

/** Messages between the core and the worker. `id` pairs a result with its job. */
export interface WorkerRequest {
  id: number;
  job: WorkerJob;
}

export interface WorkerResponse {
  id: number;
  result: WorkerResult;
}

/** Runs a job of either kind. Never throws. */
export function runJob(job: WorkerJob): Promise<WorkerResult> {
  return job.task === "process" ? processFile(job) : readMetadata(job);
}

/** Why the file couldn't be read: it is gone, or can't be read now. */
function unreadable(error: unknown): FileUnreadable {
  const code = (error as NodeJS.ErrnoException).code;
  return {
    outcome: "file-unreadable",
    gone: code === "ENOENT" || code === "ENOTDIR",
    message: messageOf(error),
  };
}

/** The file's bytes, or why they couldn't be read. */
async function readBytes(file: string): Promise<Uint8Array | FileUnreadable> {
  try {
    return await readFile(file);
  } catch (error) {
    return unreadable(error);
  }
}

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

const isPackage = (kind: DocumentKind) => kind === "docx" || kind === "pptx" || kind === "xlsx";

/**
 * A Word, PowerPoint or Excel file too large to open (see
 * `packageSizeError`), refused before it is read: its bytes are only hashed
 * as they stream past, for the version it is. Null for a file that isn't.
 */
async function refuseTooLarge(file: string): Promise<ProcessingResult | null> {
  try {
    const { size } = await stat(file);
    const refused = packageSizeError(size);
    if (!refused) return null;
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
    return {
      outcome: "failed",
      reason: refused.reason,
      message: refused.message,
      contentHash: hash.digest("hex"),
      size,
      creationDate: null,
    };
  } catch (error) {
    return unreadable(error);
  }
}

/** Never throws: every problem becomes a "failed" result. */
export async function processFile(job: ProcessingJob): Promise<ProcessingResult> {
  const refused = isPackage(job.kind) ? await refuseTooLarge(job.file) : null;
  if (refused) return refused;
  const bytes = await readBytes(job.file);
  if (!(bytes instanceof Uint8Array)) return bytes;
  const version = { contentHash: sha256(bytes), size: bytes.byteLength };
  const processed = await processBytes(job.kind, bytes);
  const firstUnit = processed.outcome === "ready" ? (processed.pages[0]?.text ?? null) : null;
  return {
    ...processed,
    ...version,
    creationDate: await readCreationDate(job.kind, bytes, firstUnit),
  };
}

/** A metadata job: the creation date of the version indexed, read from the file. Never throws. */
export async function readMetadata(job: MetadataJob): Promise<MetadataResult> {
  // Text, Markdown and CSV files have no metadata: only the stored first Unit is read.
  if (!hasMetadata(job.kind)) {
    return { outcome: "read", creationDate: creationDate([], job.firstUnit, thisYear()) };
  }
  const bytes = await readBytes(job.file);
  if (!(bytes instanceof Uint8Array)) return bytes;
  if (sha256(bytes) !== job.contentHash) return { outcome: "changed" };
  return { outcome: "read", creationDate: await readCreationDate(job.kind, bytes, job.firstUnit) };
}

const hasMetadata = (kind: DocumentKind) =>
  kind === "pdf" || kind === "docx" || kind === "pptx" || kind === "xlsx";

/** The local calendar year now: no later creation date is plausible. */
const thisYear = () => new Date().getFullYear();

/**
 * A version's creation date (see ./creationDate): the metadata's (a PDF's
 * Info dictionary, then its XMP; an Office file's core properties), or else
 * the latest year written in its first Unit. Never throws: metadata that
 * can't be read is no metadata, and never fails the Document.
 */
async function readCreationDate(
  kind: DocumentKind,
  bytes: Uint8Array,
  firstUnit: string | null,
): Promise<string | null> {
  let fromMetadata: (string | null)[] = [];
  try {
    if (kind === "pdf") {
      const { info, xmp } = await pdfCreationDates(bytes);
      fromMetadata = [info === null ? null : pdfDate(info), xmp === null ? null : w3cDate(xmp)];
    } else if (hasMetadata(kind)) {
      const created = await officeCreated(bytes);
      fromMetadata = [created === null ? null : w3cDate(created)];
    }
  } catch {
    // Unusual or malformed: the year comes from the first Unit.
  }
  return creationDate(fromMetadata, firstUnit, thisYear());
}

async function processBytes(kind: DocumentKind, bytes: Uint8Array): Promise<Processed> {
  try {
    const extracted = await extractText(kind, bytes);
    const pages = kind === "pdf" ? stripBoilerplate(extracted.units) : extracted.units;
    const { pageCount } = extracted;
    const passages = buildPassages(pages).map((passage) => ({
      ...passage,
      keywords: keywordText(passage.text),
    }));
    if (passages.length === 0) return { outcome: "no-text", pageCount };
    return { outcome: "ready", pageCount, passages, pages };
  } catch (error) {
    if (error instanceof ExtractionError) {
      return { outcome: "failed", reason: error.reason, message: error.message };
    }
    return { outcome: "failed", reason: "processing-error", message: messageOf(error) };
  }
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));
