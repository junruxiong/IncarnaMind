/**
 * Turns a Document's file into Passages: what the processing worker does with
 * each job. It reads the file where the User keeps it, once, and hashes what
 * it read, so the Passages are always of the version recorded with them, even
 * if the file changes meanwhile. It never touches the database; the core
 * writes the result.
 */
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import type { DocumentFailureReason, DocumentKind } from "../api";
import { stripBoilerplate } from "./boilerplate";
import { ExtractionError, extractText } from "./extract";
import { keywordText } from "./keywords";
import { type BuiltPassage, buildPassages, type PageText } from "./passages";

/**
 * The version of this pipeline: text extraction, Passage building, text
 * normalisation and keyword indexing. Bump it whenever any of them changes the
 * Passages or the index: Documents processed by an older version are
 * processed again at startup, going through the usual statuses.
 * 1: #25 (400/200 Passages, trigram index). 2: ADR-0009 (500/200 Passages,
 * normalised text, segmented keyword index, embeddings). 3: #30 (running
 * headers, footers and page numbers removed; page text stored for the
 * Citation check). 4: #31 (Passages built from whole lines, as the retrieval
 * prototype built them).
 */
export const PROCESSING_VERSION = 4;

export interface ProcessingJob {
  documentId: string;
  kind: DocumentKind;
  /** The Document's file, where the User keeps it. */
  file: string;
}

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
      /** The text of each page, as the Passages were built from it (see ./boilerplate). */
      pages: PageText[];
    }
  | { outcome: "no-text"; pageCount: number | null }
  | { outcome: "failed"; reason: DocumentFailureReason; message: string };

export type ProcessingResult =
  | (Processed & {
      /** The version processed: the SHA-256 of the bytes read, hex. */
      contentHash: string;
      /** How many bytes were read. */
      size: number;
    })
  /** The file couldn't be read: it is gone (`gone`), or can't be read now. Nothing was processed. */
  | { outcome: "file-unreadable"; gone: boolean; message: string }
  /** The worker stopped mid-job (a crash, or out of memory). */
  | { outcome: "crashed"; message: string };

/** Messages between the core and the worker. `id` pairs a result with its job. */
export interface WorkerRequest {
  id: number;
  job: ProcessingJob;
}

export interface WorkerResponse {
  id: number;
  result: ProcessingResult;
}

/** Never throws: every problem becomes a "failed" result. */
export async function processFile(job: ProcessingJob): Promise<ProcessingResult> {
  let bytes: Uint8Array;
  try {
    bytes = await readFile(job.file);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    return {
      outcome: "file-unreadable",
      gone: code === "ENOENT" || code === "ENOTDIR",
      message: messageOf(error),
    };
  }
  const version = {
    contentHash: createHash("sha256").update(bytes).digest("hex"),
    size: bytes.byteLength,
  };
  return { ...(await processBytes(job.kind, bytes)), ...version };
}

async function processBytes(kind: DocumentKind, bytes: Uint8Array): Promise<Processed> {
  try {
    const extracted = await extractText(kind, bytes);
    const pages = stripBoilerplate(extracted.pages);
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
