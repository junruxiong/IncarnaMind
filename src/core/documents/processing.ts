/**
 * Turns a stored file into Passages: what the processing worker does with
 * each job. It never touches the database; the core writes the result.
 */
import { readFile } from "node:fs/promises";
import type { DocumentFailureReason, DocumentKind } from "../api";
import { ExtractionError, extractText } from "./extract";
import { type BuiltPassage, buildPassages } from "./passages";

export interface ProcessingJob {
  documentId: string;
  contentHash: string;
  kind: DocumentKind;
  /** The Document's copy in the data folder. */
  file: string;
}

export type ProcessingResult =
  | { outcome: "ready"; pageCount: number | null; passages: BuiltPassage[] }
  | { outcome: "no-text"; pageCount: number | null }
  | { outcome: "failed"; reason: DocumentFailureReason; message: string };

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
    const missing = (error as NodeJS.ErrnoException).code === "ENOENT";
    return {
      outcome: "failed",
      reason: missing ? "file-missing" : "processing-error",
      message: messageOf(error),
    };
  }
  try {
    const { pageCount, pages } = await extractText(job.kind, bytes);
    const passages = buildPassages(pages);
    if (passages.length === 0) return { outcome: "no-text", pageCount };
    return { outcome: "ready", pageCount, passages };
  } catch (error) {
    if (error instanceof ExtractionError) {
      return { outcome: "failed", reason: error.reason, message: error.message };
    }
    return { outcome: "failed", reason: "processing-error", message: messageOf(error) };
  }
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));
