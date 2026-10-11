/**
 * Runs jobs one at a time on a worker thread, so extracting a large PDF never
 * blocks the core's thread (the Electron main process): processing jobs, and
 * metadata jobs, which only read a Document's creation date. A job reading a
 * PDF's scanned pages (#397) asks for each page's picture to be read, and
 * says how far it is.
 */
import type { Worker } from "node:worker_threads";
import type { TextUnit } from "../../shared/units";
import type { PagePicture, RecognizedLine } from "../adapters";
import type {
  MetadataJob,
  MetadataResult,
  ProcessingJob,
  ProcessingResult,
  WorkerJob,
  WorkerRequest,
  WorkerResponse,
  WorkerResult,
} from "./processing";
import startDocumentWorker from "./worker?nodeWorker";

export interface ProcessorHandlers {
  /** A processing job is about to go to the worker. */
  onStart(job: ProcessingJob): void;
  onResult(job: ProcessingJob, result: ProcessingResult): void;
  onMetadata(job: MetadataJob, result: MetadataResult): void;
  /**
   * How far a job is in reading the scanned pages of version `contentHash`:
   * `read` of `of`, and the page just read.
   */
  onProgress(
    job: ProcessingJob,
    progress: { contentHash: string; read: number; of: number; unit: TextUnit | null },
  ): void;
  /**
   * Reads a scanned page's picture for a job. Rejects with `unavailable` set
   * when no page can be read now (the model can't be had).
   */
  read(job: ProcessingJob, page: number, picture: PagePicture): Promise<RecognizedLine[]>;
  /** Whether an error from `read` means no page can be read now. */
  isUnavailable(error: unknown): boolean;
}

export interface Processor {
  enqueue(job: WorkerJob): void;
  /**
   * Drops a Document's processing job if it hasn't started. A started one
   * that is reading scanned pages stops before its next page, reporting
   * "stopped"; another still reports its result, as does every metadata job.
   */
  cancel(documentId: string): void;
  /** Holds jobs back, while a Meeting is recorded (#135): the one in the worker finishes, the rest wait. */
  hold(held: boolean): void;
  /** Stops the worker. Jobs not yet finished are dropped without a result. */
  close(): void;
}

export function createProcessor(
  handlers: ProcessorHandlers,
  reportError: (error: unknown) => void = (error) => console.error(error),
): Processor {
  const queue: WorkerJob[] = [];
  let worker: Worker | undefined;
  let current: { id: number; job: WorkerJob } | undefined;
  let nextId = 1;
  let pumpScheduled = false;
  let closed = false;
  let held = false;

  const call = (fn: () => void) => {
    try {
      fn();
    } catch (error) {
      reportError(error);
    }
  };

  function finish(id: number, result: WorkerResult): void {
    if (closed || current?.id !== id) return;
    const { job } = current;
    current = undefined;
    // The worker answers each job with a result of its kind (see `runJob`); a crash is either's.
    if (job.task === "process") call(() => handlers.onResult(job, result as ProcessingResult));
    else call(() => handlers.onMetadata(job, result as MetadataResult));
    schedulePump();
  }

  /** The worker died mid-job (a crash, or out of memory): fail that job and start afresh. */
  function lose(lost: Worker, message: string): void {
    if (worker !== lost) return;
    worker = undefined;
    if (current) finish(current.id, { outcome: "crashed", message });
  }

  /** A job's request to read a page's picture: answered once the model has read it. */
  function read(from: Worker, id: number, request: number, page: number, picture: PagePicture) {
    const asked = current;
    if (asked?.id !== id || asked.job.task !== "process") return;
    const answer = (message: WorkerRequest) => {
      if (worker === from && current?.id === id) from.postMessage(message);
    };
    handlers.read(asked.job, page, picture).then(
      (lines) => answer({ id, request, lines }),
      (error: unknown) =>
        answer({
          id,
          request,
          error: {
            message: error instanceof Error ? error.message : String(error),
            unavailable: handlers.isUnavailable(error),
          },
        }),
    );
  }

  function startWorker(): Worker {
    const started = startDocumentWorker({ name: "incarnamind-documents" });
    started.on("message", (response: WorkerResponse) => {
      if ("result" in response) finish(response.id, response.result);
      else if ("picture" in response) {
        read(started, response.id, response.request, response.page, response.picture);
      } else if (current?.id === response.id && current.job.task === "process") {
        const { job } = current;
        call(() => handlers.onProgress(job, response.progress));
      }
    });
    started.on("error", (error) => lose(started, `Processing stopped: ${error.message}`));
    started.on("exit", (code) => lose(started, `Processing stopped (exit code ${code}).`));
    return started;
  }

  function pump(): void {
    pumpScheduled = false;
    if (closed || held || current) return;
    const job = queue.shift();
    if (!job) return;
    current = { id: nextId++, job };
    if (job.task === "process") call(() => handlers.onStart(job));
    worker ??= startWorker();
    worker.postMessage({ id: current.id, job } satisfies WorkerRequest);
  }

  // Deferred, so a job enqueued while the core starts up runs after it has returned.
  function schedulePump(): void {
    if (pumpScheduled) return;
    pumpScheduled = true;
    queueMicrotask(pump);
  }

  return {
    enqueue(job) {
      if (closed) return;
      queue.push(job);
      schedulePump();
    },
    cancel(documentId) {
      const index = queue.findIndex(
        (job) => job.task === "process" && job.documentId === documentId,
      );
      if (index !== -1) queue.splice(index, 1);
      if (current?.job.task === "process" && current.job.documentId === documentId) {
        worker?.postMessage({ id: current.id, stop: true } satisfies WorkerRequest);
      }
    },
    hold(value) {
      held = value;
      if (!held) schedulePump();
    },
    close() {
      closed = true;
      queue.length = 0;
      current = undefined;
      const stopping = worker;
      worker = undefined;
      void stopping?.terminate();
    },
  };
}
