/**
 * Runs jobs one at a time on a worker thread, so extracting a large PDF never
 * blocks the core's thread (the Electron main process): processing jobs, and
 * metadata jobs, which only read a Document's creation date.
 */
import type { Worker } from "node:worker_threads";
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
}

export interface Processor {
  enqueue(job: WorkerJob): void;
  /**
   * Drops a Document's processing job if it hasn't started. A started job
   * still reports its result, and so does every metadata job.
   */
  cancel(documentId: string): void;
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

  function startWorker(): Worker {
    const started = startDocumentWorker({ name: "incarnamind-documents" });
    started.on("message", (response: WorkerResponse) => finish(response.id, response.result));
    started.on("error", (error) => lose(started, `Processing stopped: ${error.message}`));
    started.on("exit", (code) => lose(started, `Processing stopped (exit code ${code}).`));
    return started;
  }

  function pump(): void {
    pumpScheduled = false;
    if (closed || current) return;
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
