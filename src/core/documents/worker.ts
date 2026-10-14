/**
 * The Document-processing worker thread. The core starts it through the
 * `?nodeWorker` import in `./processor`: electron-vite builds it as its own
 * chunk, and the Vitest config does the same for tests.
 *
 * A job runs one at a time. While a PDF's scanned pages are read (#397), the
 * worker draws each page and asks the core to read its picture, which runs
 * the text recognition model, and tells it how far it is; the core can stop
 * the job between pages.
 */
import { parentPort } from "node:worker_threads";
import type { RecognizedLine } from "../adapters";
import { ReadingUnavailable } from "./extract";
import { type ReadingChannel, runJob, type WorkerRequest, type WorkerResponse } from "./processing";

const port = parentPort;
if (!port) throw new Error("The Document-processing worker must run as a worker thread.");

const send = (response: WorkerResponse) => {
  // A picture's pixels move to the core rather than being copied.
  const transfer = "picture" in response ? [response.picture.pixels.buffer as ArrayBuffer] : [];
  port.postMessage(response, transfer);
};

/** The jobs under way, by id: their stop signals and the pages waiting to be read. */
const running = new Map<
  number,
  {
    stop: AbortController;
    waiting: Map<number, { resolve(lines: RecognizedLine[]): void; reject(error: Error): void }>;
  }
>();
let nextRequest = 1;

port.on("message", (message: WorkerRequest) => {
  if ("job" in message) {
    const { id } = message;
    const job = { stop: new AbortController(), waiting: new Map() };
    running.set(id, job);
    const channel: ReadingChannel = {
      signal: job.stop.signal,
      progress: (contentHash, read, of, unit) =>
        send({ id, progress: { contentHash, read, of, unit } }),
      read: (page, picture) =>
        new Promise((resolve, reject) => {
          const request = nextRequest++;
          job.waiting.set(request, { resolve, reject });
          send({ id, request, page, picture });
        }),
    };
    void runJob(message.job, channel, message.password).then((result) => {
      running.delete(id);
      send({ id, result });
    });
    return;
  }
  const job = running.get(message.id);
  if (!job) return;
  if ("stop" in message) {
    job.stop.abort();
    for (const waiter of job.waiting.values()) waiter.reject(new Error("Stopped."));
    job.waiting.clear();
    return;
  }
  const waiter = job.waiting.get(message.request);
  job.waiting.delete(message.request);
  if (!waiter) return;
  if (message.lines) waiter.resolve(message.lines);
  else {
    const { message: text = "", unavailable = false } = message.error ?? {};
    waiter.reject(unavailable ? new ReadingUnavailable(text) : new Error(text));
  }
});
