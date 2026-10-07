/**
 * The Document-processing worker thread. The core starts it through the
 * `?nodeWorker` import in `./processor`: electron-vite builds it as its own
 * chunk, and the Vitest config does the same for tests.
 */
import { parentPort } from "node:worker_threads";
import { runJob, type WorkerRequest, type WorkerResponse } from "./processing";

const port = parentPort;
if (!port) throw new Error("The Document-processing worker must run as a worker thread.");

port.on("message", (request: WorkerRequest) => {
  void runJob(request.job).then((result) => {
    port.postMessage({ id: request.id, result } satisfies WorkerResponse);
  });
});
