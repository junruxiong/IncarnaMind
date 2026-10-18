/**
 * The analysis window's page (ADR-0023): it starts the Worker that runs
 * Python, gives it the main process's jobs and passes back what each came
 * to. Code a model wrote runs in that Worker and can post to this page, so
 * only a reply to the job running, of the shape the Worker writes and within
 * its size, is passed on.
 */
import {
  ANALYSIS_BRIDGE_KEY,
  type AnalysisBridge,
  MAX_REPLY_CHARACTERS,
} from "../../shared/analysisWindow";

const found = (window as unknown as Partial<Record<string, AnalysisBridge>>)[ANALYSIS_BRIDGE_KEY];
if (!found) throw new Error("The analysis window's preload didn't load.");
const bridge: AnalysisBridge = found;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let started = false;
let running: number | null = null;

const start = (message: string | null) => {
  started = true;
  bridge.ready(message === null ? { ok: true } : { ok: false, message: message.slice(0, 2000) });
};

const finish = (reply: { result: string } | { failure: string }) => {
  if (running === null) return;
  const id = running;
  running = null;
  bridge.done({ id, ...reply });
};

worker.addEventListener("message", ({ data }: MessageEvent<unknown>) => {
  if (!isRecord(data)) return;
  if (!started) {
    if (data.kind === "ready") start(null);
    else if (data.kind === "failed") start(String(data.message));
    return;
  }
  if (data.kind !== "done" || data.id !== running) return;
  if (typeof data.result === "string" && data.result.length <= MAX_REPLY_CHARACTERS) {
    finish({ result: data.result });
  } else if (typeof data.failure === "string") {
    finish({ failure: data.failure.slice(0, 2000) });
  } else {
    finish({ failure: "The analysis gave back more than it may." });
  }
});

worker.addEventListener("error", (event) => {
  const message = event.message || "The analysis worker stopped.";
  if (!started) start(message);
  else finish({ failure: message });
});

bridge.onRun((request) => {
  running = request.id;
  worker.postMessage(request);
});
