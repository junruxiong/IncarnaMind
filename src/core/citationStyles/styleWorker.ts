/**
 * The citation-style engine's thread (ADR-0016): formatting a few hundred references in a
 * note style takes most of a second, so it happens here, never on the core's thread. It loads
 * the engine and the locales from the folder it is given, parses each style once, and answers
 * one request at a time.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parentPort, workerData } from "node:worker_threads";
import { formatJob } from "./format";
import type { WorkerRequest, WorkerResponse } from "./protocol";
import { CslEngine, type StyleDescription, StyleError } from "./wasm";

const { folder } = workerData as { folder: string };

/** The locales the shipped styles use; a style's own `<locale>` terms override them. */
const LOCALES = ["en-US", "en-GB", "zh-CN"];

let module: WebAssembly.Module | undefined;
let locales: string[] | undefined;
let engine: CslEngine | undefined;
const styles = new Map<string, { handle: number; description: StyleDescription }>();

function styleOf(file: string) {
  const known = styles.get(file);
  if (known) return known;
  module ??= new WebAssembly.Module(readFileSync(join(folder, "engine.wasm")));
  engine ??= new CslEngine(module);
  locales ??= LOCALES.map((code) =>
    readFileSync(join(folder, "locales", `locales-${code}.xml`), "utf8"),
  );
  const parsed = engine.parse(readFileSync(join(folder, "styles", file), "utf8"), locales);
  styles.set(file, parsed);
  return parsed;
}

function answer(request: WorkerRequest): WorkerResponse {
  try {
    const { handle, description } = styleOf(request.style.file);
    const live = engine as CslEngine;
    const output = formatJob(request.job, (each) => live.render(handle, each));
    return { id: request.id, ok: true, description, output };
  } catch (error) {
    // A panic in the engine traps its instance, which can't be used again: start afresh.
    if (error instanceof WebAssembly.RuntimeError) {
      engine = undefined;
      styles.clear();
    }
    return {
      id: request.id,
      ok: false,
      message: error instanceof Error ? error.message : String(error),
      style: error instanceof StyleError,
    };
  }
}

parentPort?.on("message", (request: WorkerRequest) => {
  parentPort?.postMessage(answer(request));
});
