/**
 * The core's `Embedder`, backed by an Electron utility process
 * (./embedderProcess), so the embedding model never runs on the main process.
 * The process starts on the first `load`. If it dies (a crash, or out of
 * memory), pending requests fail, and the core's next `load` starts a new one.
 */
import { type UtilityProcess, utilityProcess } from "electron";
import type { Embedder, EmbeddingModelFiles } from "../core";
import embedderProcessPath from "./embedderProcess?modulePath";
import type { EmbedderCommand, EmbedderRequest, EmbedderResponse } from "./embedderProtocol";

export interface UtilityProcessEmbedderOptions {
  /** Runs the deterministic fake model instead of the real one (smoke tests). */
  fake?: boolean;
}

type Pending = { resolve(response: EmbedderResponse): void; reject(error: Error): void };

export function createUtilityProcessEmbedder(
  options: UtilityProcessEmbedderOptions = {},
): Embedder {
  let child: UtilityProcess | undefined;
  /** The files the running process has loaded. */
  let loadedKey: string | undefined;
  let loading: { key: string; done: Promise<void> } | undefined;
  const pending = new Map<number, Pending>();
  let nextId = 1;

  function start(): UtilityProcess {
    const started = utilityProcess.fork(embedderProcessPath, options.fake ? ["--fake"] : [], {
      serviceName: "IncarnaMind embedding",
      stdio: "inherit",
    });
    started.on("message", (response: EmbedderResponse) => {
      const waiting = pending.get(response.id);
      pending.delete(response.id);
      waiting?.resolve(response);
    });
    started.on("exit", (code) => {
      if (child !== started) return;
      child = undefined;
      loadedKey = undefined;
      const error = new Error(`The embedding process stopped (exit code ${code}).`);
      for (const waiting of pending.values()) waiting.reject(error);
      pending.clear();
    });
    return started;
  }

  function request(command: EmbedderCommand): Promise<EmbedderResponse> {
    child ??= start();
    const target = child;
    const id = nextId++;
    return new Promise<EmbedderResponse>((resolve, reject) => {
      pending.set(id, { resolve, reject });
      target.postMessage({ ...command, id } satisfies EmbedderRequest);
    }).then((response) => {
      if (response.error !== undefined) throw new Error(response.error);
      return response;
    });
  }

  async function loadFiles(files: EmbeddingModelFiles, key: string): Promise<void> {
    await request({ type: "load", files });
    loadedKey = key;
  }

  return {
    async load(files) {
      const key = JSON.stringify(files);
      if (child && loadedKey === key) return;
      if (loading?.key !== key) {
        const done = loadFiles(files, key).finally(() => {
          if (loading?.done === done) loading = undefined;
        });
        loading = { key, done };
      }
      await loading.done;
    },

    async embed(text) {
      if (!child || loadedKey === undefined) throw new Error("The embedding model isn't loaded.");
      const { vector } = await request({ type: "embed", text });
      if (!vector) throw new Error("The embedding process returned no vector.");
      return new Float32Array(vector);
    },

    close() {
      const stopping = child;
      child = undefined;
      loadedKey = undefined;
      stopping?.kill();
    },
  };
}
