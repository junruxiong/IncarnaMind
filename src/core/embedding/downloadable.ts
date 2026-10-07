/**
 * A built-in model whose files are downloaded once into the data folder and
 * then loaded into a runner off the core's thread: the embedding model (see
 * ./index) and the reranking model (see ../reranking). This is its download
 * state, as `EmbeddingModelStatus` reports it, and loading it.
 */
import { join } from "node:path";
import type { EmbeddingModelFiles, EmbeddingModelSource } from "../adapters";
import type { EmbeddingModelError, EmbeddingModelStatus } from "../api";
import { downloadModel, isModelDownloaded, ModelDownloadError } from "./download";

/** Progress events are at most this often, apart from state changes. */
const PROGRESS_INTERVAL_MS = 250;

export interface DownloadableModel {
  status(): EmbeddingModelStatus;
  /** Downloaded and checked. Loading may still fail (see `load`). */
  isReady(): boolean;
  /**
   * Starts the download if the model isn't ready and isn't downloading, also
   * after a failed download. After a failed load, only `retry` starts again.
   */
  ensure(): void;
  /** Starts the download, or tries again after a failure (a failed load included). */
  retry(): EmbeddingModelStatus;
  /** Calls `listener` each time the model becomes ready. */
  onReady(listener: () => void): void;
  /**
   * Loads the model into its runner. Resolves true once it can run, or false
   * if the model isn't ready or can't start (the state is then "failed").
   */
  load(): Promise<boolean>;
  /** Stops a download (keeping what it got). The runner is the caller's to close. */
  close(): void;
}

export interface DownloadableModelOptions {
  /** What the User sees, e.g. "multilingual-e5-small". */
  name: string;
  /** The model's folder under `models/` in the data folder. */
  folder: string;
  /** The files the runner loads, relative to the model's folder. */
  files: { model: string; tokenizer: string; tokenizerConfig: string };
  maxTokens: number;
  source: EmbeddingModelSource;
  dataDir: string;
  /** Loads the downloaded files into the runner; rejects if the model can't start. */
  loadRunner(files: EmbeddingModelFiles): Promise<void>;
  emitStatus(status: EmbeddingModelStatus): void;
  reportError?: (error: unknown) => void;
}

export function createDownloadableModel(options: DownloadableModelOptions): DownloadableModel {
  const { source, emitStatus } = options;
  const reportError = options.reportError ?? ((error) => console.error(error));
  const directory = join(options.dataDir, "models", options.folder);
  const totalBytes = source.files.reduce((sum, file) => sum + file.size, 0);
  const host = new URL(source.baseUrl).host;
  const files: EmbeddingModelFiles = {
    model: join(directory, options.files.model),
    tokenizer: join(directory, options.files.tokenizer),
    tokenizerConfig: join(directory, options.files.tokenizerConfig),
    maxTokens: options.maxTokens,
  };

  const ready = isModelDownloaded(directory, source.files);
  let status: EmbeddingModelStatus = {
    name: options.name,
    host,
    state: ready ? "ready" : "not-downloaded",
    downloadedBytes: ready ? totalBytes : 0,
    totalBytes,
    error: null,
  };
  const readyListeners = new Set<() => void>();
  const lifetime = new AbortController();
  let downloading = false;
  let loading: Promise<boolean> | undefined;
  let lastProgress = 0;

  const update = (patch: Partial<EmbeddingModelStatus>) => {
    status = { ...status, ...patch };
    if (!lifetime.signal.aborted) emitStatus(status);
  };

  const fail = (error: EmbeddingModelError) => update({ state: "failed", error });

  function start(): void {
    if (downloading || lifetime.signal.aborted) return;
    downloading = true;
    update({ state: "downloading", error: null, downloadedBytes: 0 });
    void downloadModel({
      directory,
      source,
      signal: lifetime.signal,
      onProgress(downloadedBytes) {
        status = { ...status, downloadedBytes };
        const now = Date.now();
        if (now - lastProgress < PROGRESS_INTERVAL_MS) return;
        lastProgress = now;
        update({});
      },
    }).then(
      () => {
        downloading = false;
        if (lifetime.signal.aborted) return;
        update({ state: "ready", downloadedBytes: totalBytes, error: null });
        for (const listener of readyListeners) {
          try {
            listener();
          } catch (error) {
            reportError(error);
          }
        }
      },
      (error: unknown) => {
        downloading = false;
        if (lifetime.signal.aborted) return;
        if (error instanceof ModelDownloadError) fail({ kind: error.kind, message: error.message });
        else
          fail({
            kind: "storage",
            message: error instanceof Error ? error.message : String(error),
          });
      },
    );
  }

  async function loadOnce(): Promise<boolean> {
    try {
      await options.loadRunner(files);
      return true;
    } catch (error) {
      fail({ kind: "load", message: error instanceof Error ? error.message : String(error) });
      return false;
    }
  }

  return {
    status: () => status,
    isReady: () => status.state === "ready",
    ensure() {
      const { state, error } = status;
      if (state === "not-downloaded" || (state === "failed" && error?.kind !== "load")) start();
    },
    retry() {
      if (status.state !== "ready" && status.state !== "downloading") start();
      return status;
    },
    onReady(listener) {
      readyListeners.add(listener);
    },
    load() {
      if (status.state !== "ready") return Promise.resolve(false);
      loading ??= loadOnce().finally(() => {
        loading = undefined;
      });
      return loading;
    },
    close() {
      lifetime.abort();
      readyListeners.clear();
    },
  };
}
