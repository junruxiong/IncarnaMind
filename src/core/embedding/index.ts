/**
 * The built-in embedding model as the rest of the core sees it: its download
 * state, loading it into the `Embedder`, and turning Passages and queries into
 * vectors with the model's prefixes and the shared text normalisation.
 */
import { join } from "node:path";
import { normaliseText } from "../../shared/text";
import type { Embedder, EmbeddingModelSource } from "../adapters";
import type { EmbeddingModelError, EmbeddingModelStatus } from "../api";
import { downloadModel, isModelDownloaded, ModelDownloadError } from "./download";
import type { EmbeddingModelDefinition } from "./model";

export { BUILT_IN_EMBEDDING_MODEL, type EmbeddingModelDefinition } from "./model";

/** Progress events are at most this often, apart from state changes. */
const PROGRESS_INTERVAL_MS = 250;

export interface EmbeddingModel {
  readonly id: string;
  readonly dimensions: number;
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
   * Loads the model into the embedder. Resolves true once it can embed, or
   * false if the model isn't ready or can't start (the state is then "failed").
   */
  load(): Promise<boolean>;
  /** A Passage's vector: the Document's name, then the Passage, normalised. L2-normalised. */
  embedPassage(documentName: string, text: string): Promise<Float32Array>;
  /** A search query's vector. L2-normalised. */
  embedQuery(query: string): Promise<Float32Array>;
  /**
   * Stops the embedder, freeing its memory, e.g. while another provider
   * embeds. The next `load` starts it again.
   */
  unload(): void;
  /** Stops a download (keeping what it got) and the embedder. */
  close(): void;
}

export interface EmbeddingModelOptions {
  definition: EmbeddingModelDefinition;
  /** Overrides the definition's source (tests). */
  source?: EmbeddingModelSource;
  dataDir: string;
  embedder: Embedder;
  emitStatus(status: EmbeddingModelStatus): void;
  reportError?: (error: unknown) => void;
}

/** The text the model reads for a Passage. The name goes here, not into the stored Passage, so renaming doesn't change Passages. */
export function passageEmbeddingText(
  definition: EmbeddingModelDefinition,
  documentName: string,
  text: string,
): string {
  return `${definition.passagePrefix}${normaliseText(documentName)}\n${normaliseText(text)}`;
}

export function createEmbeddingModel(options: EmbeddingModelOptions): EmbeddingModel {
  const { definition, dataDir, embedder, emitStatus } = options;
  const reportError = options.reportError ?? ((error) => console.error(error));
  const source = options.source ?? definition.source;
  const directory = join(dataDir, "models", definition.folder);
  const totalBytes = source.files.reduce((sum, file) => sum + file.size, 0);
  const host = new URL(source.baseUrl).host;
  const files = {
    model: join(directory, definition.files.model),
    tokenizer: join(directory, definition.files.tokenizer),
    tokenizerConfig: join(directory, definition.files.tokenizerConfig),
    maxTokens: definition.maxTokens,
  };

  const ready = isModelDownloaded(directory, source.files);
  let status: EmbeddingModelStatus = {
    name: definition.name,
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
      await embedder.load(files);
      return true;
    } catch (error) {
      fail({ kind: "load", message: error instanceof Error ? error.message : String(error) });
      return false;
    }
  }

  async function vectorOf(text: string): Promise<Float32Array> {
    const vector = await embedder.embed(text);
    if (vector.length !== definition.dimensions) {
      throw new Error(
        `The embedding model returned ${vector.length} dimensions instead of ${definition.dimensions}.`,
      );
    }
    let norm = 0;
    for (const value of vector) norm += value * value;
    norm = Math.sqrt(norm);
    if (!(norm > 0)) throw new Error("The embedding model returned an empty vector.");
    return vector.map((value) => value / norm);
  }

  return {
    id: definition.id,
    dimensions: definition.dimensions,
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
    embedPassage: (documentName, text) =>
      vectorOf(passageEmbeddingText(definition, documentName, text)),
    embedQuery: (query) => vectorOf(`${definition.queryPrefix}${normaliseText(query)}`),
    unload() {
      embedder.close();
    },
    close() {
      lifetime.abort();
      readyListeners.clear();
      embedder.close();
    },
  };
}
