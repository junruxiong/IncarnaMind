/**
 * The reranked modes: keyword search's top 60 (keyword + rerank), what the
 * search Tool hands a reranker by default, or keyword search's top 10 and
 * vector search's top 10, each Passage once (see `rerankCandidates` in
 * ./retrieval), as it does with embeddings on, or the other searches'
 * candidates (./searches), reranked by a built-in reranking candidate
 * (src/core/reranking/model.ts), as the search Tool reranks them. It uses
 * the core's own reranking code (download and check, what the model reads,
 * its scores) with the model on a worker thread, outside the evaluation's
 * core: one run compares every candidate given in INCARNAMIND_EVAL_RERANK
 * over the same searches, one model and one mode at a time, so each mode's
 * timings are its own.
 */
import type { CrossEncoder, PassageSearchResult, RerankingModelDefinition } from "../../src/core";
import {
  type CrossEncoderResponse,
  createChannelCrossEncoder,
} from "../../src/core/reranking/channel";
import { createRerankingModel, downloadSize } from "../../src/core/reranking/index";
import type { Log } from "./log";
import startRerankerWorker from "./rerankerWorker?nodeWorker";

/** A reranking model, run on a worker thread so it never blocks the core's thread. */
export function createWorkerCrossEncoder(): CrossEncoder {
  return createChannelCrossEncoder(() => {
    const worker = startRerankerWorker({ name: "incarnamind-eval-reranking" });
    worker.on("error", (error) => console.error("The reranking worker failed:", error));
    return {
      send: (request) => worker.postMessage(request),
      onResponse: (listener) =>
        worker.on("message", (response: CrossEncoderResponse) => listener(response)),
      onExit: (listener) => worker.on("exit", (code) => listener(code)),
      stop: () => void worker.terminate(),
    };
  });
}

/** How long something took per search, in milliseconds. */
export interface Latency {
  queries: number;
  mean: number;
  median: number;
  p95: number;
  max: number;
}

/** What the report says about a candidate in one reranked mode. */
export interface RerankerInfo {
  /** The mode it adds, e.g. "rerank:mmarco-minilm" or "keyword-rerank:mmarco-minilm". */
  mode: string;
  name: string;
  licence: string;
  downloadBytes: number;
  /** Loading the model, the first time it reranks. */
  loadSeconds: number;
  /** Reranking one search's candidates, in milliseconds, after the first. */
  latency: Latency;
  /**
   * Finding one search's candidates, before they are reranked: the searches
   * the mode runs, without a chat model's call (see `QueryModelCost` in
   * ./rewrites). Absent in reports from before it was measured.
   */
  candidates?: Latency;
}

export interface OpenReranker {
  mode: string;
  /** The Passages in the model's order. */
  rerank(query: string, passages: readonly PassageSearchResult[]): Promise<PassageSearchResult[]>;
  /** What the report says about it, from the calls so far. */
  info(): RerankerInfo;
  close(): void;
}

const quantile = (sorted: readonly number[], share: number) =>
  sorted[Math.min(sorted.length - 1, Math.ceil(share * sorted.length) - 1)] ?? 0;

/** The mean, median, 95th percentile and slowest of some timings, in milliseconds. */
export function latencyOf(timings: readonly number[]): Latency {
  const sorted = [...timings].sort((a, b) => a - b);
  return {
    queries: sorted.length,
    mean: sorted.reduce((sum, ms) => sum + ms, 0) / Math.max(1, sorted.length),
    median: quantile(sorted, 0.5),
    p95: quantile(sorted, 0.95),
    max: sorted.at(-1) ?? 0,
  };
}

/**
 * Downloads (once, into the evaluation's model cache, checked against the
 * pinned hashes) and opens a candidate, for one reranked mode: hybrid
 * search's candidates unless another mode is named (see `rerankMode` in
 * ./retrieval).
 */
export async function openReranker(
  definition: RerankingModelDefinition,
  cacheDir: string,
  log: Log,
  mode = `rerank:${definition.id}`,
): Promise<OpenReranker> {
  let lastLogged = 0;
  const ready = Promise.withResolvers<void>();
  const model = createRerankingModel({
    definition,
    // The model cache's models/ is where the core would keep it in a data folder.
    dataDir: cacheDir,
    crossEncoder: createWorkerCrossEncoder(),
    emitStatus: (status) => {
      if (status.state === "ready") ready.resolve();
      else if (status.state === "failed") {
        ready.reject(
          new Error(
            `The reranking model ${definition.name} couldn't be downloaded from ${status.host}: ${status.error?.message ?? "unknown error"}. Run once with a connection, and it is kept in ${cacheDir} for later runs.`,
          ),
        );
      } else if (status.state === "downloading" && Date.now() - lastLogged > 5000) {
        lastLogged = Date.now();
        const mb = (bytes: number) => Math.round(bytes / 1e6);
        log(
          `Downloading ${definition.name}: ${mb(status.downloadedBytes)} of ${mb(status.totalBytes)} MB`,
        );
      }
    },
  });
  if (model.isReady()) ready.resolve();
  else model.retry();
  try {
    await ready.promise;
  } catch (error) {
    model.close();
    throw error;
  }

  let loadSeconds = 0;
  const latencies: number[] = [];
  return {
    mode,
    async rerank(query, passages) {
      const started = performance.now();
      const ranked = await model.rerank(query, passages);
      const ms = performance.now() - started;
      // The first call loads the model.
      if (loadSeconds === 0) loadSeconds = ms / 1000;
      else latencies.push(ms);
      if (!ranked) throw new Error(`${definition.name} isn't ready.`);
      return ranked.map(({ score: _score, ...passage }) => passage);
    },
    info() {
      return {
        mode,
        name: definition.name,
        licence: definition.licence,
        downloadBytes: downloadSize(definition),
        loadSeconds,
        latency: latencyOf(latencies),
      };
    },
    close() {
      model.close();
    },
  };
}
