/**
 * Embedders for the evaluation's core: the built-in model on a worker thread
 * (the Node counterpart of the desktop app's utility process), and cloud
 * embedding models, which are reported next to it but never gate.
 */
import { createGoogle } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { embed } from "ai";
import { BUILT_IN_EMBEDDING_MODEL, type Embedder, type EmbeddingModelSource } from "../../src/core";
import { createChannelEmbedder, type EmbedderResponse } from "../../src/core/embedding/channel";
import type { CloudEmbeddingSettings } from "./config";
import startEmbedderWorker from "./embedderWorker?nodeWorker";

/** The built-in model, run on a worker thread so it never blocks the core's thread. */
export function createWorkerEmbedder(): Embedder {
  return createChannelEmbedder(() => {
    const worker = startEmbedderWorker({ name: "incarnamind-eval-embedding" });
    // An uncaught error in the worker is followed by "exit", which fails what is pending.
    worker.on("error", (error) => console.error("The embedding worker failed:", error));
    return {
      send: (request) => worker.postMessage(request),
      onResponse: (listener) =>
        worker.on("message", (response: EmbedderResponse) => listener(response)),
      onExit: (listener) => worker.on("exit", (code) => listener(code)),
      stop: () => void worker.terminate(),
    };
  });
}

const { passagePrefix, queryPrefix, dimensions } = BUILT_IN_EMBEDDING_MODEL;

/**
 * A cloud embedding model in place of the built-in one. The core still adds
 * e5's "passage: " and "query: " prefixes; they are taken off here, and tell
 * Google's models which task a text is for. Vectors are asked for at the
 * built-in model's 384 dimensions, which both providers' current models
 * support by shortening (Matryoshka) vectors; the core refuses any other size.
 */
export function createCloudEmbedder(settings: CloudEmbeddingSettings): Embedder {
  const model =
    settings.kind === "openai"
      ? createOpenAI({ apiKey: settings.apiKey, baseURL: settings.baseUrl ?? undefined }).embedding(
          settings.modelId,
        )
      : createGoogle({ apiKey: settings.apiKey }).embedding(settings.modelId);
  return {
    async load() {},
    async embed(text) {
      const isQuery = text.startsWith(queryPrefix);
      const value = isQuery
        ? text.slice(queryPrefix.length)
        : text.startsWith(passagePrefix)
          ? text.slice(passagePrefix.length)
          : text;
      const { embedding } = await embed({
        model,
        value,
        maxRetries: 5,
        providerOptions:
          settings.kind === "openai"
            ? { openai: { dimensions } }
            : {
                google: {
                  outputDimensionality: dimensions,
                  taskType: isQuery ? "RETRIEVAL_QUERY" : "RETRIEVAL_DOCUMENT",
                },
              },
      });
      return Float32Array.from(embedding);
    },
    close() {},
  };
}

/** A model source with nothing to download, named after the cloud API, for a cloud embedder. */
export function cloudModelSource(settings: CloudEmbeddingSettings): EmbeddingModelSource {
  const baseUrl =
    settings.baseUrl ??
    (settings.kind === "openai"
      ? "https://api.openai.com/"
      : "https://generativelanguage.googleapis.com/");
  return { baseUrl: baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`, files: [] };
}
