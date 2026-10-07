/**
 * Embedding for the evaluation's core: the built-in model on a worker thread
 * (the Node counterpart of the desktop app's utility process), and the cloud
 * embedding provider a run can be given, which is reported next to it but
 * never gates.
 */
import type { Embedder, SaveEmbeddingProviderInput } from "../../src/core";
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

/**
 * The cloud embedding settings as the User would choose them in Settings
 * (`saveEmbeddingProvider`, #32): the core's own provider code embeds the
 * Passages and the searches, at the model's own vector size, recorded with
 * the vectors. A base URL means an OpenAI-compatible server instead of OpenAI.
 */
export function cloudEmbeddingProvider(
  settings: CloudEmbeddingSettings,
): SaveEmbeddingProviderInput {
  const { kind, modelId, apiKey, baseUrl } = settings;
  return baseUrl
    ? { kind: "openai-compatible", baseUrl, apiKey, modelId }
    : { kind, apiKey, modelId };
}
