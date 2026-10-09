import { totalmem } from "node:os";
import { isRecord, TaggingNotReadyError } from "../errors";
import { JevRequestError } from "../providers/jev";
import { detectOllama, hasOllamaModel } from "../providers/ollama";
import { decisionGroupClassifier, type GroupClassifier } from "./classifier";
import type { ClassificationModel } from "./types";

// Product policy, not a model benchmark: ignore cold loads and require two slow calls.
export const AUTO_CLASSIFICATION = {
  slowMs: 8_000,
  slowCalls: 2,
  textTimeoutMs: 45_000,
  smallMemoryBytes: 12 * 2 ** 30,
};
const TEXT = "tev1:4b";
const FAST = "tev1:0.8b";
const VISION = "clef-flash";

/** One controller per local endpoint, used sequentially by the background queue. */
export function automaticGroupClassifier(
  baseUrl: string,
  options: { memoryBytes?: number; clock?: () => number; textTimeoutMs?: number } = {},
): GroupClassifier {
  const clock = options.clock ?? (() => performance.now());
  let fallback: "slow" | "memory" | undefined =
    (options.memoryBytes ?? totalmem()) < AUTO_CLASSIFICATION.smallMemoryBytes
      ? "memory"
      : undefined;
  let slowCalls = 0;
  let lastModel: string | null = null;
  const classifier: GroupClassifier = {
    local: true,
    pageImages: "auto",
    async decide(groups, excerpt, signal, images = []) {
      return (await this.organize(groups, [], excerpt, signal, images)).groupId;
    },
    async organize(groups, tags, excerpt, signal, images = []) {
      signal.throwIfAborted();
      const status = await detectOllama(baseUrl);
      signal.throwIfAborted();
      if (!status.running)
        throw new TaggingNotReadyError("Start Ollama to use automatic local classification.");
      const models = status.models;
      const visual = excerpt.kind === "pdf" && images.length > 0;
      let model = visual ? VISION : fallback ? FAST : TEXT;
      let reason: ClassificationModel["reason"] = visual ? "visual" : (fallback ?? "text");
      if (model === TEXT && !hasOllamaModel(models, TEXT)) {
        model = FAST;
        reason = "unavailable";
      }
      const requireModel = (id: string) => {
        if (!hasOllamaModel(models, id))
          throw new TaggingNotReadyError(
            id === VISION
              ? "This PDF needs page images. Install clef-flash in Ollama, then classify it again, or assign a group manually."
              : `Install ${id} in Ollama, then classify again, or select a model manually.`,
          );
      };
      const loaded = await fetch(`${baseUrl}/api/ps`, {
        signal: AbortSignal.any([signal, AbortSignal.timeout(2_000)]),
      });
      if (!loaded.ok)
        throw new Error("Ollama could not report its loaded models. Try classification again.");
      const body: unknown = await loaded.json();
      const warm =
        isRecord(body) &&
        Array.isArray(body.models) &&
        body.models.some(
          (item: unknown) => isRecord(item) && (item.name === TEXT || item.model === TEXT),
        );
      const deadline = AbortSignal.timeout(
        options.textTimeoutMs ?? AUTO_CLASSIFICATION.textTimeoutMs,
      );
      const run = async () => {
        requireModel(model);
        // Release only the previous model used by this Auto controller; leave other models alone.
        if (lastModel && lastModel !== model) {
          const response = await fetch(`${baseUrl}/api/generate`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ model: lastModel, keep_alive: 0 }),
            signal: AbortSignal.any([signal, AbortSignal.timeout(10_000)]),
          });
          if (!response.ok)
            throw new Error(
              "Ollama could not release the previous classification model. Try again.",
            );
          lastModel = null;
        }
        signal.throwIfAborted();
        lastModel = model;
        const chosen = await decisionGroupClassifier({
          baseUrl,
          apiKey: "ollama",
          model,
          local: true,
          usePageImages: visual,
        }).organize(
          groups,
          tags,
          excerpt,
          model === TEXT ? AbortSignal.any([signal, deadline]) : signal,
          images,
        );
        classifier.model = { id: model, images: visual, reason };
        return chosen;
      };
      const started = clock();
      try {
        const chosen = await run();
        if (model === TEXT) {
          slowCalls = warm && clock() - started > AUTO_CLASSIFICATION.slowMs ? slowCalls + 1 : 0;
          if (slowCalls >= AUTO_CLASSIFICATION.slowCalls) fallback = "slow";
        }
        return chosen;
      } catch (error) {
        signal.throwIfAborted();
        const memoryFailure =
          error instanceof JevRequestError &&
          (error.status ?? 0) >= 500 &&
          /out of memory|not enough (?:system )?memory|requires more (?:system )?memory|failed to allocate/i.test(
            error.message,
          );
        if ((memoryFailure || deadline.aborted) && model === TEXT) {
          fallback = memoryFailure ? "memory" : "slow";
          model = FAST;
          reason = fallback;
          return run();
        }
        if (memoryFailure && visual)
          throw new TaggingNotReadyError(
            "Clef-Flash needs more available memory for this PDF. Close other models and retry, or assign a group manually.",
          );
        throw error;
      }
    },
  };
  return classifier;
}
