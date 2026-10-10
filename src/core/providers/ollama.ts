/**
 * Ollama's own API, used for local mode: `/api/tags` to detect it and list
 * pulled models, `/api/pull` to download a model with progress. Chat goes
 * through `/api/chat` (see ./ollamaChat), with settings chosen per model
 * from `/api/show` (see ./ollamaModels).
 *
 * Pulling a model sends no User content, so it isn't a consented data flow;
 * the Privacy page lists it as traffic without User content.
 */
import type { ExternalService, OllamaPullProgress, OllamaStatus } from "../api";
import { isRecord } from "../errors";
import { LOCAL_MODELS } from "./catalog/local";
import { catalogProvider } from "./catalog/providers";

/** Where Ollama downloads the models it pulls from: its registry, not this app. */
export const OLLAMA_REGISTRY: Readonly<ExternalService> = {
  id: "https://registry.ollama.ai",
  name: "Ollama",
};

/**
 * The model one click pulls and selects: Ollama's Answers model in the
 * catalog (see ./catalog/providers), the one place the choice is made.
 * Qwen3.5-4B: Apache-2.0, multilingual (English and Chinese), calls Tools,
 * and its context needs little memory (only a quarter of its layers use full
 * attention), so a 16 GB laptop holds a 16k window. It can think, but every
 * request to it turns thinking off (see ./ollamaModels). Not the bare
 * "qwen3:4b", which now points to a thinking-only build that takes minutes
 * per Answer (docs/research/ollama-integration.md). Measured with the
 * evaluation in #67.
 *
 * Changing it: change the catalog's role, give the model an entry in
 * ./catalog/local, and set the size in the "providers.ollama.use" and
 * "providers.ollama.pulling" strings (a test checks they agree).
 */
export const RECOMMENDED_OLLAMA_MODEL: string = catalogProvider("ollama")?.roles.answers ?? "";

/** Its download in Ollama's library (the GGUF build), in GB to one decimal, as the one-click copy names it. */
export const RECOMMENDED_OLLAMA_DOWNLOAD_GB =
  Math.round(
    (LOCAL_MODELS.find((model) => model.tag === RECOMMENDED_OLLAMA_MODEL)?.download.gguf ?? 0) * 10,
  ) / 10;

const DETECT_TIMEOUT_MS = 2_000;

export async function detectOllama(baseUrl: string): Promise<OllamaStatus> {
  try {
    const response = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(DETECT_TIMEOUT_MS),
    });
    if (!response.ok) return { running: false, baseUrl };
    const body: unknown = await response.json();
    const listed = isRecord(body) && Array.isArray(body.models) ? body.models : [];
    const models = listed
      .map((model: unknown) => (isRecord(model) ? (model.name ?? model.model) : undefined))
      .filter((name): name is string => typeof name === "string");
    return { running: true, baseUrl, models, recommendedModel: RECOMMENDED_OLLAMA_MODEL };
  } catch {
    return { running: false, baseUrl };
  }
}

/** Ollama lists a model pulled without a tag as "name:latest". */
export function hasOllamaModel(models: readonly string[], model: string): boolean {
  return models.includes(model) || (!model.includes(":") && models.includes(`${model}:latest`));
}

const numberOrNull = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

/** Pulls a model, reporting each progress line Ollama streams. Resolves once Ollama reports success. */
export async function pullOllamaModel(
  baseUrl: string,
  model: string,
  onProgress: (progress: OllamaPullProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${baseUrl}/api/pull`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, stream: true }),
    signal,
  });
  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `Ollama couldn't download ${model} (HTTP ${response.status}). ${detail}`.trim(),
    );
  }

  let succeeded = false;
  const handle = (line: string) => {
    let update: unknown;
    try {
      update = JSON.parse(line);
    } catch {
      return;
    }
    if (!isRecord(update)) return;
    if (typeof update.error === "string") {
      throw new Error(`Ollama couldn't download ${model}: ${update.error}`);
    }
    const status = typeof update.status === "string" ? update.status : "";
    if (status === "success") succeeded = true;
    onProgress({
      model,
      status,
      completed: numberOrNull(update.completed),
      total: numberOrNull(update.total),
    });
  };

  // The body is newline-delimited JSON, one progress update per line.
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = done ? "" : (lines.pop() ?? "");
      for (const line of lines) if (line.trim()) handle(line);
      if (done) break;
    }
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    throw error;
  }
  if (!succeeded) throw new Error(`Ollama stopped before ${model} finished downloading.`);
}
