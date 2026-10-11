/**
 * Ollama cloud models: Ollama runs them on ollama.com. The request goes to the
 * Ollama on this computer, and that Ollama forwards it, so the address alone
 * says nothing about where it ends up. Kept apart from ./ollamaModels so the
 * window can use these too.
 */
import type { ExternalService } from "../api";

/** Where an Ollama cloud model's requests end up. */
export const OLLAMA_CLOUD: Readonly<ExternalService> = {
  id: "https://ollama.com",
  name: "ollama.com",
};

/**
 * Whether the name is one Ollama gives its cloud models: "gpt-oss:120b-cloud",
 * "gpt-oss:cloud". Read before anything Ollama says, because a false "local"
 * would send data away unasked.
 */
export const isCloudModelName = (model: string): boolean => /[:-]cloud$/i.test(model.trim());

/**
 * Whether Ollama says a model runs remotely: `remote_host` or `remote_model`
 * in its `/api/tags` entry or its `/api/show` answer.
 */
export function isRemoteEntry(entry: Record<string, unknown>): boolean {
  const set = (value: unknown) => typeof value === "string" && value.trim() !== "";
  return set(entry.remote_host) || set(entry.remote_model);
}
