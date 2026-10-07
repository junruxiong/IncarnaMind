/**
 * The models each kind of chat provider offers, from its own model-list
 * endpoint, for a Question's model picker. Asking for the list sends the API
 * key and nothing of the User's; still, like the connection test, the caller
 * asks a cloud service only once the User has allowed the chat flow to it.
 * Any failure gives an empty list: the picker still has the default model.
 */
import type { ChatProviderKind } from "../api";
import { isRecord } from "../errors";
import { CHATGPT_PLAN_MODELS } from "./chatgpt/codexEndpoint";

const LIST_TIMEOUT_MS = 5_000;
/** The picker shows at most this many models per provider. */
const MAX_MODELS = 100;

export interface ModelListSpec {
  kind: ChatProviderKind;
  baseUrl: string | null;
  apiKey: string | null;
}

/** Model kinds that can't answer a Question: embeddings, speech, images, moderation. */
const NOT_CHAT = /embed|rerank|moderation|whisper|tts|transcribe|audio|realtime|image|dall-e/i;

async function getJson(url: string, headers: Record<string, string>): Promise<unknown> {
  const response = await fetch(url, { headers, signal: AbortSignal.timeout(LIST_TIMEOUT_MS) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

/** `data[].id`, the OpenAI shape that Anthropic and OpenAI-compatible servers share. */
function dataIds(body: unknown): string[] {
  const data = isRecord(body) && Array.isArray(body.data) ? body.data : [];
  return data
    .map((model: unknown) => (isRecord(model) ? model.id : undefined))
    .filter((id): id is string => typeof id === "string");
}

async function fetchModels({ kind, baseUrl, apiKey }: ModelListSpec): Promise<string[]> {
  const bearer: Record<string, string> = apiKey ? { authorization: `Bearer ${apiKey}` } : {};
  switch (kind) {
    case "chatgpt":
      // The plan's endpoint has no model list: these are the ones the Codex CLI offers.
      return CHATGPT_PLAN_MODELS.map((model) => model.id);
    case "ollama": {
      const body = await getJson(`${baseUrl}/api/tags`, {});
      const models = isRecord(body) && Array.isArray(body.models) ? body.models : [];
      return models
        .map((model: unknown) => (isRecord(model) ? (model.name ?? model.model) : undefined))
        .filter((name): name is string => typeof name === "string");
    }
    case "openai-compatible":
      return dataIds(await getJson(`${baseUrl}/models`, bearer));
    case "openai":
      if (!apiKey) return [];
      return dataIds(await getJson("https://api.openai.com/v1/models", bearer)).filter((id) =>
        /^(gpt-|o\d|chatgpt-)/.test(id),
      );
    case "anthropic":
      if (!apiKey) return [];
      return dataIds(
        await getJson("https://api.anthropic.com/v1/models?limit=100", {
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
        }),
      );
    case "google": {
      if (!apiKey) return [];
      const body = await getJson(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        { "x-goog-api-key": apiKey },
      );
      const models = isRecord(body) && Array.isArray(body.models) ? body.models : [];
      return models
        .filter(
          (model: unknown) =>
            isRecord(model) &&
            Array.isArray(model.supportedGenerationMethods) &&
            model.supportedGenerationMethods.includes("generateContent"),
        )
        .map((model: unknown) => (isRecord(model) ? model.name : undefined))
        .filter((name): name is string => typeof name === "string")
        .map((name) => name.replace(/^models\//, ""));
    }
  }
}

/** The chat models a provider lists, sorted; empty if it can't be reached. */
export async function listProviderModels(spec: ModelListSpec): Promise<string[]> {
  try {
    const models = (await fetchModels(spec)).filter((id) => !NOT_CHAT.test(id));
    return [...new Set(models)].sort((a, b) => a.localeCompare(b)).slice(0, MAX_MODELS);
  } catch {
    return [];
  }
}
