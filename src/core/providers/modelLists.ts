/**
 * The models each kind of chat provider offers, from its own model-list
 * endpoint, for a Question's model picker. Asking for the list sends the API
 * key and nothing of the User's; still, like the connection test, the caller
 * asks a cloud service only once the User has allowed the chat flow to it.
 * Any failure gives an empty list: the picker still has the default model.
 *
 * What isn't a chat model is left out by what the provider and the catalog
 * say, never by its name: Google lists the methods each model supports,
 * Ollama its capabilities, and the catalog the ids its sources know as
 * embedding, speech, image or video models (see ./catalog). An id none of
 * them knows is listed, its capabilities unknown. Where a list says what a
 * model can do (Anthropic's capabilities and limits, Ollama's capabilities),
 * that is kept as the server's word on it (see ./capabilities).
 */
import type { ChatProviderKind } from "../api";
import { isRecord } from "../errors";
import type { ModelFactsLayer } from "./capabilities";
import { isNonChatModel, type ModelInput, modelsProviderOfKind } from "./catalog";
import { CHATGPT_PLAN_MODELS } from "./chatgpt/codexEndpoint";
import { endpointUrl } from "./kinds";

const LIST_TIMEOUT_MS = 5_000;
/** The picker shows at most this many models per provider. */
const MAX_MODELS = 100;

export interface ModelListSpec {
  kind: ChatProviderKind;
  /** The server's URL, for "openai-compatible" and "ollama". */
  baseUrl: string | null;
  apiKey: string | null;
  /** A hosted provider's endpoint id; its first when left out. */
  endpoint?: string | null;
}

/** A model a provider lists, with what the list says it can do, if anything. */
export interface ListedModel {
  id: string;
  facts?: ModelFactsLayer;
}

async function getJson(url: string, headers: Record<string, string>): Promise<unknown> {
  const response = await fetch(url, { headers, signal: AbortSignal.timeout(LIST_TIMEOUT_MS) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

/** `data[]`, the OpenAI shape that Anthropic and OpenAI-compatible servers share. */
const dataOf = (body: unknown): Record<string, unknown>[] =>
  isRecord(body) && Array.isArray(body.data) ? body.data.filter(isRecord) : [];

const idsOf = (body: unknown): ListedModel[] =>
  dataOf(body)
    .map((model) => model.id)
    .filter((id): id is string => typeof id === "string")
    .map((id) => ({ id }));

/** Whether a capability in Anthropic's tree is supported (`{ supported: true }`); undefined when absent. */
const supported = (capabilities: Record<string, unknown>, name: string): boolean | undefined => {
  const node = capabilities[name];
  return isRecord(node) && typeof node.supported === "boolean" ? node.supported : undefined;
};

const positive = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;

/**
 * What Anthropic's `/v1/models` says of a model: its context window
 * (`max_input_tokens`), its longest reply (`max_tokens`), and from
 * `capabilities`, whether it reads images and PDFs and gives structured output.
 */
export function anthropicModelFacts(model: Record<string, unknown>): ModelFactsLayer {
  const facts: ModelFactsLayer = {};
  const context = positive(model.max_input_tokens);
  const maxOutput = positive(model.max_tokens);
  if (context) facts.context = context;
  if (maxOutput) facts.maxOutput = maxOutput;
  const capabilities = isRecord(model.capabilities) ? model.capabilities : null;
  if (capabilities) {
    const images = supported(capabilities, "image_input");
    const pdf = supported(capabilities, "pdf_input");
    if (images !== undefined) {
      const input: ModelInput[] = ["text"];
      if (images) input.push("image");
      if (pdf) input.push("pdf");
      facts.input = input;
    }
    const structured = supported(capabilities, "structured_outputs");
    if (structured !== undefined) facts.structuredOutput = structured ? "json_schema" : "none";
  }
  return facts;
}

/** What Mistral's model list says of a model: its capabilities and context window. */
export function mistralModelFacts(model: Record<string, unknown>): ModelFactsLayer {
  const facts: ModelFactsLayer = {};
  const capabilities = isRecord(model.capabilities) ? model.capabilities : null;
  if (capabilities) {
    if (typeof capabilities.function_calling === "boolean") {
      facts.tools = capabilities.function_calling;
    }
    if (typeof capabilities.vision === "boolean") {
      facts.input = capabilities.vision ? ["text", "image"] : ["text"];
    }
  }
  const context = positive(model.max_context_length);
  if (context) facts.context = context;
  return facts;
}

/** What OpenRouter's model list says of a model: its inputs, context window, longest reply, and Tools. */
export function openRouterModelFacts(model: Record<string, unknown>): ModelFactsLayer {
  const facts: ModelFactsLayer = {};
  const architecture = isRecord(model.architecture) ? model.architecture : null;
  const modalities =
    architecture && Array.isArray(architecture.input_modalities)
      ? architecture.input_modalities
      : null;
  if (modalities) {
    const input: ModelInput[] = ["text"];
    if (modalities.includes("image")) input.push("image");
    if (modalities.includes("file")) input.push("pdf");
    facts.input = input;
  }
  const context = positive(model.context_length);
  if (context) facts.context = context;
  const top = isRecord(model.top_provider) ? model.top_provider : null;
  const maxOutput = top ? positive(top.max_completion_tokens) : undefined;
  if (maxOutput) facts.maxOutput = maxOutput;
  if (Array.isArray(model.supported_parameters)) {
    facts.tools = model.supported_parameters.includes("tools");
  }
  return facts;
}

/** Whether a Mistral list entry chats: an embedding or moderation model says it doesn't. */
const mistralChats = (model: Record<string, unknown>) =>
  !isRecord(model.capabilities) || model.capabilities.completion_chat !== false;

/** The models of an OpenAI-shaped list, with facts read from each entry when the provider gives them. */
const listed = (
  body: unknown,
  facts?: (model: Record<string, unknown>) => ModelFactsLayer,
  keep: (model: Record<string, unknown>) => boolean = () => true,
): ListedModel[] =>
  dataOf(body).flatMap((model): ListedModel[] => {
    if (typeof model.id !== "string" || !keep(model)) return [];
    const found = facts?.(model);
    return found && Object.keys(found).length > 0
      ? [{ id: model.id, facts: found }]
      : [{ id: model.id }];
  });

async function fetchModels({ kind, baseUrl, apiKey, endpoint }: ModelListSpec) {
  const bearer: Record<string, string> = apiKey ? { authorization: `Bearer ${apiKey}` } : {};
  const hosted = endpointUrl(kind, endpoint);
  switch (kind) {
    case "chatgpt":
      // The plan's endpoint has no model list: these are the ones the Codex CLI offers.
      return CHATGPT_PLAN_MODELS.map((model) => ({ id: model.id }));
    case "ollama": {
      const body = await getJson(`${baseUrl}/api/tags`, {});
      const models = isRecord(body) && Array.isArray(body.models) ? body.models : [];
      return (
        models
          .filter(isRecord)
          // Ollama 0.40 lists each model's capabilities: one that can't complete (e.g. an embedding model) can't answer.
          .filter(
            (model) =>
              !Array.isArray(model.capabilities) || model.capabilities.includes("completion"),
          )
          .flatMap((model): ListedModel[] => {
            const name = model.name ?? model.model;
            if (typeof name !== "string") return [];
            const capabilities = Array.isArray(model.capabilities) ? model.capabilities : null;
            return [
              capabilities
                ? { id: name, facts: { tools: capabilities.includes("tools") } }
                : { id: name },
            ];
          })
      );
    }
    case "openai-compatible":
      return idsOf(await getJson(`${baseUrl}/models`, bearer));
    case "deepseek":
    case "qwen":
    case "kimi":
    case "glm":
    case "xai":
      if (!apiKey) return [];
      return idsOf(await getJson(`${hosted}/models`, bearer));
    case "siliconflow":
      if (!apiKey) return [];
      return idsOf(await getJson(`${hosted}/models?type=text&sub_type=chat`, bearer));
    case "mistral":
      if (!apiKey) return [];
      return listed(await getJson(`${hosted}/models`, bearer), mistralModelFacts, mistralChats);
    case "openrouter":
      // OpenRouter's list is public and says much about each model.
      return listed(await getJson(`${hosted}/models`, bearer), openRouterModelFacts);
    case "openai":
      if (!apiKey) return [];
      return idsOf(await getJson(`${hosted}/models`, bearer));
    case "anthropic": {
      if (!apiKey) return [];
      const body = await getJson(`${hosted}/models?limit=100`, {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      });
      return dataOf(body).flatMap((model): ListedModel[] =>
        typeof model.id === "string" ? [{ id: model.id, facts: anthropicModelFacts(model) }] : [],
      );
    }
    case "google": {
      if (!apiKey) return [];
      const body = await getJson(`${hosted}/models?pageSize=200`, { "x-goog-api-key": apiKey });
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
        .map((name) => ({ id: name.replace(/^models\//, "") }));
    }
  }
}

/** The chat models a provider lists, sorted; empty if it can't be reached. */
export async function listProviderModels(spec: ModelListSpec): Promise<ListedModel[]> {
  try {
    const catalogId = modelsProviderOfKind(spec.kind, spec.endpoint);
    const byId = new Map<string, ListedModel>();
    for (const model of await fetchModels(spec)) {
      if (!byId.has(model.id) && !isNonChatModel(catalogId, model.id)) byId.set(model.id, model);
    }
    return [...byId.values()].sort((a, b) => a.id.localeCompare(b.id)).slice(0, MAX_MODELS);
  } catch {
    return [];
  }
}
