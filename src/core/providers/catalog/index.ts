/**
 * The provider catalog: what IncarnaMind knows about model providers and
 * their models, in one place, so that adding or updating a provider or a
 * model is a change of data.
 *
 * - The providers are written by hand (./providers).
 * - Their models' facts are generated from models.dev, cross-checked against
 *   LiteLLM (`node scripts/catalogModels.ts` writes ./models.json), and
 *   corrected by hand (./overrides).
 * - Models recommended for Ollama by this computer's memory are in ./local.
 *
 * The catalog ships with the app and changes only with its releases: nothing
 * fetches it at runtime. A provider's own model list, fetched after consent
 * (see ../modelLists), brings new model ids on the day they launch; until the
 * catalog knows them, their capabilities are unknown. What the catalog says
 * gives way to what the User set and what the server reports (see ../capabilities).
 */
import type { ChatProviderKind } from "../../api";
import generated from "./models.json";
import { OVERRIDES } from "./overrides";
import type { CatalogModel, GeneratedProviderModels, ModelFacts } from "./types";

export { LOCAL_MODEL_TIERS, LOCAL_MODELS, recommendedLocalModels } from "./local";
export {
  CATALOG_PROVIDERS,
  type CatalogProviderId,
  catalogProvider,
  catalogProviderOfKind,
  endpointOf,
} from "./providers";
export type * from "./types";

/** The generated facts; their sources' licence notice is resources/notices/model-catalog.txt. */
const GENERATED = generated as unknown as { providers: Record<string, GeneratedProviderModels> };

/** A generated model with the hand-written corrections: its own over its provider's. */
const corrected = (providerId: string, model: CatalogModel): CatalogModel => ({
  ...model,
  ...OVERRIDES.providers[providerId],
  ...OVERRIDES.models[providerId]?.[model.id],
});

const MODELS = new Map(
  Object.entries(GENERATED.providers).map(([providerId, { models }]) => [
    providerId,
    new Map(models.map((model) => [model.id, corrected(providerId, model)])),
  ]),
);

const NON_CHAT = new Map(
  Object.entries(GENERATED.providers).map(([providerId, { nonChat }]) => [
    providerId,
    new Set(nonChat),
  ]),
);

/** A model id without a date suffix: "gpt-4o-2024-08-06" is "gpt-4o", "claude-haiku-4-5-20251001" is "claude-haiku-4-5". */
export const withoutDateSuffix = (modelId: string) =>
  modelId.replace(/-(?:\d{4}-\d{2}-\d{2}|\d{8})$/, "");

/** The catalog providers whose models the generated facts cover. */
export const providersWithModels = (): string[] => [...MODELS.keys()];

/** A provider's models in the catalog. */
export const catalogModels = (providerId: string): CatalogModel[] => [
  ...(MODELS.get(providerId)?.values() ?? []),
];

/**
 * A model's catalog entry: by its exact id, else by its id without a date
 * suffix (a dated snapshot of a model the catalog knows). Nothing further is
 * guessed from the name.
 */
export function catalogModel(
  providerId: string | null | undefined,
  modelId: string,
): CatalogModel | undefined {
  const models = providerId ? MODELS.get(providerId) : undefined;
  return models?.get(modelId) ?? models?.get(withoutDateSuffix(modelId));
}

/** What the catalog says a model can do and costs; undefined for a model it doesn't know. */
export function catalogFacts(
  providerId: string | null | undefined,
  modelId: string,
): Partial<ModelFacts> | undefined {
  const model = catalogModel(providerId, modelId);
  if (!model) return undefined;
  const { id: _id, name: _name, status: _status, ...facts } = model;
  return facts;
}

/** Whether the provider's list has this id as something other than a chat model (an embedding model, speech, images…). */
export function isNonChatModel(providerId: string | null | undefined, modelId: string): boolean {
  const ids = providerId ? NON_CHAT.get(providerId) : undefined;
  return !!ids && (ids.has(modelId) || ids.has(withoutDateSuffix(modelId)));
}

/**
 * The model set a provider kind uses on an endpoint: the kind's own, and
 * OpenAI's for the ChatGPT plan, which serves OpenAI's models. A provider whose
 * regions differ in their models has a set for each region after the first
 * ("qwen/intl"); an endpoint without a set of its own, or none given, uses the
 * provider's.
 */
export function modelsProviderOfKind(kind: ChatProviderKind, endpoint?: string | null): string {
  if (kind === "chatgpt") return "openai";
  const regional = endpoint ? `${kind}/${endpoint}` : undefined;
  return regional && MODELS.has(regional) ? regional : kind;
}
