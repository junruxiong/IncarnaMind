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
import type { CannotAnswerReason, ChatProviderKind, OtherModel } from "../../api";
import generated from "./models.json";
import { OVERRIDES } from "./overrides";
import { catalogProvider, catalogProviderOf, kindOfCatalogId } from "./providers";
import type { CatalogModel, GeneratedProviderModels, ModelFacts, Role } from "./types";

export { LOCAL_MODEL_TIERS, LOCAL_MODELS, recommendedLocalModels } from "./local";
export {
  CATALOG_PROVIDERS,
  type CatalogProviderId,
  catalogProvider,
  catalogProviderOf,
  catalogProviderOfKind,
  endpointOf,
  isPreset,
  kindOfCatalogId,
  untestedForCitations,
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

const CANNOT_ANSWER = new Map(
  Object.entries(GENERATED.providers).map(([providerId, { cannotAnswer }]) => [
    providerId,
    new Map<string, CannotAnswerReason>([
      ...Object.entries(cannotAnswer ?? {}),
      ...Object.entries(OVERRIDES.cannotAnswer[providerId] ?? {}),
    ]),
  ]),
);

/** Which API other than chat a model needs, when a source says; a dated snapshot has its model's. */
export function cannotAnswerReason(
  providerId: string | null | undefined,
  modelId: string,
): CannotAnswerReason | undefined {
  const reasons = providerId ? CANNOT_ANSWER.get(providerId) : undefined;
  return reasons?.get(modelId) ?? reasons?.get(withoutDateSuffix(modelId));
}

/**
 * Where the catalog puts a listed id: in the menu by default (undefined), or
 * under Other models. In the menu: a chat model it knows by that id (or a
 * dated snapshot of one, when `listed` doesn't also have the model itself)
 * that isn't deprecated. A provider the catalog has no models for (a server
 * the User gives) offers everything it lists. `shuttingDown`: the provider
 * has announced the model's end.
 */
export function otherModel(
  providerId: string,
  modelId: string,
  listed: ReadonlySet<string>,
  shuttingDown = false,
): Omit<OtherModel, "id"> | undefined {
  const models = MODELS.get(providerId);
  if (!models) return undefined;
  const reason = cannotAnswerReason(providerId, modelId);
  if (reason) return { why: "cannot-answer", reason };
  const known = models.get(modelId);
  const base = withoutDateSuffix(modelId);
  const snapshot = !known && listed.has(base) && models.has(base);
  const model = known ?? (snapshot ? undefined : models.get(base));
  if (snapshot || shuttingDown || model?.status === "deprecated") return { why: "older" };
  return model ? undefined : { why: "unknown" };
}

/**
 * The model set a provider kind uses on an endpoint: the kind's own, or its
 * preset's (`catalogId`), OpenAI's for the ChatGPT plan, which serves
 * OpenAI's models, and Anthropic's for a Claude account. A provider whose
 * regions differ in their models has a set for each region after the first
 * ("qwen/intl"); an endpoint without a set of its own, or none given, uses
 * the provider's.
 */
export function modelsProviderOfKind(
  kind: ChatProviderKind,
  endpoint?: string | null,
  catalogId?: string | null,
): string {
  if (kind === "chatgpt") return "openai";
  if (kind === "claude-account") return "anthropic";
  const set = catalogProviderOf(kind, catalogId)?.id ?? kind;
  const regional = endpoint ? `${set}/${endpoint}` : undefined;
  return regional && MODELS.has(regional) ? regional : set;
}

export const MAX_MENU_MODELS = 100;

/** One order for Settings' Default model menu and the key form's list, so the two agree (#890). */
export const withRolesFirst = (
  roles: Partial<Record<Role, string>> | undefined,
  inMenu: readonly string[],
): string[] => [
  ...new Set([
    ...[roles?.answers, roles?.strongest, roles?.quickTasks].filter(
      (id): id is string => id !== undefined && inMenu.includes(id),
    ),
    ...inMenu,
  ]),
];

/** Before a key is saved its provider can't be asked for its list, so the catalog stands in for it. */
export function catalogMenu(catalogId: string, endpoint?: string | null): string[] {
  const provider = catalogProvider(catalogId);
  if (!provider) return [];
  const providerId = modelsProviderOfKind(kindOfCatalogId(catalogId), endpoint, catalogId);
  const models = MODELS.get(providerId);
  const ids = new Set(models?.keys());
  const current = models
    ? [...ids]
        .filter((id) => {
          const base = withoutDateSuffix(id);
          // A provider lists a model or its dated snapshot, never both.
          return !(base !== id && ids.has(base)) && !otherModel(providerId, id, ids);
        })
        .sort((a, b) => a.localeCompare(b))
    : [provider.roles.answers, provider.roles.strongest, provider.roles.quickTasks].filter(
        (id): id is string => id !== undefined,
      );
  return withRolesFirst(provider.roles, current).slice(0, MAX_MENU_MODELS);
}
