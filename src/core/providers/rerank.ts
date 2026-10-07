/**
 * Rerank (ADR-0005): with a Cohere or Voyage key, document search reranks its
 * hybrid candidates with the provider's reranking model before grouping them
 * (see ../documents/searchTool), through the AI SDK's `rerank` and the
 * official @ai-sdk/cohere and @ai-sdk/voyage providers. Without a key,
 * nothing changes.
 *
 * Its requests are the "rerank" data flow: the search query and the candidate
 * Passages, to the provider. Nothing is sent before the User accepts it; when
 * they decline, or a request fails, search keeps its own order. Local mode
 * pauses it. The settings are per device; the key is in the keychain, never
 * in SQLite.
 */
import { createCohere } from "@ai-sdk/cohere";
import { createVoyage } from "@ai-sdk/voyage";
import { type RerankingModel, rerank } from "ai";
import {
  type ConnectionTestResult,
  DEFAULT_RERANK_MODELS,
  type ExternalService,
  type RerankProviderKind,
  type RerankSettings,
  rerankProviderKinds,
} from "../api";
import type { Consent } from "../consent";
import type { Reranker, SearchCandidate } from "../documents/searchTool";
import { InvalidInputError, isRecord } from "../errors";
import type { Secrets } from "../secrets";
import type { SettingsStore } from "../settings";
import { classifyProviderError } from "./providerErrors";

/** A model object, never a model id string (which the AI SDK would send to its gateway). */
export type ApiRerankingModel = Exclude<RerankingModel, string>;

export interface RerankingModelSpec {
  kind: RerankProviderKind;
  apiKey: string;
  modelId: string;
}

export type RerankingModelFactory = (spec: RerankingModelSpec) => ApiRerankingModel;

/** The AI SDK provider for each kind. */
export const createAiSdkRerankingModel: RerankingModelFactory = ({ kind, apiKey, modelId }) =>
  kind === "cohere"
    ? createCohere({ apiKey }).reranking(modelId)
    : createVoyage({ apiKey }).reranking(modelId);

/** Where each provider's requests go. */
export const RERANK_SERVICES: Readonly<Record<RerankProviderKind, ExternalService>> = {
  cohere: { id: "https://api.cohere.com", name: "Cohere" },
  voyage: { id: "https://api.voyageai.com", name: "Voyage AI" },
};

/** The data the "rerank" flow sends. */
export const RERANK_FLOW_SENDS = ["queries", "passages"] as const;

const KEY_NAME = "rerank:api-key";
/** A per-device value, outside `DeviceSettings`: only these methods change it. */
const SETTINGS_VALUE = "rerank";

const RERANK_TIMEOUT_MS = 20_000;
const TEST_TIMEOUT_MS = 30_000;

/** What a connection test reranks: nothing of the User's. */
const TEST_QUERY = "Which city is the capital of France?";
const TEST_DOCUMENTS = ["Bananas are rich in potassium.", "Paris is the capital of France."];

/** What is stored: a null model means the provider's default. */
interface StoredRerank {
  kind: RerankProviderKind;
  modelId: string | null;
}

const isRerankKind = (value: unknown): value is RerankProviderKind =>
  rerankProviderKinds.some((kind) => kind === value);

function parseStored(value: unknown): StoredRerank | null {
  if (!isRecord(value) || !isRerankKind(value.kind)) return null;
  return { kind: value.kind, modelId: typeof value.modelId === "string" ? value.modelId : null };
}

/** undefined: keep the stored key. Blank text counts as "keep". */
function parseApiKey(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string") throw new InvalidInputError("The API key must be text.");
  return value.trim() || undefined;
}

function parseModel(value: unknown): string | null {
  if (value === null || (typeof value === "string" && value.trim() === "")) return null;
  if (typeof value !== "string" || value.length > 200) {
    throw new InvalidInputError("Enter a reranking model name.");
  }
  return value.trim();
}

function parseKind(value: unknown): RerankProviderKind {
  if (!isRerankKind(value)) {
    throw new InvalidInputError(`Unknown reranking provider "${String(value)}".`);
  }
  return value;
}

/** What the reranking model reads for a candidate: its Document's name, then the Passage. */
const candidateText = (candidate: SearchCandidate) =>
  `${candidate.documentName}\n${candidate.text}`;

export type Rerank = ReturnType<typeof createRerank>;

export function createRerank(options: {
  settings: SettingsStore;
  secrets: Secrets;
  consent: Consent;
  createModel: RerankingModelFactory;
  /** Local mode is on: nothing is reranked. */
  localOnly(): boolean;
  /** Aborts requests when the core closes. */
  signal: AbortSignal;
  reportError?: (error: unknown) => void;
}) {
  const { settings, secrets, consent, createModel, signal } = options;
  const reportError = options.reportError ?? ((error) => console.error(error));

  const stored = () => parseStored(settings.readDeviceValue(SETTINGS_VALUE));
  const modelOf = (rerankSettings: StoredRerank) =>
    rerankSettings.modelId ?? DEFAULT_RERANK_MODELS[rerankSettings.kind];

  const refuseInLocalMode = () => {
    if (options.localOnly()) {
      throw new InvalidInputError(
        "Local mode is on: search results aren't sent to a reranking service. Turn it off in Settings first.",
      );
    }
  };

  consent.registry.register({
    id: "rerank",
    sends: RERANK_FLOW_SENDS,
    async services() {
      const current = stored();
      return current ? [RERANK_SERVICES[current.kind]] : [];
    },
  });

  const status = async (): Promise<RerankSettings> => {
    const current = stored();
    return {
      enabled: current !== null,
      kind: current?.kind ?? null,
      modelId: current ? modelOf(current) : null,
      hasApiKey: current !== null && (await secrets.tryGet(KEY_NAME)) !== null,
      service: current ? RERANK_SERVICES[current.kind] : null,
      paused: current !== null && options.localOnly(),
    };
  };

  /**
   * The document-search Tool's reranker: the candidates in the reranking
   * model's order, scored by it. Unchanged, with nothing sent, while rerank
   * isn't set up or is paused, and when the User declines or the request fails.
   */
  const reranker: Reranker = async (query, candidates, abortSignal) => {
    const current = stored();
    if (!current || options.localOnly() || candidates.length < 2) return [...candidates];
    const apiKey = await secrets.tryGet(KEY_NAME);
    if (!apiKey) return [...candidates];
    try {
      await consent.ensure("rerank", RERANK_SERVICES[current.kind]);
      const { ranking } = await rerank({
        model: createModel({ kind: current.kind, apiKey, modelId: modelOf(current) }),
        documents: candidates.map(candidateText),
        query,
        topN: candidates.length,
        maxRetries: 1,
        abortSignal: AbortSignal.any([
          signal,
          AbortSignal.timeout(RERANK_TIMEOUT_MS),
          ...(abortSignal ? [abortSignal] : []),
        ]),
      });
      const ranked = ranking.flatMap(({ originalIndex, score }) => {
        const candidate = candidates[originalIndex];
        return candidate ? [{ ...candidate, score }] : [];
      });
      // Any the model left out follow, in search's order, below its lowest score.
      const seen = new Set(ranking.map((each) => each.originalIndex));
      const lowest = Math.min(0, ...ranked.map((candidate) => candidate.score));
      const rest = candidates
        .filter((_, index) => !seen.has(index))
        .map((candidate, index) => ({ ...candidate, score: lowest - (index + 1) * 1e-6 }));
      return [...ranked, ...rest];
    } catch (error) {
      // The Answer was stopped: stop too.
      if (abortSignal?.aborted) throw error;
      if (!signal.aborted) reportError(error);
      return [...candidates];
    }
  };

  return {
    status,
    reranker,

    /**
     * Sets rerank up, or changes it, once the User has accepted the rerank
     * flow to its service. The key is stored first: if the keychain refuses
     * it, nothing changes.
     */
    async save(input: unknown): Promise<RerankSettings> {
      if (!isRecord(input)) throw new InvalidInputError("saveRerankSettings expects an object.");
      for (const key of Object.keys(input)) {
        if (!["kind", "apiKey", "modelId"].includes(key)) {
          throw new InvalidInputError(`Rerank has no setting "${key}".`);
        }
      }
      const kind = parseKind(input.kind);
      const apiKey = parseApiKey(input.apiKey);
      const saved = stored();
      const sameKind = saved?.kind === kind;
      const modelId =
        input.modelId === undefined ? (sameKind ? saved.modelId : null) : parseModel(input.modelId);
      refuseInLocalMode();
      if (apiKey === undefined && (!sameKind || (await secrets.tryGet(KEY_NAME)) === null)) {
        throw new InvalidInputError(`Enter your ${RERANK_SERVICES[kind].name} API key.`);
      }
      // Nothing changes unless the User accepts sending search results there.
      await consent.ensure("rerank", RERANK_SERVICES[kind]);
      if (apiKey !== undefined) await secrets.set(KEY_NAME, apiKey);
      settings.writeDeviceValue(SETTINGS_VALUE, { kind, modelId } satisfies StoredRerank);
      return status();
    },

    /** Forgets rerank on this device: its settings, then its key. */
    async remove(): Promise<RerankSettings> {
      settings.writeDeviceValue(SETTINGS_VALUE, null);
      await secrets.delete(KEY_NAME);
      return status();
    },

    /**
     * Reranks two fixed texts against a fixed query with the given settings,
     * or the saved ones. The rerank flow must be accepted first.
     */
    async test(input: unknown): Promise<ConnectionTestResult> {
      if (input !== undefined && !isRecord(input)) {
        throw new InvalidInputError("testRerankConnection expects an object.");
      }
      const saved = stored();
      const kind = input?.kind === undefined ? saved?.kind : parseKind(input.kind);
      if (!kind) throw new InvalidInputError("Choose Cohere or Voyage AI.");
      const sameKind = saved?.kind === kind;
      const modelId =
        input?.modelId === undefined
          ? sameKind
            ? saved.modelId
            : null
          : parseModel(input.modelId);
      const apiKey =
        parseApiKey(input?.apiKey) ?? (sameKind ? await secrets.tryGet(KEY_NAME) : null);
      if (!apiKey) throw new InvalidInputError(`Enter your ${RERANK_SERVICES[kind].name} API key.`);
      refuseInLocalMode();
      try {
        await consent.ensure("rerank", RERANK_SERVICES[kind]);
        const { ranking } = await rerank({
          model: createModel({ kind, apiKey, modelId: modelId ?? DEFAULT_RERANK_MODELS[kind] }),
          documents: TEST_DOCUMENTS,
          query: TEST_QUERY,
          topN: TEST_DOCUMENTS.length,
          maxRetries: 0,
          abortSignal: AbortSignal.any([signal, AbortSignal.timeout(TEST_TIMEOUT_MS)]),
        });
        if (ranking.length === 0) {
          return {
            ok: false,
            error: { kind: "provider", message: "The reranking model returned no ranking." },
          };
        }
        return { ok: true };
      } catch (error) {
        return { ok: false, error: classifyProviderError(error) };
      }
    },
  };
}
