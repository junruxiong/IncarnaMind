/**
 * Chat providers: their settings (in SQLite, without secrets), their API keys
 * (through `Secrets`, never in SQLite), whether Questions can be asked, the
 * connection test, and the model seam Answers use (#29).
 */
import { randomUUID } from "node:crypto";
import { generateText } from "ai";
import type {
  ChatModelChoice,
  ChatModelGroup,
  ChatProvider,
  ChatProviderKind,
  ChatReadiness,
  CitationSupport,
  ConnectionTestResult,
  DataFlowId,
  ExternalService,
} from "../api";
import type { Consent } from "../consent";
import { ChatNotReadyError, InvalidInputError, isRecord } from "../errors";
import type { Secrets } from "../secrets";
import type { SettingsStore } from "../settings";
import type { Database } from "../storage";
import {
  cloudWindow,
  type ModelCapabilities,
  type ModelFactsLayer,
  readsImages,
  resolveCapabilities,
  startingSupport,
} from "./capabilities";
import { catalogFacts, catalogProviderOfKind, modelsProviderOfKind } from "./catalog";
import { isChatGptPlanModel } from "./chatgpt/codexEndpoint";
import type { ChatGptPlan } from "./chatgpt/plan";
import {
  acceptsApiKey,
  baseUrlFor,
  endpointUrl,
  isChatProviderKind,
  requiresApiKey,
  serviceFor,
} from "./kinds";
import { type ListedModel, listProviderModels } from "./modelLists";
import type { ChatLanguageModel, ChatModelFactory, ChatModelSpec, ContextWindow } from "./models";
import {
  DEFAULT_OLLAMA_SETTINGS,
  type OllamaModelProfile,
  type OllamaModels,
} from "./ollamaModels";
import { classifyProviderError, LocalModelError } from "./providerErrors";

/** How long a provider's model list is kept before it is asked again. */
const MODEL_LIST_TTL_MS = 5 * 60_000;

/**
 * The chat flow: what a Question sends to a cloud chat provider. While a
 * Connector is on, the core adds "tool-results" (what its Tools return),
 * which makes it ask again.
 */
export const CHAT_FLOW_SENDS = ["blocks", "passages"] as const;

const TEST_TIMEOUT_MS = 30_000;

interface ProviderRow {
  id: string;
  kind: string;
  base_url: string | null;
  /** The provider in the catalog (see ./catalog); null for one it doesn't have (the ChatGPT plan). */
  catalog_id: string | null;
  /** A hosted provider's endpoint (its region); null for a server the User gives. */
  endpoint: string | null;
}

const ROW_COLUMNS = "id, kind, base_url, catalog_id, endpoint";

interface ParsedServer {
  kind: ChatProviderKind;
  baseUrl: string | null;
  /** Its catalog provider's id, or null. */
  catalogId: string | null;
}

/** A model ready to use, for Answers. */
export interface PreparedChatModel {
  model: ChatLanguageModel;
  provider: ChatProvider;
  modelId: string;
  /** What the model can do, as known before the first request (see ./capabilities). */
  capabilities: ModelCapabilities;
  /**
   * The context window each request must fit: a local model's, which
   * IncarnaMind sets, or a cloud model's own when its context is known.
   */
  window?: ContextWindow;
  /** How the model gives Citations, known before the first request (see ./capabilities). */
  support?: CitationSupport;
  /**
   * A local model's quotes the Citation check doesn't find are asked for once
   * more (see ../answers/quoteRetry); never a cloud model's.
   */
  retryQuotes?: boolean;
  /** Which build of the model this is (Ollama's digest), when known: what was learnt about it holds for this build. */
  revision?: string;
  /**
   * For a local model: whether it is loaded and ready now, or a request would
   * first wait for it to load; null when that can't be told.
   */
  loaded?: () => Promise<boolean | null>;
}

/** A model in Ollama that can't answer Questions, such as an embedding model, fails before anything is sent. */
function checkCanChat(profile: OllamaModelProfile | null, modelId: string): void {
  if (profile && !profile.chat) {
    throw new LocalModelError(
      "model",
      `${modelId} can't answer Questions: Ollama lists it as ${profile.capabilities?.join(", ") || "not a chat model"}. Pick a chat model.`,
    );
  }
}

const keyName = (providerId: string) => `chat-provider:${providerId}:api-key`;

function parseServer(input: Record<string, unknown>): ParsedServer {
  if (!isChatProviderKind(input.kind)) {
    throw new InvalidInputError(`Unknown chat provider "${String(input.kind)}".`);
  }
  return {
    kind: input.kind,
    baseUrl: baseUrlFor(input.kind, input.baseUrl),
    catalogId: catalogProviderOfKind(input.kind)?.id ?? null,
  };
}

/** What Ollama says of a model, as the server's word on it: whether it calls Tools, and how it cites. */
const ollamaFacts = (profile: OllamaModelProfile): ModelFactsLayer => ({
  ...(profile.capabilities && { tools: profile.capabilities.includes("tools") }),
  ...(profile.support && { citing: profile.support }),
});

function parseModelId(value: unknown): string {
  if (typeof value !== "string" || value.trim() === "" || value.length > 500) {
    throw new InvalidInputError("Enter a model name.");
  }
  return value.trim();
}

/** undefined: keep the stored key. null: remove it. Blank text counts as "keep". */
function parseApiKey(value: unknown, kind: ChatProviderKind): string | null | undefined {
  if (value === undefined || value === null) return value;
  if (typeof value !== "string") throw new InvalidInputError("The API key must be text.");
  const key = value.trim();
  if (key === "") return undefined;
  if (!acceptsApiKey(kind)) throw new InvalidInputError("This provider doesn't take an API key.");
  return key;
}

const displayName = (kind: ChatProviderKind, baseUrl: string | null) =>
  serviceFor(kind, baseUrl)?.name ?? baseUrl ?? kind;

/** What chat providers need from the experimental ChatGPT plan provider. */
export type ChatGptPlanAccess = Pick<
  ChatGptPlan,
  "enabled" | "hasSignIn" | "credentials" | "codexBaseUrl"
>;

export function createChat(options: {
  db: Database;
  now: () => string;
  settings: SettingsStore;
  secrets: Secrets;
  consent: Consent;
  createModel: ChatModelFactory;
  chatGpt: ChatGptPlanAccess;
  /** Looks up models in Ollama, for the settings their requests carry and how they cite. */
  ollamaModels: OllamaModels;
}) {
  const { db, now, settings, secrets, consent, createModel, chatGpt, ollamaModels } = options;

  /** What Ollama says about a model, for the "ollama" kind; null for other kinds, or when it can't say. */
  const profileFor = (kind: ChatProviderKind, baseUrl: string | null, modelId: string) =>
    kind === "ollama" && baseUrl ? ollamaModels.describe(baseUrl, modelId) : Promise.resolve(null);

  /** A ChatGPT plan provider counts only while the experimental switch is on. */
  const usable = (row: ProviderRow | undefined) =>
    row && (row.kind !== "chatgpt" || chatGpt.enabled()) ? row : undefined;
  /** Model lists by provider and server, with when they were fetched. */
  const modelLists = new Map<string, { at: number; models: ListedModel[] }>();
  const listKey = (row: ProviderRow) => `${row.id}\n${row.base_url ?? ""}\n${row.endpoint ?? ""}`;

  const rowById = (id: string) =>
    usable(
      db.get<ProviderRow>(
        `SELECT ${ROW_COLUMNS} FROM chat_providers WHERE id = ? AND deleted_at IS NULL`,
        [id],
      ),
    );

  const rowByServer = ({ kind, baseUrl, catalogId }: ParsedServer) =>
    usable(
      db.get<ProviderRow>(
        `SELECT ${ROW_COLUMNS} FROM chat_providers
         WHERE coalesce(catalog_id, kind) = ? AND coalesce(base_url, '') = ? AND deleted_at IS NULL`,
        [catalogId ?? kind, baseUrl ?? ""],
      ),
    );

  /** Rows of a kind this version knows; rows from a newer version are left alone. */
  const knownRows = () =>
    db
      .all<ProviderRow>(
        `SELECT ${ROW_COLUMNS} FROM chat_providers WHERE deleted_at IS NULL
         ORDER BY created_at, rowid`,
      )
      .filter((row) => isChatProviderKind(row.kind) && usable(row));

  /** Where a row's requests go. */
  const rowService = (row: ProviderRow) =>
    serviceFor(row.kind as ChatProviderKind, row.base_url, row.endpoint);

  /**
   * What is known of a model on a provider, by precedence (see ./capabilities):
   * the server's word (`server`, else what its model list said), then the
   * catalog's. Nothing the User sets on a model is stored yet.
   */
  const capabilitiesOf = (
    row: ProviderRow,
    modelId: string,
    server?: ModelFactsLayer,
  ): ModelCapabilities =>
    resolveCapabilities({
      server: server ?? modelLists.get(listKey(row))?.models.find((m) => m.id === modelId)?.facts,
      catalog: catalogFacts(modelsProviderOfKind(row.kind as ChatProviderKind), modelId),
    });

  /**
   * What the model factory needs: a hosted provider's endpoint from the
   * catalog, else the server's URL. The ChatGPT plan signs with the sign-in
   * instead of a key, and a model in Ollama gets its fixed settings, the
   * defaults when Ollama couldn't say.
   */
  const specFor = (
    server: { kind: ChatProviderKind; baseUrl: string | null; endpoint?: string | null },
    apiKey: string | null,
    modelId: string,
    profile: OllamaModelProfile | null = null,
  ): ChatModelSpec => {
    const { kind } = server;
    if (kind === "chatgpt") {
      return {
        kind,
        baseUrl: chatGpt.codexBaseUrl,
        apiKey: null,
        modelId,
        credentials: chatGpt.credentials,
      };
    }
    const baseUrl = endpointUrl(kind, server.endpoint) ?? server.baseUrl;
    return kind === "ollama"
      ? { kind, baseUrl, apiKey, modelId, ollama: profile?.settings ?? DEFAULT_OLLAMA_SETTINGS }
      : { kind, baseUrl, apiKey, modelId };
  };

  /** The ChatGPT plan takes only the models its endpoint accepts, and only while it's turned on. */
  const checkChatGpt = (kind: ChatProviderKind, modelId: string) => {
    if (kind !== "chatgpt") return;
    if (!chatGpt.enabled()) {
      throw new InvalidInputError("Turn on the ChatGPT plan provider in Settings first.");
    }
    if (!isChatGptPlanModel(modelId)) {
      throw new InvalidInputError(`The ChatGPT plan doesn't offer the model "${modelId}".`);
    }
  };

  const toProvider = async (row: ProviderRow): Promise<ChatProvider> => {
    const kind = row.kind as ChatProviderKind;
    const hasApiKey = acceptsApiKey(kind) && (await secrets.tryGet(keyName(row.id))) !== null;
    return {
      id: row.id,
      kind,
      catalogId: row.catalog_id,
      endpoint: row.endpoint,
      baseUrl: row.base_url,
      hasApiKey,
      service: rowService(row),
    };
  };

  consent.registry.register({
    id: "chat",
    sends: CHAT_FLOW_SENDS,
    async services() {
      const services: ExternalService[] = [];
      for (const row of knownRows()) {
        const service = rowService(row);
        if (service) services.push(service);
      }
      return services;
    },
  });

  /** The saved provider of the default chat model, if there is one. */
  const defaultRow = () => {
    const choice = settings.get().user.chatModel;
    const row = choice && rowById(choice.providerId);
    return row && isChatProviderKind(row.kind) ? row : undefined;
  };

  /**
   * Whether `choice` can take requests on `flow` (Questions use "chat";
   * automatic tagging uses "tagging", with its own consent).
   */
  const readinessFor = async (
    choice: ChatModelChoice | null,
    flow: DataFlowId,
  ): Promise<ChatReadiness> => {
    const row = choice && rowById(choice.providerId);
    if (!choice || !row || !isChatProviderKind(row.kind)) {
      return { ready: false, reason: "no-provider" };
    }
    const provider = await toProvider(row);
    const { modelId } = choice;
    if (requiresApiKey(provider.kind) && !provider.hasApiKey) {
      return { ready: false, reason: "missing-api-key", provider, modelId };
    }
    if (provider.kind === "chatgpt" && !(await chatGpt.hasSignIn())) {
      return { ready: false, reason: "sign-in-required", provider, modelId };
    }
    if (!provider.service) return { ready: true, provider, modelId, consent: "not-required" };
    const decision = consent.status(flow, provider.service);
    if (decision === "declined") {
      return { ready: false, reason: "consent-declined", provider, modelId };
    }
    return {
      ready: true,
      provider,
      modelId,
      consent: decision === "accepted" ? "accepted" : "needed",
    };
  };

  return {
    async list(): Promise<ChatProvider[]> {
      return Promise.all(knownRows().map(toProvider));
    },

    /** Saves the provider and makes the model on it the default. The key is stored first: if the keychain refuses it, nothing changes. */
    async save(input: unknown): Promise<ChatProvider> {
      if (!isRecord(input)) throw new InvalidInputError("saveChatProvider expects an object.");
      const server = parseServer(input);
      const apiKey = parseApiKey(input.apiKey, server.kind);
      const modelId = parseModelId(input.modelId);
      checkChatGpt(server.kind, modelId);

      const existing = rowByServer(server);
      const id = existing?.id ?? randomUUID();
      const keyAfterSave =
        apiKey === undefined
          ? existing !== undefined && (await secrets.tryGet(keyName(id))) !== null
          : apiKey !== null;
      if (requiresApiKey(server.kind) && !keyAfterSave) {
        throw new InvalidInputError(
          `Enter an API key for ${displayName(server.kind, server.baseUrl)}.`,
        );
      }

      if (typeof apiKey === "string") await secrets.set(keyName(id), apiKey);
      const at = now();
      db.transaction(() => {
        if (existing) {
          db.run("UPDATE chat_providers SET updated_at = ? WHERE id = ?", [at, id]);
        } else {
          // A hosted provider starts on its first endpoint.
          const endpoint = catalogProviderOfKind(server.kind)?.endpoints[0]?.id ?? null;
          db.run(
            `INSERT INTO chat_providers (id, kind, base_url, catalog_id, endpoint, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [id, server.kind, server.baseUrl, server.catalogId, endpoint, at, at],
          );
        }
        settings.update({ user: { chatModel: { providerId: id, modelId } } });
      });
      if (apiKey === null) await secrets.delete(keyName(id));

      const row = rowById(id);
      if (!row) throw new Error("The chat provider wasn't saved.");
      return toProvider(row);
    },

    /** Soft-deletes the provider and deletes its key. Deleting one that doesn't exist does nothing. */
    async delete(id: unknown): Promise<void> {
      if (typeof id !== "string") throw new InvalidInputError("A provider id must be text.");
      if (!rowById(id)) return;
      const at = now();
      db.transaction(() => {
        db.run("UPDATE chat_providers SET deleted_at = ?, updated_at = ? WHERE id = ?", [
          at,
          at,
          id,
        ]);
        if (settings.get().user.chatModel?.providerId === id) {
          settings.update({ user: { chatModel: null } });
        }
      });
      await secrets.delete(keyName(id));
    },

    /** Removes every provider of a kind (e.g. when the ChatGPT plan is turned off), keys included. */
    async deleteKind(kind: ChatProviderKind): Promise<void> {
      const rows = db.all<{ id: string }>(
        "SELECT id FROM chat_providers WHERE kind = ? AND deleted_at IS NULL",
        [kind],
      );
      const at = now();
      db.transaction(() => {
        for (const { id } of rows) {
          db.run("UPDATE chat_providers SET deleted_at = ?, updated_at = ? WHERE id = ?", [
            at,
            at,
            id,
          ]);
          if (settings.get().user.chatModel?.providerId === id) {
            settings.update({ user: { chatModel: null } });
          }
        }
      });
      for (const { id } of rows) await secrets.delete(keyName(id));
    },

    exists: (id: string) => rowById(id) !== undefined,

    /**
     * Whether `choice`'s model reads images, as known now (see ./capabilities):
     * false when nothing says so, or its provider is gone.
     */
    readsImages(choice: ChatModelChoice): boolean {
      const row = rowById(choice.providerId);
      return (
        !!row && isChatProviderKind(row.kind) && readsImages(capabilitiesOf(row, choice.modelId))
      );
    },

    /**
     * Whether Questions can be asked with `choice`, or with the default model
     * when it is left out. With another `flow`, whether that flow can use it.
     */
    readiness: (choice?: ChatModelChoice, flow: DataFlowId = "chat") =>
      readinessFor(choice ?? settings.get().user.chatModel, flow),

    /** The service the default chat model sends to, or null when there is none or it runs on this computer. */
    defaultService(): ExternalService | null {
      const row = defaultRow();
      return row ? rowService(row) : null;
    },

    /**
     * A quick check, with nothing to wait for: false when the default chat
     * model certainly can't be used on `flow` (there is none, or the User
     * declined the flow to its service). `readiness` checks keys and sign-ins too.
     */
    mightBeReady(flow: DataFlowId): boolean {
      const row = defaultRow();
      if (!row) return false;
      const service = rowService(row);
      return !service || consent.status(flow, service) !== "declined";
    },

    /**
     * Each saved provider with the models it lists and its default model, for
     * a Question's model picker, and which of them nothing says what they can
     * do. Lists are kept for a few minutes.
     */
    async listModels(): Promise<ChatModelGroup[]> {
      const chatModel = settings.get().user.chatModel;
      return Promise.all(
        knownRows().map(async (row) => {
          const provider = await toProvider(row);
          // Like the connection test, a cloud service isn't contacted before the User allows the chat flow to it.
          const mayAsk =
            !provider.service || consent.status("chat", provider.service) === "accepted";
          const cacheKey = listKey(row);
          const cached = modelLists.get(cacheKey);
          let listed = cached && Date.now() - cached.at < MODEL_LIST_TTL_MS ? cached.models : null;
          if (!listed && !mayAsk) listed = [];
          if (!listed) {
            const apiKey = provider.hasApiKey ? await secrets.tryGet(keyName(row.id)) : null;
            listed = await listProviderModels({
              kind: provider.kind,
              baseUrl: row.base_url,
              apiKey,
              endpoint: row.endpoint,
            });
            if (listed.length > 0) modelLists.set(cacheKey, { at: Date.now(), models: listed });
          }
          const defaultModel = chatModel?.providerId === row.id ? [chatModel.modelId] : [];
          const models = [...new Set([...defaultModel, ...listed.map((each) => each.id)])];
          const unknown = models.filter((id) => !capabilitiesOf(row, id).known);
          return { provider, models, unknown };
        }),
      );
    },

    /**
     * Sends one tiny prompt with the given settings, which need not be saved.
     * Without a key in the input, the key stored for the same provider is used.
     * A cloud provider's chat flow must be accepted first: nothing is sent before.
     */
    async test(input: unknown): Promise<ConnectionTestResult> {
      if (!isRecord(input)) throw new InvalidInputError("testChatConnection expects an object.");
      const server = parseServer(input);
      const modelId = parseModelId(input.modelId);
      let apiKey = parseApiKey(input.apiKey, server.kind) ?? null;
      if (apiKey === null && acceptsApiKey(server.kind)) {
        const saved = rowByServer(server);
        if (saved) apiKey = await secrets.tryGet(keyName(saved.id));
      }
      if (requiresApiKey(server.kind) && !apiKey) {
        throw new InvalidInputError(
          `Enter an API key for ${displayName(server.kind, server.baseUrl)}.`,
        );
      }
      checkChatGpt(server.kind, modelId);
      if (server.kind === "chatgpt" && !(await chatGpt.hasSignIn())) {
        return {
          ok: false,
          error: { kind: "not-signed-in", message: "Sign in to ChatGPT first." },
        };
      }

      try {
        const service = serviceFor(server.kind, server.baseUrl);
        if (service) await consent.ensure("chat", service);
        // A model in Ollama is tested with the settings Answers use, so it stays loaded for them.
        const profile = await profileFor(server.kind, server.baseUrl, modelId);
        checkCanChat(profile, modelId);
        const model = createModel(specFor(server, apiKey, modelId, profile));
        await generateText({
          model,
          prompt: "Reply with the word OK.",
          maxOutputTokens: 64,
          maxRetries: 0,
          abortSignal: AbortSignal.timeout(TEST_TIMEOUT_MS),
        });
        return { ok: true };
      } catch (error) {
        return { ok: false, error: classifyProviderError(error) };
      }
    },

    /**
     * For Answers (#29): the default model, or `choice` from a Question's model
     * picker, after checking readiness and getting consent for its service on
     * `flow` (automatic tagging passes "tagging").
     * Throws `ChatNotReadyError` or `ConsentDeclinedError`; nothing is sent then.
     */
    async prepareModel(
      choice?: ChatModelChoice,
      flow: DataFlowId = "chat",
    ): Promise<PreparedChatModel> {
      const readiness = await readinessFor(choice ?? settings.get().user.chatModel, flow);
      if (!readiness.ready) throw new ChatNotReadyError(readiness);
      const { provider, modelId } = readiness;
      if (provider.service) await consent.ensure(flow, provider.service);
      const apiKey = acceptsApiKey(provider.kind) ? await secrets.get(keyName(provider.id)) : null;
      const profile = await profileFor(provider.kind, provider.baseUrl, modelId);
      checkCanChat(profile, modelId);
      const model = createModel(specFor(provider, apiKey, modelId, profile));
      const row: ProviderRow = {
        id: provider.id,
        kind: provider.kind,
        base_url: provider.baseUrl,
        catalog_id: provider.catalogId,
        endpoint: provider.endpoint,
      };
      const capabilities = capabilitiesOf(row, modelId, profile ? ollamaFacts(profile) : undefined);
      const support = startingSupport(capabilities);
      const prepared = { model, provider, modelId, capabilities, ...(support && { support }) };
      if (provider.kind !== "ollama") {
        // A cloud model's requests are kept within its own context window, when it is known.
        const window = cloudWindow(capabilities);
        return { ...prepared, ...(window && { window }) };
      }
      const baseUrl = provider.baseUrl;
      // Without Ollama's word on the model (it couldn't be asked, or doesn't have it), its
      // request fails anyway: nothing is sized to a window that may not be its own.
      if (!profile || !baseUrl) return prepared;
      const { numCtx, outputTokens } = profile.settings;
      return {
        ...prepared,
        window: { tokens: numCtx, outputTokens },
        retryQuotes: true,
        ...(profile.digest && { revision: profile.digest }),
        loaded: () => ollamaModels.loaded(baseUrl, modelId, numCtx),
      };
    },
  };
}
