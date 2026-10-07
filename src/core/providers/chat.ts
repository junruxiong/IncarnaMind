/**
 * Chat providers: their settings (in SQLite, without secrets), their API keys
 * (through `Secrets`, never in SQLite), whether Questions can be asked, the
 * connection test, and the model seam Answers use (#29).
 */
import { randomUUID } from "node:crypto";
import { generateText } from "ai";
import type {
  ChatModelChoice,
  ChatProvider,
  ChatProviderKind,
  ChatReadiness,
  ConnectionTestResult,
  ExternalService,
} from "../api";
import type { Consent } from "../consent";
import { ChatNotReadyError, InvalidInputError, isRecord } from "../errors";
import type { Secrets } from "../secrets";
import type { SettingsStore } from "../settings";
import type { Database } from "../storage";
import { isChatGptPlanModel } from "./chatgpt/codexEndpoint";
import type { ChatGptPlan } from "./chatgpt/plan";
import { acceptsApiKey, baseUrlFor, isChatProviderKind, requiresApiKey, serviceFor } from "./kinds";
import type { ChatLanguageModel, ChatModelFactory, ChatModelSpec } from "./models";
import { classifyProviderError } from "./providerErrors";

/**
 * The chat flow: what a Question sends to a cloud chat provider. Connectors
 * (#37, #39) add "tool-results", which makes the core ask again.
 */
export const CHAT_FLOW_SENDS = ["blocks", "passages"] as const;

const TEST_TIMEOUT_MS = 30_000;

interface ProviderRow {
  id: string;
  kind: string;
  base_url: string | null;
}

interface ParsedServer {
  kind: ChatProviderKind;
  baseUrl: string | null;
}

/** A model ready to use, for Answers. */
export interface PreparedChatModel {
  model: ChatLanguageModel;
  provider: ChatProvider;
  modelId: string;
}

const keyName = (providerId: string) => `chat-provider:${providerId}:api-key`;

function parseServer(input: Record<string, unknown>): ParsedServer {
  if (!isChatProviderKind(input.kind)) {
    throw new InvalidInputError(`Unknown chat provider "${String(input.kind)}".`);
  }
  return { kind: input.kind, baseUrl: baseUrlFor(input.kind, input.baseUrl) };
}

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
}) {
  const { db, now, settings, secrets, consent, createModel, chatGpt } = options;

  /** A ChatGPT plan provider counts only while the experimental switch is on. */
  const usable = (row: ProviderRow | undefined) =>
    row && (row.kind !== "chatgpt" || chatGpt.enabled()) ? row : undefined;

  const rowById = (id: string) =>
    usable(
      db.get<ProviderRow>(
        "SELECT id, kind, base_url FROM chat_providers WHERE id = ? AND deleted_at IS NULL",
        [id],
      ),
    );

  const rowByServer = ({ kind, baseUrl }: ParsedServer) =>
    usable(
      db.get<ProviderRow>(
        `SELECT id, kind, base_url FROM chat_providers
         WHERE kind = ? AND coalesce(base_url, '') = ? AND deleted_at IS NULL`,
        [kind, baseUrl ?? ""],
      ),
    );

  /** Rows of a kind this version knows; rows from a newer version are left alone. */
  const knownRows = () =>
    db
      .all<ProviderRow>(
        `SELECT id, kind, base_url FROM chat_providers WHERE deleted_at IS NULL
         ORDER BY created_at, rowid`,
      )
      .filter((row) => isChatProviderKind(row.kind) && usable(row));

  /** What the model factory needs; the ChatGPT plan signs with the sign-in instead of a key. */
  const specFor = (
    kind: ChatProviderKind,
    baseUrl: string | null,
    apiKey: string | null,
    modelId: string,
  ): ChatModelSpec =>
    kind === "chatgpt"
      ? {
          kind,
          baseUrl: chatGpt.codexBaseUrl,
          apiKey: null,
          modelId,
          credentials: chatGpt.credentials,
        }
      : { kind, baseUrl, apiKey, modelId };

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
      baseUrl: row.base_url,
      hasApiKey,
      service: serviceFor(kind, row.base_url),
    };
  };

  consent.registry.register({
    id: "chat",
    sends: CHAT_FLOW_SENDS,
    async services() {
      const services: ExternalService[] = [];
      for (const row of knownRows()) {
        const service = serviceFor(row.kind as ChatProviderKind, row.base_url);
        if (service) services.push(service);
      }
      return services;
    },
  });

  const readinessFor = async (choice: ChatModelChoice | null): Promise<ChatReadiness> => {
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
    const decision = consent.status("chat", provider.service);
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
          db.run(
            `INSERT INTO chat_providers (id, kind, base_url, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?)`,
            [id, server.kind, server.baseUrl, at, at],
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

    readiness: () => readinessFor(settings.get().user.chatModel),

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
        const model = createModel(specFor(server.kind, server.baseUrl, apiKey, modelId));
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
     * picker, after checking readiness and getting consent for its service.
     * Throws `ChatNotReadyError` or `ConsentDeclinedError`; nothing is sent then.
     */
    async prepareModel(choice?: ChatModelChoice): Promise<PreparedChatModel> {
      const readiness = await readinessFor(choice ?? settings.get().user.chatModel);
      if (!readiness.ready) throw new ChatNotReadyError(readiness);
      const { provider, modelId } = readiness;
      if (provider.service) await consent.ensure("chat", provider.service);
      const apiKey = acceptsApiKey(provider.kind) ? await secrets.get(keyName(provider.id)) : null;
      const model = createModel(specFor(provider.kind, provider.baseUrl, apiKey, modelId));
      return { model, provider, modelId };
    },
  };
}
