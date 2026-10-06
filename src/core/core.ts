import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { CoreAdapters } from "./adapters";
import type { ChatModelChoice, CoreApi, CoreEventSource, Unsubscribe } from "./api";
import { createConsent, type DataFlowRegistry } from "./consent";
import { createDocuments, parseListOptions } from "./documents";
import { InvalidInputError, isRecord } from "./errors";
import { type AnyEventListener, createEventHub } from "./events";
import { createFolders, parseFolderId } from "./folders";
import { createMindContent } from "./mindContent";
import { createMinds, parseMindId } from "./minds";
import { createChat, type PreparedChatModel } from "./providers/chat";
import { ollamaBaseUrl } from "./providers/kinds";
import { createAiSdkChatModel } from "./providers/models";
import {
  detectOllama,
  hasOllamaModel,
  pullOllamaModel,
  RECOMMENDED_OLLAMA_MODEL,
} from "./providers/ollama";
import { createSecrets } from "./secrets";
import { createSettings, isChatModelChoice } from "./settings";
import { migrate, openDatabase } from "./storage";

export const DATABASE_FILE = "incarnamind.db";

/** The core as its host sees it: the public interface (methods and events) plus host-only hooks. */
export interface Core extends CoreApi, CoreEventSource {
  /** Every event the core emits, for the host to forward to the UI. */
  onAnyEvent(listener: AnyEventListener): Unsubscribe;
  /** Every external data flow. Core modules register theirs here; consent covers each one. */
  readonly dataFlows: DataFlowRegistry;
  /**
   * For Answers (#29): the default chat model (or the one given), once Questions
   * can be asked and the User has accepted its data flow. Throws otherwise.
   */
  prepareChatModel(choice?: ChatModelChoice): Promise<PreparedChatModel>;
  /** Closes the database and drops all listeners. The core can't be used afterwards. Safe to call twice. */
  close(): void;
}

/**
 * Opens (or creates) the data folder and its database, brings the schema up to
 * date, and returns the core. The host passes in every platform capability.
 */
export function createCore(adapters: CoreAdapters): Core {
  const { dataDir } = adapters.paths;
  mkdirSync(dataDir, { recursive: true });

  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    migrate(db);
  } catch (error) {
    db.close();
    throw error;
  }

  const now = () => (adapters.now?.() ?? new Date()).toISOString();
  const events = createEventHub();
  const minds = createMinds(db, now);
  const mindsChanged = () => events.emit("minds.changed", minds.list());
  const content = createMindContent(db, now, (mindId, update) => {
    events.emit("mind.update", { mindId, update });
    if (minds.markEdited(mindId)) mindsChanged();
  });
  const settings = createSettings(db, now, adapters.systemLanguages);
  const folders = createFolders(db, now);
  const foldersChanged = () => events.emit("folders.changed", folders.list());
  let documents: ReturnType<typeof createDocuments>;
  try {
    documents = createDocuments({
      db,
      dataDir,
      now,
      emitStatus: (document) => events.emit("document.status", document),
    });
  } catch (error) {
    db.close();
    throw error;
  }
  const secrets = createSecrets(adapters.keychain, settings);
  const consent = createConsent(db, events, now);
  const chat = createChat({
    db,
    now,
    settings,
    secrets,
    consent,
    createModel: adapters.createChatModel ?? createAiSdkChatModel,
  });
  /** Aborts work still running (model downloads) when the core closes. */
  const lifetime = new AbortController();

  const settingsChanged = () => events.emit("settings.changed", settings.get());
  const readinessChanged = async () => {
    try {
      const readiness = await chat.readiness();
      if (!lifetime.signal.aborted) events.emit("chatReadiness.changed", readiness);
    } catch (error) {
      // Reading the keychain is async, so the core may have closed meanwhile: nobody is listening.
      if (!lifetime.signal.aborted) throw error;
    }
  };

  const ollamaUrl = (input: unknown) => {
    if (input !== undefined && !isRecord(input)) throw new InvalidInputError("Expected an object.");
    return ollamaBaseUrl(input?.baseUrl);
  };

  // Async on purpose: the renderer reaches these over IPC, and a future hosted core may be remote.
  return {
    createMind: async (input) => {
      const mind = minds.create(input);
      mindsChanged();
      return mind;
    },
    listMinds: async () => minds.list(),
    renameMind: async (mindId, title) => {
      const mind = minds.rename(mindId, title);
      mindsChanged();
      return mind;
    },
    deleteMind: async (mindId) => {
      const at = now();
      db.transaction(() => {
        const mind = minds.delete(mindId, at);
        content.remove(mind.id, at);
      });
      mindsChanged();
    },
    openMind: async (mindId) => {
      const mind = minds.get(mindId);
      return { mind, state: content.state(mind.id) };
    },
    applyMindUpdate: async (mindId, update) => {
      content.apply(minds.get(mindId).id, update);
    },
    closeMind: async (mindId) => {
      content.close(parseMindId(mindId));
    },
    getSettings: async () => settings.get(),
    updateSettings: async (patch) => {
      // The default chat model must be on a saved provider.
      const chatModel = isRecord(patch) && isRecord(patch.user) ? patch.user.chatModel : undefined;
      if (isChatModelChoice(chatModel) && !chat.exists(chatModel.providerId)) {
        throw new InvalidInputError("That chat provider doesn't exist.");
      }
      const updated = settings.update(patch);
      events.emit("settings.changed", updated);
      if (chatModel !== undefined) await readinessChanged();
      return updated;
    },
    addDocuments: (paths) => documents.add(paths),
    listDocuments: async (options) => {
      const { folderId, includeSubfolders } = parseListOptions(options);
      if (folderId === undefined) return documents.list();
      const folder = folders.get(folderId);
      return documents.list(includeSubfolders ? folders.subtree(folder.id) : [folder.id]);
    },
    renameDocument: async (id, name) => documents.rename(id, name),
    deleteDocument: (id) => documents.delete(id),
    searchPassages: async (query, limit) => documents.search(query, limit),

    listChatProviders: () => chat.list(),
    saveChatProvider: async (input) => {
      const provider = await chat.save(input);
      settingsChanged();
      await readinessChanged();
      return provider;
    },
    deleteChatProvider: async (id) => {
      await chat.delete(id);
      settingsChanged();
      await readinessChanged();
    },
    testChatConnection: (input) => chat.test(input),
    getChatReadiness: () => chat.readiness(),

    getSecretStorage: async () => secrets.status(),
    acceptPlainTextSecretStorage: async () => secrets.acceptPlainText(),

    detectOllama: (input) => detectOllama(ollamaUrl(input)),
    selectOllama: async (input) => {
      const baseUrl = ollamaUrl(input);
      const model = input?.model ?? RECOMMENDED_OLLAMA_MODEL;
      if (typeof model !== "string" || model.trim() === "") {
        throw new InvalidInputError("Enter a model name.");
      }
      const status = await detectOllama(baseUrl);
      if (!status.running) throw new Error(`Ollama isn't running at ${baseUrl}.`);
      if (!hasOllamaModel(status.models, model)) {
        await pullOllamaModel(
          baseUrl,
          model,
          (progress) => events.emit("ollama.pullProgress", progress),
          lifetime.signal,
        );
      }
      const provider = await chat.save({ kind: "ollama", baseUrl, modelId: model });
      settingsChanged();
      await readinessChanged();
      return provider;
    },

    listDataFlows: () => consent.list(),
    listConsentRequests: async () => consent.requests(),
    respondToConsent: async (requestId, accept) => {
      consent.respond(requestId, accept);
      await readinessChanged();
    },
    revokeConsent: async (flowId, serviceId) => {
      consent.revoke(flowId, serviceId);
      await readinessChanged();
    },

    moveDocument: async (documentId, folderInput) => {
      const folderId = folderInput === null ? null : parseFolderId(folderInput);
      const { document, moved } = db.transaction(() => {
        if (folderId !== null) folders.get(folderId);
        return documents.move(documentId, folderId);
      });
      if (moved) events.emit("documents.moved", [document]);
      return document;
    },
    createFolder: async (input) => {
      const folder = folders.create(input);
      foldersChanged();
      return folder;
    },
    listFolders: async () => folders.list(),
    renameFolder: async (folderId, name) => {
      const folder = folders.rename(folderId, name);
      foldersChanged();
      return folder;
    },
    moveFolder: async (folderId, parentId) => {
      const folder = folders.move(folderId, parentId);
      foldersChanged();
      return folder;
    },
    deleteFolder: async (folderId) => {
      const at = now();
      const unfiled = db.transaction(() => documents.unfile(folders.delete(folderId, at), at));
      // Folders first, so a listener filtering by a deleted Folder hears it's gone before it refreshes.
      foldersChanged();
      if (unfiled.length > 0) events.emit("documents.moved", unfiled);
    },
    on: (event, listener) => events.on(event, listener),
    onAnyEvent: (listener) => events.onAny(listener),
    dataFlows: consent.registry,
    prepareChatModel: (choice) => chat.prepareModel(choice),
    close: () => {
      lifetime.abort();
      consent.close();
      documents.close();
      events.clear();
      content.closeAll();
      db.close();
    },
  };
}
