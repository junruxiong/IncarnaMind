import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { translate } from "../shared/i18n";
import type { CoreAdapters } from "./adapters";
import { createAiSdkAnswerEngine, createAnswers } from "./answers";
import type {
  ChatModelChoice,
  CoreApi,
  CoreEventSource,
  EmbeddingSettings,
  Unsubscribe,
} from "./api";
import { createConnectors } from "./connectors";
import { createConsent, type DataFlowRegistry } from "./consent";
import { createDocuments, type DocumentFile, parseListOptions } from "./documents";
import { BUILT_IN_EMBEDDING_MODEL, createEmbeddingModel } from "./embedding";
import { createActiveEmbedding } from "./embedding/active";
import { InvalidInputError, isRecord } from "./errors";
import { type AnyEventListener, createEventHub } from "./events";
import { createExports } from "./exports";
import { createFolders, parseFolderId } from "./folders";
import { createMindContent } from "./mindContent";
import { createMinds, parseMindId } from "./minds";
import { CHAT_FLOW_SENDS, createChat, type PreparedChatModel } from "./providers/chat";
import { CHATGPT_PLAN_ENDPOINTS, createChatGptPlan } from "./providers/chatgpt/plan";
import { createAiSdkEmbeddingModel } from "./providers/embeddings";
import { ollamaBaseUrl } from "./providers/kinds";
import { createAiSdkChatModel } from "./providers/models";
import {
  detectOllama,
  hasOllamaModel,
  pullOllamaModel,
  RECOMMENDED_OLLAMA_MODEL,
} from "./providers/ollama";
import { createAiSdkRerankingModel, createRerank } from "./providers/rerank";
import { resolveSearchScope } from "./scope";
import { createSecrets } from "./secrets";
import { createSettings, isChatModelChoice } from "./settings";
import { createSkills } from "./skills";
import { migrate, openDatabase } from "./storage";
import { createTags } from "./tags";
import { chatClassifier } from "./tags/classify";
import { createJevTagging } from "./tags/jev";
import { createTagger, TAGGING_FLOW_SENDS } from "./tags/tagger";

export const DATABASE_FILE = "incarnamind.db";

/** The core as its host sees it: the public interface (methods and events) plus host-only hooks. */
export interface Core extends CoreApi, CoreEventSource {
  /**
   * Opens a Document's stored file for reading, for the host to serve to the UI
   * (the desktop app streams it over a custom protocol, so files never cross IPC).
   * Only live Documents: throws NotFoundError for an unknown or deleted Document,
   * or if its file is missing from the data folder.
   */
  openDocumentFile(documentId: string): Promise<DocumentFile>;
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
  const tags = createTags(db, now);
  // First run: the preset Tags, in the interface language of the time.
  tags.seedPresets(settings.get().language);
  const tagsChanged = () => events.emit("tags.changed", tags.list());
  /** Set once automatic tagging exists: it hears about every Document that becomes ready. */
  let documentReady = (_documentId: string) => {};
  const secrets = createSecrets(adapters.keychain, settings);
  const consent = createConsent(db, events, now);
  /** Aborts work still running (model downloads, a ChatGPT sign-in, provider requests) when the core closes. */
  const lifetime = new AbortController();
  const embeddingModel = createEmbeddingModel({
    definition: BUILT_IN_EMBEDDING_MODEL,
    source: adapters.embeddingModelSource,
    dataDir,
    embedder: adapters.embedder,
    emitStatus: (status) => events.emit("embeddingModel.status", status),
  });
  /** Reports the embedding settings again; set once Documents exist. */
  let embeddingChanged = () => {};
  /** A Document's status changed: during a rebuild, its progress may have too. */
  let rebuildMayHaveProgressed = () => {};
  // The embedding model search uses: the built-in one, or the provider the User chose.
  const embedding = createActiveEmbedding({
    builtIn: embeddingModel,
    settings,
    secrets,
    consent,
    createModel: adapters.createEmbeddingModel ?? createAiSdkEmbeddingModel,
    signal: lifetime.signal,
    onChange: () => embeddingChanged(),
  });
  let documents: ReturnType<typeof createDocuments>;
  try {
    documents = createDocuments({
      db,
      dataDir,
      now,
      model: embedding.model,
      emitStatus: (document) => {
        events.emit("document.status", document);
        rebuildMayHaveProgressed();
      },
      onReady: (documentId) => documentReady(documentId),
    });
  } catch (error) {
    lifetime.abort();
    embeddingModel.close();
    db.close();
    throw error;
  }
  let skills: ReturnType<typeof createSkills>;
  try {
    skills = createSkills({
      db,
      dataDir,
      now,
      reportError: (error) => console.error(error),
      builtInSkills: adapters.paths.builtInSkills,
    });
  } catch (error) {
    lifetime.abort();
    documents.close();
    embeddingModel.close();
    db.close();
    throw error;
  }
  const skillsChanged = () => events.emit("skills.changed", skills.list());

  /** The embedding settings, with the rebuild's progress; a rebuild that has finished ends here. */
  const embeddingSettings = async (): Promise<EmbeddingSettings> => {
    const reason = embedding.rebuildReason();
    let rebuild: EmbeddingSettings["rebuild"] = null;
    if (reason) {
      const progress = documents.embeddingProgress();
      if (progress.done >= progress.total) embedding.endRebuild();
      else rebuild = { reason, ...progress };
    }
    return {
      provider: await embedding.provider(),
      localOnly: embedding.localOnly(),
      rebuild,
      error: embedding.error(),
    };
  };
  embeddingChanged = () => {
    embeddingSettings().then(
      (current) => {
        if (!lifetime.signal.aborted) events.emit("embedding.changed", current);
      },
      (error: unknown) => {
        // Reading the keychain is async, so the core may have closed meanwhile.
        if (!lifetime.signal.aborted) console.error(error);
      },
    );
  };
  /** The rebuild's progress last reported, so each Document finishing is reported once. */
  let reportedProgress = "";
  rebuildMayHaveProgressed = () => {
    if (!embedding.rebuildReason()) return;
    const { total, done } = documents.embeddingProgress();
    const progress = `${done}/${total}`;
    if (progress === reportedProgress) return;
    reportedProgress = progress;
    embeddingChanged();
  };

  // Rerank, with a Cohere or Voyage key: paused in local mode.
  const rerank = createRerank({
    settings,
    secrets,
    consent,
    createModel: adapters.createRerankingModel ?? createAiSdkRerankingModel,
    localOnly: () => embedding.localOnly(),
    signal: lifetime.signal,
  });
  const rerankChanged = async () => {
    const status = await rerank.status();
    if (!lifetime.signal.aborted) events.emit("rerank.changed", status);
    return status;
  };
  /** Local mode on or off: embeddings may switch back to the built-in model, and rerank pauses. */
  const setLocalOnly = async (enabled: unknown) => {
    const wasLocal = embedding.localOnly();
    await embedding.setLocalOnly(enabled);
    if (wasLocal !== embedding.localOnly()) await rerankChanged();
    return embeddingSettings();
  };

  // Sign-in changes can happen in the middle of a request (a refresh that fails), so they report through events.
  let chatGptChanged = () => {};
  const chatGpt = createChatGptPlan({
    endpoints: { ...CHATGPT_PLAN_ENDPOINTS, ...adapters.chatGptPlan },
    secrets,
    settings,
    browser: adapters.browser,
    now: () => (adapters.now?.() ?? new Date()).getTime(),
    onChange: () => chatGptChanged(),
  });
  const chat = createChat({
    db,
    now,
    settings,
    secrets,
    consent,
    createModel: adapters.createChatModel ?? createAiSdkChatModel,
    chatGpt,
  });

  /** Set once readiness can be reported: Connectors turned on or off change what chat sends. */
  let connectorsToggled = () => {};
  const connectors = createConnectors({
    db,
    now,
    secrets,
    processes: adapters.processes,
    consent,
    onChange: (list) => events.emit("connectors.changed", list),
    onEnabledChange: () => connectorsToggled(),
    reportError: (error) => console.error(error),
  });
  /**
   * With a Connector on, Answers send the chat model what its Tools return,
   * so the chat flow sends "tool-results" too, and asks again for it.
   *
   * Skills don't: their Tools (`use_skill`, `read_skill_file`) return only
   * the text of Skills the User imported, which is sent like the system
   * prompt, not data fetched from another service. Skill scripts, whose
   * output could be anything, would need "tool-results" when they land.
   */
  const syncChatFlow = () => {
    const flow = consent.registry.get("chat");
    if (!flow) return;
    const sends = connectors.anyEnabled()
      ? [...CHAT_FLOW_SENDS, "tool-results" as const]
      : CHAT_FLOW_SENDS;
    consent.registry.register({ ...flow, sends });
  };
  syncChatFlow();

  const answers = createAnswers({
    content,
    events,
    engine: adapters.answerEngine ?? createAiSdkAnswerEngine(),
    requireMind: (mindId) => minds.get(mindId).id,
    mindExists: (mindId) => {
      try {
        minds.get(mindId);
        return true;
      } catch {
        return false;
      }
    },
    providerExists: (providerId) => chat.exists(providerId),
    readiness: (choice) => chat.readiness(choice),
    prepareModel: (choice) => chat.prepareModel(choice),
    documents: {
      searchableCount: (documentIds) => documents.searchableCount(documentIds ?? undefined),
      search: (query, documentIds, signal) =>
        documents.searchTool(query, {
          signal,
          rerank: adapters.reranker ?? rerank.reranker,
          documentIds: documentIds ?? undefined,
        }),
      citationSource: (passageId) => documents.citationSource(passageId),
      pageTexts: (documentId, from, to) => documents.pageTexts(documentId, from, to),
    },
    connectorTools: (signal) => connectors.toolsForAnswer(signal),
    skills: {
      availability: (name) => skills.availability(name),
      openSession: (forced) => skills.openSession(forced),
    },
    resolveScope: (scope) =>
      resolveSearchScope(scope, {
        folderTree: (folderId) => folders.subtree(folderId),
        documentIds: (filter) => documents.ids(filter),
      }),
    emptyScopeAnswer: () => translate(settings.get().language, "scope.answer.empty"),
    reportError: (error) => console.error(error),
  });

  const mindExports = createExports({
    mind: (mindId) => minds.get(mindId),
    read: (mindId, look) => content.read(mindId, look),
    liveDocuments: () => documents.list(),
    language: () => settings.get().language,
    now,
  });

  // Automatic tagging: Jev when it is set up on this device, otherwise the
  // default chat model; either way its own data flow and consent.
  const jev = createJevTagging({
    settings,
    secrets,
    consent,
    hostedUrl: adapters.jevHostedUrl,
    signal: lifetime.signal,
  });
  consent.registry.register({
    id: "tagging",
    sends: TAGGING_FLOW_SENDS,
    async services() {
      const service = jev.enabled() ? jev.service() : chat.defaultService();
      return service ? [service] : [];
    },
  });
  /** Pushes "documents.tagged" for those of these Documents that still exist. */
  const announceTagged = (documentIds: readonly string[]) => {
    const live = documents.getMany(documentIds);
    if (live.length > 0) events.emit("documents.tagged", live);
  };
  const tagger = createTagger({
    db,
    now,
    tags,
    canRun: async () =>
      jev.enabled() ? jev.canRun() : (await chat.readiness(undefined, "tagging")).ready,
    mightBeReady: () => (jev.enabled() ? jev.mightBeReady() : chat.mightBeReady("tagging")),
    prepare: async () =>
      jev.enabled()
        ? jev.prepare()
        : chatClassifier((await chat.prepareModel(undefined, "tagging")).model),
    announce: announceTagged,
    reportError: (error) => console.error(error),
  });
  /** Jev was set up, changed or removed: say so, and Documents waiting may go on. */
  const jevChanged = async () => {
    const status = await jev.status();
    if (lifetime.signal.aborted) return status;
    events.emit("jev.changed", status);
    tagger.resume();
    return status;
  };
  documentReady = (documentId) => tagger.documentReady(documentId);

  const settingsChanged = () => events.emit("settings.changed", settings.get());
  const readinessChanged = async () => {
    try {
      const readiness = await chat.readiness();
      if (!lifetime.signal.aborted) events.emit("chatReadiness.changed", readiness);
    } catch (error) {
      // Reading the keychain is async, so the core may have closed meanwhile: nobody is listening.
      if (!lifetime.signal.aborted) throw error;
    }
    // Documents waiting for a chat model may be able to go on now.
    if (!lifetime.signal.aborted) tagger.resume();
  };
  chatGptChanged = () => {
    const report = async () => {
      const status = await chatGpt.status();
      if (lifetime.signal.aborted) return;
      events.emit("chatGptPlan.changed", status);
      await readinessChanged();
    };
    report().catch((error: unknown) => {
      if (!lifetime.signal.aborted) console.error(error);
    });
  };

  // Tagging a quit interrupted starts again; Documents waiting for a chat model are checked.
  tagger.start();

  connectorsToggled = () => {
    syncChatFlow();
    readinessChanged().catch((error: unknown) => console.error(error));
  };
  // Connectors that are on start with the app.
  connectors.startAll();

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
      answers.settleOrphans(mind.id);
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
      const { folderId, includeSubfolders, tagId } = parseListOptions(options);
      const folder = folderId === undefined ? undefined : folders.get(folderId);
      return documents.list({
        folderIds: folder && (includeSubfolders ? folders.subtree(folder.id) : [folder.id]),
        tagId: tagId === undefined ? undefined : tags.get(tagId).id,
      });
    },
    renameDocument: async (id, name) => documents.rename(id, name),
    deleteDocument: (id) => documents.delete(id),
    searchPassages: (query, options) => documents.search(query, options),
    getEmbeddingModel: async () => embeddingModel.status(),
    downloadEmbeddingModel: async () => embeddingModel.retry(),

    getEmbeddingSettings: () => embeddingSettings(),
    saveEmbeddingProvider: async (input) => {
      await embedding.save(input);
      return embeddingSettings();
    },
    testEmbeddingConnection: (input) => embedding.test(input),
    retryEmbedding: async () => {
      embedding.retry();
      return embeddingSettings();
    },
    setLocalOnly: (enabled) => setLocalOnly(enabled),

    getRerankSettings: () => rerank.status(),
    saveRerankSettings: async (input) => {
      await rerank.save(input);
      return rerankChanged();
    },
    removeRerankSettings: async () => {
      await rerank.remove();
      return rerankChanged();
    },
    testRerankConnection: (input) => rerank.test(input),

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
    listChatModels: () => chat.listModels(),

    askQuestion: async (input) => answers.ask(input),
    regenerateAnswer: async (input) => answers.regenerate(input),
    stopAnswer: async (input) => answers.stop(input),

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
      // Local mode: Document search stays on this computer too. A cloud
      // embedding provider goes back to the built-in model, and the rebuild
      // ("embedding.changed", reason "local-mode") tells the User.
      await setLocalOnly(true);
      return provider;
    },

    getChatGptPlan: () => chatGpt.status(),
    setChatGptPlanEnabled: async (enabled) => {
      await chatGpt.setEnabled(enabled);
      if (!enabled) {
        // Off means nothing of it is left: no tokens (signed out above) and no provider.
        await chat.deleteKind("chatgpt");
        settingsChanged();
        await readinessChanged();
      }
      return chatGpt.status();
    },
    signInToChatGpt: () => chatGpt.signIn(),
    cancelChatGptSignIn: () => chatGpt.cancelSignIn(),
    signOutOfChatGpt: () => chatGpt.signOut(),

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

    listTags: async () => tags.list(),
    createTag: async (input) => {
      const tag = tags.create(input);
      tagsChanged();
      return tag;
    },
    updateTag: async (tagId, patch) => {
      const tag = tags.update(tagId, patch);
      tagsChanged();
      return tag;
    },
    deleteTag: async (tagId) => {
      const documentIds = tags.delete(tagId, now());
      // Tags first, so a listener filtering by the deleted Tag hears it's gone before it refreshes.
      tagsChanged();
      announceTagged(documentIds);
    },
    addDocumentTag: async (documentId, tagId) => {
      const { document, changed } = db.transaction(() => {
        const document = documents.get(documentId);
        return { document, changed: tags.addToDocument(document.id, tagId) };
      });
      if (!changed) return document;
      const updated = documents.get(document.id);
      events.emit("documents.tagged", [updated]);
      return updated;
    },
    removeDocumentTag: async (documentId, tagId) => {
      const { document, changed } = db.transaction(() => {
        const document = documents.get(documentId);
        return { document, changed: tags.removeFromDocument(document.id, tagId) };
      });
      if (!changed) return document;
      const updated = documents.get(document.id);
      events.emit("documents.tagged", [updated]);
      return updated;
    },
    retagDocuments: async (documentIds) => tagger.retag(documentIds),

    getJevSettings: () => jev.status(),
    saveJevSettings: async (input) => {
      await jev.save(input);
      return jevChanged();
    },
    removeJevSettings: async () => {
      await jev.remove();
      return jevChanged();
    },
    testJevConnection: (input) => jev.test(input),

    listConnectors: async () => connectors.list(),
    addConnector: (input) => connectors.add(input),
    setConnectorEnabled: async (connectorId, enabled) =>
      connectors.setEnabled(connectorId, enabled),
    restartConnector: async (connectorId) => connectors.restart(connectorId),
    deleteConnector: (connectorId) => connectors.delete(connectorId),
    previewConnectorImport: async (json) => connectors.previewImport(json),
    importConnectors: (json) => connectors.import(json),

    listSkills: async () => skills.list(),
    previewSkillImport: (path) => skills.preview(path),
    importSkill: async (importId) => {
      const skill = await skills.import(importId);
      skillsChanged();
      return skill;
    },
    cancelSkillImport: async (importId) => skills.cancel(importId),
    setSkillEnabled: async (skillId, enabled) => {
      const skill = skills.setEnabled(skillId, enabled);
      skillsChanged();
      return skill;
    },
    removeSkill: async (skillId) => {
      await skills.remove(skillId);
      skillsChanged();
    },
    duplicateSkill: async (skillId) => {
      const skill = await skills.duplicate(skillId);
      skillsChanged();
      return skill;
    },
    listRemovedBuiltInSkills: async () => skills.removedBuiltIns(),
    restoreBuiltInSkills: async () => {
      const restored = await skills.restoreBuiltIns();
      if (restored.length > 0) skillsChanged();
      return restored;
    },

    previewMindExport: async (mindId, options) => mindExports.preview(mindId, options),
    exportMind: async (mindId, options) => mindExports.export(mindId, options),

    openDocumentFile: (documentId) => documents.openFile(documentId),
    on: (event, listener) => events.on(event, listener),
    onAnyEvent: (listener) => events.onAny(listener),
    dataFlows: consent.registry,
    prepareChatModel: (choice) => chat.prepareModel(choice),
    close: () => {
      if (lifetime.signal.aborted) return;
      lifetime.abort();
      void chatGpt.cancelSignIn();
      // Answers being written keep what they have, marked "stopped".
      answers.stopAll();
      connectors.close();
      tagger.close();
      skills.close();
      consent.close();
      documents.close();
      embeddingModel.close();
      events.clear();
      content.closeAll();
      db.close();
    },
  };
}
