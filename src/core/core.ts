import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { translate } from "../shared/i18n";
import { logActivity, silentLogger } from "./activityLog";
import type { CoreAdapters } from "./adapters";
import { createAiSdkAnswerEngine, createAnswers } from "./answers";
import { recheckCitation } from "./answers/citations";
import { addCitedVersions, type CitedVersions, isCited } from "./answers/citedVersions";
import type {
  ChatModelChoice,
  CoreApi,
  CoreEventSource,
  EmbeddingSettings,
  ProviderError,
  Unsubscribe,
} from "./api";
import { createApprovals } from "./approvals";
import { createConnectors } from "./connectors";
import { createConsent, type DataFlowRegistry } from "./consent";
import { createDocuments, type DocumentFile, parseListOptions } from "./documents";
import { fsWatchFolder } from "./documents/watcher";
import { BUILT_IN_EMBEDDING_MODEL, createEmbeddingModel } from "./embedding";
import { createActiveEmbedding } from "./embedding/active";
import { InvalidInputError, isRecord } from "./errors";
import { type AnyEventListener, createEventHub } from "./events";
import { createExports } from "./exports";
import { createFolders } from "./folders";
import { createMindContent } from "./mindContent";
import { createMinds, parseMindId } from "./minds";
import { createPrivacy, type NetworkTrafficRegistry } from "./privacy";
import { CHAT_FLOW_SENDS, createChat, type PreparedChatModel } from "./providers/chat";
import { CHATGPT_PLAN_ENDPOINTS, createChatGptPlan } from "./providers/chatgpt/plan";
import { createAiSdkEmbeddingModel } from "./providers/embeddings";
import { ollamaBaseUrl } from "./providers/kinds";
import { createAiSdkChatModel } from "./providers/models";
import {
  detectOllama,
  hasOllamaModel,
  OLLAMA_REGISTRY,
  pullOllamaModel,
  RECOMMENDED_OLLAMA_MODEL,
} from "./providers/ollama";
import { createOllamaModels } from "./providers/ollamaModels";
import { createAiSdkRerankingModel, createRerank } from "./providers/rerank";
import { resolveSearchScope } from "./scope";
import { createSecrets } from "./secrets";
import { createSettings, isChatModelChoice } from "./settings";
import { createSkills } from "./skills";
import { createScriptRunner } from "./skills/scripts";
import { migrate, openDatabase } from "./storage";
import { createTags } from "./tags";
import { chatClassifier } from "./tags/classify";
import { createJevTagging } from "./tags/jev";
import { createTagger, TAGGING_FLOW_SENDS } from "./tags/tagger";

export const DATABASE_FILE = "incarnamind.db";

/** The core as its host sees it: the public interface (methods and events) plus host-only hooks. */
export interface Core extends CoreApi, CoreEventSource {
  /**
   * Opens a Document's file where the User keeps it, for the host to serve
   * to the UI (the desktop app streams it over a custom protocol, so files
   * never cross IPC). The file is checked first, as any opened file is.
   * Only live Documents whose file is there: throws NotFoundError for an
   * unknown or deleted Document, or one whose file is missing or can't be
   * reached (the UI then shows `readDocumentText` instead).
   */
  openDocumentFile(documentId: string): Promise<DocumentFile>;
  /**
   * Opens a live Document's file, where it is, in the default app for its
   * type, through the `shell` adapter. Throws NotFoundError as
   * `openDocumentFile` does.
   */
  openDocumentInApp(documentId: string): Promise<void>;
  /**
   * Shows a live Document's file selected in the system's file manager,
   * through the `shell` adapter. Throws NotFoundError as `openDocumentFile` does.
   */
  showDocumentInFolder(documentId: string): Promise<void>;
  /**
   * Reconciles every Linked folder and single file with the disk now, and
   * resolves once that, and any change already reported, has been taken in.
   */
  reconcileDocuments(): Promise<void>;
  /** Every event the core emits, for the host to forward to the UI. */
  onAnyEvent(listener: AnyEventListener): Unsubscribe;
  /** Every external data flow. Core modules register theirs here; consent covers each one. */
  readonly dataFlows: DataFlowRegistry;
  /**
   * Network traffic that carries no User content, for the Privacy page. Core
   * modules and the host (e.g. its update check) register theirs here.
   */
  readonly networkTraffic: NetworkTrafficRegistry;
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
  // Logs what happens from here on, from the events: never the User's content.
  const activity = logActivity(events, adapters.log ?? silentLogger);
  /** Logs a failed test of a provider's settings, by the kind of error. */
  const loggingTest =
    (what: Parameters<typeof activity.testFailed>[0]) =>
    async <R extends { ok: true } | { ok: false; error: ProviderError }>(
      test: Promise<R>,
    ): Promise<R> => {
      const result = await test;
      if (!result.ok) activity.testFailed(what, result.error.kind);
      return result;
    };
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
  const privacy = createPrivacy({ settings, crashReporter: adapters.crashReporter });
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
  const platform = process.platform;
  let documents: ReturnType<typeof createDocuments>;
  /** Set once created, so a failure after it can close it. */
  let createdDocuments: ReturnType<typeof createDocuments> | undefined;
  try {
    documents = createdDocuments = createDocuments({
      db,
      dataDir,
      now,
      model: embedding.model,
      folders,
      emitStatus: (document) => {
        events.emit("document.status", document);
        rebuildMayHaveProgressed();
      },
      emitMoved: (moved) => events.emit("documents.moved", moved),
      emitRemoved: (ids) => events.emit("documents.removed", ids),
      foldersChanged,
      linkedFoldersChanged: (list) => events.emit("linkedFolders.changed", list),
      onReady: (documentId) => documentReady(documentId),
      // iCloud Drive before macOS 14 downloads a stub's file when asked by its command-line tool.
      ...(platform === "darwin" && {
        downloadStub: async (path: string) => {
          const child = await adapters.processes.spawn("brctl", ["download", path]);
          await new Promise<void>((resolve) => {
            child.once("exit", () => resolve());
            child.once("error", () => resolve());
          });
        },
      }),
      linkedFolders: {
        watch:
          adapters.linkedFolders?.watch ?? fsWatchFolder(adapters.linkedFolders?.settleMs ?? 300),
        retryMs: adapters.linkedFolders?.retryMs ?? 30_000,
        detectDataless: adapters.linkedFolders?.detectDatalessFiles ?? platform === "darwin",
      },
    });
    // Old versions' text no Citation quotes any more goes, at startup only:
    // nothing is being written or undone then, so no Citation is in flight.
    if (documents.hasOldVersions()) {
      const cited: CitedVersions = new Map();
      for (const mind of minds.list()) {
        content.read(mind.id, (fragment) => addCitedVersions(fragment, cited));
        content.close(mind.id);
      }
      documents.collectOldVersions((documentId, contentHash) =>
        isCited(cited, documentId, contentHash),
      );
    }
  } catch (error) {
    lifetime.abort();
    // Created, but collecting old versions failed.
    createdDocuments?.close();
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
  /** Skills changed: say so; whether scripts can run, and so what chat sends, may have too. */
  const skillsChanged = () => {
    events.emit("skills.changed", skills.list());
    if (syncChatFlow()) readinessChanged().catch((error: unknown) => console.error(error));
  };

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
    ollamaModels: adapters.ollamaModels ?? createOllamaModels(),
  });

  /** Set once readiness can be reported: Connectors turned on or off change what chat sends. */
  let connectorsToggled = () => {};
  const connectors = createConnectors({
    db,
    now,
    clock: () => (adapters.now?.() ?? new Date()).getTime(),
    secrets,
    processes: adapters.processes,
    browser: adapters.browser,
    consent,
    signInPages: (name) => {
      const { language } = settings.get();
      return {
        success: {
          title: translate(language, "remoteConnectors.page.success.title", { name }),
          body: translate(language, "remoteConnectors.page.success.body"),
        },
        failure: {
          title: translate(language, "remoteConnectors.page.failure.title", { name }),
          body: translate(language, "remoteConnectors.page.failure.body"),
        },
      };
    },
    onChange: (list) => events.emit("connectors.changed", list),
    onEnabledChange: () => connectorsToggled(),
    reportError: (error) => console.error(error),
    ...(adapters.connectorSignInTimeoutMs !== undefined && {
      timing: { signInTimeoutMs: adapters.connectorSignInTimeoutMs },
    }),
  });
  /** Skill scripts can run: the User hasn't turned them off, and a Skill that is on has some. */
  const scriptsOffered = () =>
    settings.get().device.skillScriptsEnabled && skills.anyEnabledWithScripts();
  /**
   * With a Connector on, Answers send the chat model what its Tools return,
   * so the chat flow sends "tool-results" too, and asks again for it. So do
   * Skill scripts that can run (#41): what they print could be anything.
   *
   * Skills alone don't: their Tools (`use_skill`, `read_skill_file`) return
   * only the text of Skills the User imported, which is sent like the system
   * prompt, not data fetched from another service.
   */
  const sendsToolResults = () => connectors.anyEnabled() || scriptsOffered();
  /** Whether the chat flow sends "tool-results", as last registered. */
  let toolResultsSent: boolean | null = null;
  /** Registers what the chat flow sends now; true if "tool-results" came or went. */
  const syncChatFlow = (): boolean => {
    const flow = consent.registry.get("chat");
    if (!flow) return false;
    const withResults = sendsToolResults();
    const sends = withResults ? [...CHAT_FLOW_SENDS, "tool-results" as const] : CHAT_FLOW_SENDS;
    consent.registry.register({ ...flow, sends });
    const changed = toolResultsSent !== null && toolResultsSent !== withResults;
    toolResultsSent = withResults;
    return changed;
  };
  syncChatFlow();

  // Running Skill scripts (#41), each in its own temporary folder.
  const scriptRunner = createScriptRunner({
    processes: adapters.processes,
    runtimes: adapters.scriptRuntimes,
    tempDir: adapters.paths.tempDir ?? tmpdir(),
    reportError: (error) => console.error(error),
  });

  // Asking the User before a Connector Tool that may change something runs (#38).
  const approvals = createApprovals({
    db,
    events,
    now,
    owners: {
      connectors: () => new Map(connectors.list().map((each) => [each.id, each.name])),
      skills: () => new Map(skills.list().map((each) => [each.id, each.name])),
    },
  });

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
      searchableNames: (documentIds, limit) =>
        documents.searchableNames(documentIds ?? undefined, limit),
      search: (query, documentIds, signal) =>
        documents.searchTool(query, {
          signal,
          rerank: adapters.reranker ?? rerank.reranker,
          documentIds: documentIds ?? undefined,
        }),
      citationSource: (passageId) => documents.citationSource(passageId),
      pageTexts: (documentId, contentHash, from, to) =>
        documents.pageTexts(documentId, contentHash, from, to),
    },
    connectorTools: (signal) => connectors.toolsForAnswer(signal),
    connectorsNeedingSignIn: () => connectors.needingSignIn(),
    approvals: {
      toolNeedsApproval: (connectorId, tool, readOnly) =>
        approvals.toolNeedsApproval(connectorId, tool, readOnly),
      scriptNeedsApproval: (skillId) => approvals.scriptNeedsApproval(skillId),
      request: (call, signal) => approvals.request(call, signal),
    },
    scripts: {
      enabled: () => settings.get().device.skillScriptsEnabled,
      timeoutSeconds: () => settings.get().device.skillScriptTimeoutSeconds,
      check: (script) => scriptRunner.check(script),
      run: (request) => scriptRunner.run(request),
    },
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

  // Traffic that carries nothing of the User's, for the Privacy page.
  const modelSource = adapters.embeddingModelSource ?? BUILT_IN_EMBEDDING_MODEL.source;
  const modelHost = new URL(modelSource.baseUrl);
  privacy.traffic.register({
    id: "embedding-model",
    service: {
      id: modelHost.origin,
      name: modelHost.host === "huggingface.co" ? "Hugging Face" : modelHost.host,
    },
    // A model with no files to download (the tests' fake) makes no traffic.
    listed: () => modelSource.files.length > 0,
  });
  privacy.traffic.register({ id: "ollama-pull", service: OLLAMA_REGISTRY });
  const chatGptSignIn = new URL(
    adapters.chatGptPlan?.authorizeUrl ?? CHATGPT_PLAN_ENDPOINTS.authorizeUrl,
  );
  privacy.traffic.register({
    id: "chatgpt-sign-in",
    service: { id: chatGptSignIn.origin, name: "OpenAI" },
    listed: () => chatGpt.enabled(),
  });
  privacy.traffic.register({
    id: "remote-connectors",
    services: () => connectors.remoteTraffic(),
  });
  // Crash reports start now if the User opted in before; otherwise not at all.
  privacy.start();
  const privacyChanged = () => events.emit("privacy.changed", privacy.status());
  /** After the User allows or revokes a flow: the flows as they are now, and what that changes. */
  const dataFlowsChanged = async () => {
    const flows = await consent.listRegistered();
    if (lifetime.signal.aborted) return;
    events.emit("dataFlows.changed", flows);
    await readinessChanged();
  };

  connectorsToggled = () => {
    syncChatFlow();
    readinessChanged().catch((error: unknown) => console.error(error));
  };
  // Connectors that are on start with the app.
  connectors.startAll();
  // Linked folders and single files are compared with the disk, then watched.
  documents.start();

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
      const scriptsWere = settings.get().device.skillScriptsEnabled;
      const updated = settings.update(patch);
      events.emit("settings.changed", updated);
      // Skill scripts turned off: none runs any more, and none waits to.
      if (scriptsWere && !updated.device.skillScriptsEnabled) {
        approvals.denyScripts();
        scriptRunner.stopAll();
      }
      const flowChanged = syncChatFlow();
      if (chatModel !== undefined || flowChanged) await readinessChanged();
      return updated;
    },
    addDocuments: (paths) => documents.add(paths),
    listDocuments: async (options) => {
      const { folderId, includeSubfolders, tagId, linkedFolderId } = parseListOptions(options);
      const folder = folderId === undefined ? undefined : folders.get(folderId);
      return documents.list({
        folderIds: folder && (includeSubfolders ? folders.subtree(folder.id) : [folder.id]),
        tagId: tagId === undefined ? undefined : tags.get(tagId).id,
        linkedFolderId,
      });
    },
    renameDocument: async (id, name) => documents.rename(id, name),
    deleteDocument: async (id) => {
      await documents.delete(id);
      activity.documentDeleted(id as string);
    },
    readDocumentText: async (id) => documents.readText(id),
    previewLinkedFolder: (path) => documents.linkedFolders.preview(path),
    addLinkedFolder: (path) => documents.linkedFolders.add(path),
    listLinkedFolders: async () => documents.linkedFolders.list(),
    removeLinkedFolder: async (linkedFolderId) => documents.linkedFolders.remove(linkedFolderId),
    setLinkedFolderLayout: async (linkedFolderId, layout) =>
      documents.linkedFolders.setLayout(linkedFolderId, layout),
    setLinkedFolderPaused: async (linkedFolderId, paused) =>
      documents.linkedFolders.setPaused(linkedFolderId, paused),
    downloadOnlineOnlyFiles: async (linkedFolderId) =>
      documents.linkedFolders.downloadOnlineOnly(linkedFolderId),
    recheckCitation: async (input) => {
      if (!isRecord(input) || typeof input.documentId !== "string" || input.documentId === "") {
        throw new InvalidInputError("recheckCitation needs the Citation's Document id.");
      }
      const page = (value: unknown) =>
        typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : null;
      return recheckCitation(documents, {
        documentId: input.documentId,
        quote: typeof input.quote === "string" ? input.quote : "",
        pageFrom: page(input.pageFrom),
        pageTo: page(input.pageTo) ?? page(input.pageFrom),
      });
    },
    searchPassages: (query, options) => documents.search(query, options),
    getEmbeddingModel: async () => embeddingModel.status(),
    downloadEmbeddingModel: async () => embeddingModel.retry(),

    getEmbeddingSettings: () => embeddingSettings(),
    saveEmbeddingProvider: async (input) => {
      await embedding.save(input);
      return embeddingSettings();
    },
    testEmbeddingConnection: (input) => loggingTest("embedding")(embedding.test(input)),
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
    testRerankConnection: (input) => loggingTest("rerank")(rerank.test(input)),

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
    testChatConnection: (input) => loggingTest("chat")(chat.test(input)),
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
      await dataFlowsChanged();
    },
    listRegisteredDataFlows: () => consent.listRegistered(),
    allowDataFlow: async (flowId, serviceId) => {
      await consent.allow(flowId, serviceId);
      await dataFlowsChanged();
    },

    listNetworkTraffic: () => privacy.listTraffic(),
    getPrivacySettings: async () => privacy.status(),
    updatePrivacySettings: async (patch) => {
      const { changed, settings: updated } = privacy.update(patch);
      if (changed) privacyChanged();
      return updated;
    },

    listFolders: async () => folders.list(),

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
    testJevConnection: (input) => loggingTest("jev")(jev.test(input)),

    listConnectors: async () => connectors.list(),
    addConnector: (input) => connectors.add(input),
    setConnectorEnabled: async (connectorId, enabled) =>
      connectors.setEnabled(connectorId, enabled),
    restartConnector: (connectorId) => connectors.restart(connectorId),
    deleteConnector: async (connectorId) => {
      await connectors.delete(connectorId);
      // Its "always allow" and "ask" go with it: added again, it is a new Connector.
      approvals.forgetConnector(connectorId);
    },
    signInToConnector: (connectorId) => connectors.signIn(connectorId),
    cancelConnectorSignIn: (connectorId) => connectors.cancelSignIn(connectorId),
    signOutOfConnector: (connectorId) => connectors.signOut(connectorId),
    setConnectorClient: (connectorId, client) => connectors.setClient(connectorId, client),
    previewConnectorImport: async (json) => connectors.previewImport(json),
    importConnectors: (json) => connectors.import(json),

    listApprovalPolicies: async () => approvals.list(),
    setApprovalPolicy: async (input) => approvals.set(input),
    revokeApprovalPolicy: async (policyId) => approvals.revoke(policyId),
    listApprovalRequests: async () => approvals.requests(),
    respondToApproval: async (requestId, decision, options) =>
      approvals.respond(requestId, decision, options),

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
      approvals.forgetSkill(skillId);
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
    openDocumentInApp: async (documentId) => {
      const path = await documents.filePath(documentId);
      if (!adapters.shell)
        throw new Error("This copy of IncarnaMind can't open files in other apps.");
      await adapters.shell.openPath(path);
    },
    showDocumentInFolder: async (documentId) => {
      const path = await documents.filePath(documentId);
      if (!adapters.shell) throw new Error("This copy of IncarnaMind can't show files.");
      adapters.shell.showItemInFolder(path);
    },
    reconcileDocuments: () => documents.linkedFolders.sync(),
    on: (event, listener) => events.on(event, listener),
    onAnyEvent: (listener) => events.onAny(listener),
    dataFlows: consent.registry,
    networkTraffic: privacy.traffic,
    prepareChatModel: (choice) => chat.prepareModel(choice),
    close: () => {
      if (lifetime.signal.aborted) return;
      lifetime.abort();
      void chatGpt.cancelSignIn();
      // Answers being written keep what they have, marked "stopped"; Tool calls waiting for
      // the User's approval are denied, and Skill scripts running are stopped.
      answers.stopAll();
      scriptRunner.close();
      approvals.close();
      connectors.close();
      tagger.close();
      skills.close();
      consent.close();
      documents.close();
      embeddingModel.close();
      activity.stop();
      events.clear();
      content.closeAll();
      db.close();
    },
  };
}
