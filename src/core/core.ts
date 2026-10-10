import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { translate } from "../shared/i18n";
import { logActivity, silentLogger } from "./activityLog";
import type { CoreAdapters } from "./adapters";
import { createAiSdkAnswerEngine, createAnswers } from "./answers";
import { recheckCitation } from "./answers/citations";
import {
  addCitedUnits,
  addCitedVersions,
  type CitedUnits,
  type CitedVersions,
  isCited,
} from "./answers/citedVersions";
import type {
  ChatModelChoice,
  CoreApi,
  CoreEventSource,
  EmbeddingSettings,
  ProviderError,
  Unsubscribe,
} from "./api";
import { createApprovals } from "./approvals";
import { createBackgroundQueue } from "./backgroundQueue";
import { createConnectors } from "./connectors";
import { createConsent, type DataFlowRegistry } from "./consent";
import {
  createDocuments,
  type DocumentFile,
  type DocumentImage,
  parseListOptions,
} from "./documents";
import { fsWatchFolder } from "./documents/watcher";
import { BUILT_IN_EMBEDDING_MODEL, createEmbeddingModel } from "./embedding";
import { createActiveEmbedding } from "./embedding/active";
import { InvalidInputError, isRecord, TaggingNotReadyError } from "./errors";
import { type AnyEventListener, createEventHub } from "./events";
import { createExamples } from "./examples";
import { createLocalExecutor } from "./execution";
import { createExports } from "./exports";
import { createFolders } from "./folders";
import { CLASSIFICATION_FLOW_SENDS, createLibrary } from "./library";
import { automaticGroupClassifier } from "./library/automatic";
import { chatGroupClassifier, decisionGroupClassifier } from "./library/classifier";
import { documentPageImages } from "./library/pageImages";
import { createMindContent } from "./mindContent";
import { createMinds, parseMindId } from "./minds";
import { createPrivacy, type NetworkTrafficRegistry } from "./privacy";
import { CHAT_FLOW_SENDS, createChat, type PreparedChatModel } from "./providers/chat";
import { CHATGPT_PLAN_ENDPOINTS, createChatGptPlan } from "./providers/chatgpt/plan";
import { createAiSdkEmbeddingModel } from "./providers/embeddings";
import { chatModelReadsImages } from "./providers/imageInput";
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
import { BUILT_IN_RERANKING_MODEL, createRerankingModel } from "./reranking";
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
import { createUsageData } from "./usageData";
import { addedCounts, trackUsage } from "./usageTracking";

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
   * Opens a picture a live Markdown Document shows from beside it, by the
   * relative path written in the file ("figures/map.png"), for the host to
   * serve to the viewer: only images, only from the file's folder or below
   * it. Throws NotFoundError for anything else.
   */
  openDocumentImage(documentId: string, path: string): Promise<DocumentImage>;
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
  // The level Skill scripts run at: the given Executor's, else the local one's below ("none").
  const scriptSandbox = adapters.executor?.level ?? "none";
  const settings = createSettings(db, now, adapters.systemLanguages, scriptSandbox);
  const folders = createFolders(db, now);
  const foldersChanged = () => events.emit("folders.changed", folders.list());
  const tags = createTags(db, now);
  // First run: the preset Tags, in the interface language of the time.
  tags.seedPresets(settings.get().language);
  const tagsChanged = () => events.emit("tags.changed", tags.list());
  /** Set once automatic tagging exists: it hears about every Document that becomes ready. */
  let documentReady = (_documentId: string) => {};
  /** Set once rerank exists: a Document became ready, so the built-in reranking model is needed. */
  let prepareRerank = () => {};
  let libraryDocumentChanged = (_documentId: string) => {};
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
  // Usage data: only from a copy built to send it, only while the User agrees, never in local mode.
  const usageData = createUsageData({
    settings,
    sender: adapters.usageData,
    localOnly: () => embedding.localOnly(),
  });
  const privacy = createPrivacy({ settings, crashReporter: adapters.crashReporter, usageData });
  const privacyChanged = () => events.emit("privacy.changed", privacy.status());
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
        libraryDocumentChanged(document.id);
        rebuildMayHaveProgressed();
      },
      emitMoved: (moved) => events.emit("documents.moved", moved),
      emitRemoved: (ids) => events.emit("documents.removed", ids),
      emitKept: (kept) => events.emit("keptCitationTexts.changed", kept),
      foldersChanged,
      linkedFoldersChanged: (list) => events.emit("linkedFolders.changed", list),
      onReady: (documentId) => {
        // There is something to search now: the built-in reranking model downloads, if it reranks.
        prepareRerank();
        documentReady(documentId);
      },
      // Citations live in Minds, anywhere: in Answers, or copied into Notes.
      citedUnits: (documentIds) => {
        const wanted = new Set(documentIds);
        const cited: CitedUnits[] = [];
        for (const mind of minds.list()) {
          content.peek(mind.id, (fragment) => addCitedUnits(fragment, wanted, cited));
        }
        return cited;
      },
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
    // Old versions' text, and the text kept of unlinked Documents, that no
    // Citation quotes any more goes, at startup only: nothing is being
    // written or undone then, so no Citation is in flight.
    if (documents.hasOldVersions() || documents.keptCitationTexts().length > 0) {
      const cited: CitedVersions = new Map();
      for (const mind of minds.list()) {
        content.peek(mind.id, (fragment) => addCitedVersions(fragment, cited));
      }
      const quoted = (documentId: string, contentHash: string) =>
        isCited(cited, documentId, contentHash);
      documents.collectOldVersions(quoted);
      documents.releaseKeptText(quoted);
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

  /** Reports rerank's settings again; set once rerank exists. */
  let reportRerank = () => {};
  // The built-in reranking model, the default: downloaded once there are Documents to search.
  const rerankingModel = createRerankingModel({
    definition: BUILT_IN_RERANKING_MODEL,
    source: adapters.rerankingModelSource,
    dataDir,
    crossEncoder: adapters.crossEncoder,
    emitStatus: () => reportRerank(),
  });
  // Rerank: the built-in model by default, a Cohere or Voyage key (paused in local mode), or off.
  const rerank = createRerank({
    settings,
    secrets,
    consent,
    createModel: adapters.createRerankingModel ?? createAiSdkRerankingModel,
    builtIn: rerankingModel,
    localOnly: () => embedding.localOnly(),
    signal: lifetime.signal,
  });
  const rerankChanged = async () => {
    const status = await rerank.status();
    if (!lifetime.signal.aborted) events.emit("rerank.changed", status);
    return status;
  };
  reportRerank = () => {
    if (!rerank.usesBuiltIn()) return;
    rerankChanged().catch((error: unknown) => {
      // Reading the keychain is async, so the core may have closed meanwhile.
      if (!lifetime.signal.aborted) console.error(error);
    });
  };
  prepareRerank = () => rerank.prepare();
  // Documents to search from an earlier run: the built-in reranking model downloads now, or
  // carries on from where the app quit, if it reranks them.
  if (documents.searchableCount() > 0) rerank.prepare();
  /**
   * Local mode on or off: embeddings may switch back to the built-in model,
   * rerank pauses, and usage data stops.
   */
  const setLocalOnly = async (enabled: unknown) => {
    const wasLocal = embedding.localOnly();
    await embedding.setLocalOnly(enabled);
    library.resume();
    if (wasLocal !== embedding.localOnly()) {
      usageData.localModeChanged();
      privacyChanged();
      await rerankChanged();
    }
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

  // The programs Tools start go through one Executor (#61): by default the local one, at
  // sandbox level "none".
  const executor =
    adapters.executor ??
    createLocalExecutor({
      processes: adapters.processes,
      tempDir: adapters.paths.tempDir ?? tmpdir(),
      reportError: (error) => console.error(error),
    });
  // Running Skill scripts (#41), each in its own temporary folder.
  const scriptRunner = createScriptRunner({ executor, runtimes: adapters.scriptRuntimes });

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

  // Background model work (automatic tagging, and Organize's classification): one call at a
  // time, giving way to Answers.
  const background = createBackgroundQueue({ reportError: (error) => console.error(error) });

  const answers = createAnswers({
    content,
    events,
    engine: adapters.answerEngine ?? createAiSdkAnswerEngine({ runEngine: adapters.runEngine }),
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
      searchableLanguages: (documentIds) => documents.searchableLanguages(documentIds ?? undefined),
      search: (query, documentIds, signal) =>
        documents.searchTool(query, {
          signal,
          rerank: adapters.reranker ?? rerank.active(),
          documentIds: documentIds ?? undefined,
        }),
      citationSource: (passageId) => documents.citationSource(passageId),
      pageTexts: (documentId, contentHash, from, to) =>
        documents.pageTexts(documentId, contentHash, from, to),
    },
    toolProviders: [{ kind: "connector", tools: (signal) => connectors.tools(signal) }],
    connectorsNeedingSignIn: () => connectors.needingSignIn(),
    approvals: {
      decide: (call) => approvals.decide(call),
      request: (call, signal) => approvals.request(call, signal),
    },
    scripts: {
      enabled: () => settings.get().device.skillScriptsEnabled,
      timeoutSeconds: () => settings.get().device.skillScriptTimeoutSeconds,
      check: (script) => scriptRunner.check(script),
      access: (skillDir) => scriptRunner.access(skillDir),
      run: (request) => scriptRunner.run(request),
    },
    skills: {
      availability: (name) => skills.availability(name),
      openSession: (forced) => skills.openSession(forced),
    },
    resolveScope: (scope) =>
      resolveSearchScope(scope, {
        folderTree: (folderId) => folders.subtree(folderId),
        organizedFolderDocuments: (folderId) => library.documentIds(folderId),
        documentIds: (filter) => documents.ids(filter),
      }),
    emptyScopeAnswer: () => translate(settings.get().language, "scope.answer.empty"),
    onWritingChange: (writing) => background.setAnswering(writing),
    reportError: (error) => console.error(error),
  });

  const mindExports = createExports({
    mind: (mindId) => minds.get(mindId),
    read: (mindId, look) => content.read(mindId, look),
    liveDocuments: () => documents.list(),
    keptCitationTexts: () => documents.keptCitationTexts(),
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
  /**
   * The User's change to one Tag on several Documents: all of it or nothing,
   * in one transaction, then one "documents.tagged" event for those that changed.
   */
  const tagMany = (documentIds: unknown, change: (documentId: string) => boolean) => {
    if (
      !Array.isArray(documentIds) ||
      !documentIds.every((id) => typeof id === "string" && id !== "")
    )
      throw new InvalidInputError("documentIds must be a list of Document ids.");
    const ids = [...new Set(documentIds as string[])];
    const changed = db.transaction(() => ids.filter((id) => change(documents.get(id).id)));
    if (changed.length > 0) events.emit("documents.tagged", documents.getMany(changed));
    return documents.getMany(ids);
  };
  // Existing installations keep the legacy tagger until organization is configured.
  // Once folders are in use, only the selected organizer may suggest tags.
  const organizationConfigured = () =>
    !!settings.readDeviceValue("library") ||
    !!db.get("SELECT 1 FROM library_groups WHERE deleted_at IS NULL LIMIT 1");
  const tagger = createTagger({
    db,
    now,
    tags,
    enabled: () => !organizationConfigured(),
    canRun: async () =>
      jev.enabled() ? jev.canRun() : (await chat.readiness(undefined, "tagging")).ready,
    mightBeReady: () => (jev.enabled() ? jev.mightBeReady() : chat.mightBeReady("tagging")),
    prepare: async () => {
      if (jev.enabled()) return jev.prepare();
      const prepared = await chat.prepareModel(undefined, "tagging");
      // No service to send to: the model runs on a server on this computer.
      return chatClassifier(prepared.model, { local: prepared.provider.service === null });
    },
    announce: announceTagged,
    reportError: (error) => console.error(error),
    background,
  });
  const automaticClassifiers = new Map<string, ReturnType<typeof automaticGroupClassifier>>();
  const library = createLibrary({
    db,
    tags,
    tagged: announceTagged,
    now,
    settings,
    background,
    changed: () => events.emit("library.changed", null),
    assigned: (assignments) => events.emit("library.assignments", assignments),
    providerExists: (id) => chat.exists(id),
    chatReadsImages: (choice) => chat.readsImages(choice),
    async pageImages(id, contentHash, signal) {
      try {
        return await documentPageImages(await documents.filePath(id), contentHash, signal);
      } catch (error) {
        // A missing source file is a visible classification failure, not a retry loop.
        throw new Error(error instanceof Error ? error.message : String(error), { cause: error });
      }
    },
    async prepare({ classifier }) {
      if (!classifier) throw new TaggingNotReadyError("Choose a classification model.");
      if (classifier.kind === "auto") {
        let automatic = automaticClassifiers.get(classifier.baseUrl);
        if (!automatic) {
          automatic = automaticGroupClassifier(classifier.baseUrl);
          automaticClassifiers.set(classifier.baseUrl, automatic);
        }
        return automatic;
      }
      if (classifier.kind === "ollama")
        return decisionGroupClassifier({
          baseUrl: classifier.baseUrl,
          model: classifier.modelId,
          apiKey: "ollama",
          local: true,
          usePageImages: classifier.usePageImages,
        });
      if (classifier.kind === "jev") {
        if (embedding.localOnly() && jev.service())
          throw new TaggingNotReadyError("Cloud classification is paused in local mode.");
        const prepared = await jev.prepareGroups();
        if (embedding.localOnly() && !prepared.local)
          throw new TaggingNotReadyError("Cloud classification is paused in local mode.");
        return prepared;
      }
      const ready = await chat.readiness(classifier.choice, "classification");
      if (embedding.localOnly() && "provider" in ready && ready.provider.service)
        throw new TaggingNotReadyError("Cloud classification is paused in local mode.");
      const prepared = await chat.prepareModel(classifier.choice, "classification");
      if (embedding.localOnly() && prepared.provider.service)
        throw new TaggingNotReadyError("Cloud classification is paused in local mode.");
      // prepareModel got consent for everything the classification flow sends, page images included.
      const images = chatModelReadsImages(prepared.provider.kind, prepared.modelId);
      return {
        ...chatGroupClassifier(prepared.model, prepared.provider.service === null, images),
        model: { id: classifier.choice.modelId, images, reason: "selected" as const },
      };
    },
  });
  consent.registry.register({
    id: "classification",
    sends: CLASSIFICATION_FLOW_SENDS,
    async services() {
      const selected = library.settings().classifier;
      if (selected?.kind === "jev") {
        const service = jev.service();
        return service ? [service] : [];
      }
      if (selected?.kind === "chat") {
        const provider = (await chat.list()).find((item) => item.id === selected.choice.providerId);
        return provider?.service ? [provider.service] : [];
      }
      return [];
    },
  });
  libraryDocumentChanged = (id) => library.documentChanged(id);
  library.start();

  /** Jev was set up, changed or removed: say so, and Documents waiting may go on. */
  const jevChanged = async () => {
    const status = await jev.status();
    if (lifetime.signal.aborted) return status;
    events.emit("jev.changed", status);
    tagger.resume();
    library.resume();
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
    if (!lifetime.signal.aborted) {
      tagger.resume();
      library.resume();
    }
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
  const rerankingSource = adapters.rerankingModelSource ?? BUILT_IN_RERANKING_MODEL.source;
  const rerankingHost = new URL(rerankingSource.baseUrl);
  privacy.traffic.register({
    id: "reranking-model",
    service: {
      id: rerankingHost.origin,
      name: rerankingHost.host === "huggingface.co" ? "Hugging Face" : rerankingHost.host,
    },
    // While the built-in reranking model reranks: by default, unless the User chose otherwise.
    listed: () => rerankingSource.files.length > 0 && rerank.usesBuiltIn(),
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
  const usageSender = adapters.usageData;
  if (usageSender) {
    privacy.traffic.register({
      id: "usage-data",
      service: usageSender.service,
      enabled: () => usageData.status().enabled,
    });
  }
  // Crash reports start now if the User opted in before; otherwise not at all.
  privacy.start();
  // So does usage data: if the User agreed before, or, in a test build, hasn't said no.
  usageData.start();
  const stopTrackingUsage = trackUsage(events, (event) => usageData.record(event), {
    chatProvider: (providerId) => chat.describe(providerId),
    connectorIsRemote: (connectorId) => {
      const connector = connectors.list().find((each) => each.id === connectorId);
      return connector ? connector.transport === "http" : null;
    },
    skillIsBuiltIn: (name) => skills.list().some((skill) => skill.name === name && skill.builtIn),
    now: () => (adapters.now?.() ?? new Date()).getTime(),
  });
  usageData.record({ event: "app_opened", fields: { language: settings.get().language } });
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
  // The example Mind: its Citations are checked once its Documents have been read.
  const examples = createExamples({
    source: adapters.paths.examples,
    dataDir,
    language: () => settings.get().language,
    readValue: (key) => settings.readDeviceValue(key),
    writeValue: (key, value) => settings.writeDeviceValue(key, value),
    createMind: (title) => {
      const mind = minds.create({ title });
      mindsChanged();
      return mind;
    },
    mindExists: (mindId) => minds.list().some((mind) => mind.id === mindId),
    deleteMind: (mindId) => {
      const at = now();
      db.transaction(() => {
        const mind = minds.delete(mindId, at);
        content.remove(mind.id, at);
      });
      mindsChanged();
    },
    editMind: (mindId, change) => content.edit(mindId, change),
    linkFolder: (path) => documents.linkedFolders.add(path),
    unlinkFolder: (linkedFolderId) => documents.linkedFolders.remove(linkedFolderId),
    linkedFolderExists: (linkedFolderId) =>
      documents.linkedFolders.list().some((folder) => folder.id === linkedFolderId),
    documentsIn: (linkedFolderId) => documents.list({ linkedFolderId }),
    recheck: (input) => recheckCitation(documents, input),
    changed: (value) => events.emit("examples.changed", value),
  });
  events.on("document.status", () => examples.documentChanged());
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
      usageData.record({ event: "mind_created", fields: {} });
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
    addDocuments: async (paths) => {
      const since = now();
      const result = await documents.add(paths);
      usageData.record({ event: "documents_added", fields: addedCounts(result, since) });
      return result;
    },
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
    retryDocument: async (id) => documents.retry(id),
    readDocumentText: async (id) => documents.readText(id),
    previewLinkedFolder: (path) => documents.linkedFolders.preview(path),
    addLinkedFolder: async (path) => {
      const linked = await documents.linkedFolders.add(path);
      usageData.record({ event: "folder_linked", fields: {} });
      return linked;
    },
    listLinkedFolders: async () => documents.linkedFolders.list(),
    removeLinkedFolder: async (linkedFolderId) => documents.linkedFolders.remove(linkedFolderId),
    listKeptCitationTexts: async () => documents.keptCitationTexts(),
    getExamples: async (group) => examples.status(group),
    offerExamples: async () => examples.offer(minds.list().length > 0),
    createExamples: async (group) => examples.create(group),
    removeExamples: async (group) => examples.remove(group),
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
    downloadRerankingModel: async () => {
      await rerank.download();
      return rerankChanged();
    },
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
    recordUsage: async (event) => usageData.recordFromUi(event),
    resetUsageInstallId: async () => usageData.resetInstallId(),

    listFolders: async () => folders.list(),

    getLibrary: async () => library.snapshot(),
    createLibraryGroup: async (input) => library.create(input),
    updateLibraryGroup: async (id, input) => library.update(id, input),
    deleteLibraryGroup: async (id) => library.delete(id),
    addLibraryStarterGroups: async (keys) => library.addStarters(keys),
    saveLibrarySettings: async (input) => library.saveSettings(input),
    classifyDocuments: async (ids) => {
      library.classify(ids);
      const classifier = library.settings().classifier;
      if (classifier) {
        usageData.record({
          event: "organize_run",
          fields: {
            documents: Array.isArray(ids) ? new Set(ids).size : documents.list().length,
            all: ids === undefined,
            model: classifier.kind,
          },
        });
      }
    },
    assignDocumentGroup: async (id, groupId) => library.assign(id, groupId),

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
    addTagToDocuments: async (documentIds, tagId) =>
      tagMany(documentIds, (id) => tags.addToDocument(id, tagId)),
    removeTagFromDocuments: async (documentIds, tagId) =>
      tagMany(documentIds, (id) => tags.removeFromDocument(id, tagId)),
    mergeTags: async (tagId, intoTagId) => {
      const documentIds = tags.merge(tagId, intoTagId, now());
      // Tags first, so a listener filtering by the merged Tag hears it's gone before it refreshes.
      tagsChanged();
      announceTagged(documentIds);
      return tags.get(intoTagId);
    },
    retagDocuments: async (documentIds) => {
      if (organizationConfigured()) return library.classify(documentIds);
      tagger.retag(documentIds);
    },

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
    editConnector: (connectorId, input) => connectors.edit(connectorId, input),
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
    exportMind: async (mindId, options) => {
      const exported = await mindExports.export(mindId, options);
      usageData.record({
        event: "mind_exported",
        fields: {
          format: options.format,
          // Questions are in a Markdown export unless left out, and in a .docx only if asked for.
          questions:
            typeof options.includeQuestions === "boolean"
              ? options.includeQuestions
              : options.format === "markdown",
        },
      });
      return exported;
    },

    openDocumentFile: (documentId) => documents.openFile(documentId),
    openDocumentImage: (documentId, path) => documents.openImage(documentId, path),
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
      // First, so stopping the Answers below starts no background work.
      background.close();
      void chatGpt.cancelSignIn();
      // Answers being written keep what they have, marked "stopped"; Tool calls waiting for
      // the User's approval are denied, and Skill scripts running are stopped.
      answers.stopAll();
      scriptRunner.close();
      approvals.close();
      connectors.close();
      tagger.close();
      library.close();
      skills.close();
      stopTrackingUsage();
      consent.close();
      documents.close();
      embeddingModel.close();
      rerankingModel.close();
      activity.stop();
      events.clear();
      content.closeAll();
      db.close();
    },
  };
}
