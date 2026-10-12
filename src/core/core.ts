import { mkdirSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import * as Y from "yjs";
import { isCitationStyleId } from "../shared/citationStyles";
import { translate } from "../shared/i18n";
import { bibliographyShown } from "../shared/mindBibliography";
import {
  chosenCitationStyle,
  citationStyleIn,
  setCitationStyle,
} from "../shared/mindCitationStyle";
import { mindModelOf } from "../shared/mindModel";
import { paperFor } from "../shared/printDocument";
import { logActivity, silentLogger } from "./activityLog";
import type { CoreAdapters } from "./adapters";
import { agentSocketPath, createAgents } from "./agents";
import { createAgentSetup, skillInstalled } from "./agents/setup";
import { createAiUse } from "./aiUse";
import { createAiSdkAnswerEngine, createAnswers } from "./answers";
import { recheckCitation, recheckKept } from "./answers/citations";
import {
  addCitedUnits,
  addCitedVersions,
  type CitedUnits,
  type CitedVersions,
  isCited,
} from "./answers/citedVersions";
import { createCiting } from "./answers/citing";
import type {
  AddDocumentsResult,
  ChatModelChoice,
  CoreApi,
  CoreEventSource,
  Document,
  EmbeddingModelStatus,
  EmbeddingSettings,
  ExportMindOptions,
  Meeting,
  Mind,
  MindCitationStyle,
  OwnFileKind,
  ProviderError,
  SavedExport,
  SearchScope,
  Unsubscribe,
} from "./api";
import { createApprovals } from "./approvals";
import { createBackgroundQueue } from "./backgroundQueue";
import { createCitationStyles } from "./citationStyles";
import { createConnectors } from "./connectors";
import { createConsent, type DataFlowRegistry } from "./consent";
import { createConversations } from "./conversations";
import { createMeter } from "./cost/meter";
import {
  createDocuments,
  type DocumentFile,
  type DocumentImage,
  parseListOptions,
} from "./documents";
import { createEvidence, parseCheckedDocument, passageResult } from "./documents/evidence";
import { createOwnFiles } from "./documents/ownFiles";
import { createVersionRetention, type VersionRetention } from "./documents/retention";
import { BUILT_IN_TEXT_RECOGNITION_MODEL, createTextRecognitionModel } from "./documents/scan";
import { tunedSearch } from "./documents/searchTuning";
import { fsWatchFolder } from "./documents/watcher";
import { createWebPages } from "./documents/web";
import { createEdits } from "./edits";
import { BUILT_IN_EMBEDDING_MODEL, createEmbeddingModel } from "./embedding";
import { createActiveEmbedding } from "./embedding/active";
import {
  FileWriteCancelledError,
  InvalidInputError,
  isRecord,
  LinkedFolderWritesDeclinedError,
  MindReadOnlyError,
  NotFoundError,
  TaggingNotReadyError,
} from "./errors";
import { type AnyEventListener, createEventHub } from "./events";
import { createExamples } from "./examples";
import { createLocalExecutor } from "./execution";
import { createExports } from "./exports";
import { exportLabels } from "./exports/labels";
import { renderMarkdown } from "./exports/markdown";
import { createFileLock } from "./fileLock";
import { safeFileName, writeNewFile } from "./fileNames";
import { createFolders } from "./folders";
import { systemRegion } from "./language";
import { createLayouts } from "./layout";
import { CLASSIFICATION_FLOW_SENDS, createLibrary } from "./library";
import { automaticGroupClassifier } from "./library/automatic";
import { createBatches } from "./library/batches";
import { chatGroupClassifier, decisionGroupClassifier } from "./library/classifier";
import { documentPageImages } from "./library/pageImages";
import { createMeetings } from "./meetings";
import { listParts, meetingFolder } from "./meetings/audioFiles";
import { createCaptureHealth } from "./meetings/health";
import { createTranscripts } from "./meetings/transcript";
import { accessFor, isKnownKind } from "./mindAccess";
import { createMindContent } from "./mindContent";
import {
  createMinds,
  parseCreateMindInput,
  parseFolderChoice,
  parseIds,
  parseMindId,
} from "./minds";
import { firstLineOf, readBlocks } from "./mindText";
import { createNoteAdding } from "./noteAdding";
import { createOutputFiles } from "./outputFiles";
import { createPrivacy, type NetworkTrafficRegistry } from "./privacy";
import { readsImages } from "./providers/capabilities";
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
import { isCloudModelName } from "./providers/ollamaCloud";
import { createOllamaModels } from "./providers/ollamaModels";
import { createAiSdkRerankingModel, createRerank } from "./providers/rerank";
import { createRecentlyDeleted } from "./recentlyDeleted";
import {
  createReferenceImport,
  createReferences,
  REFERENCE_LOOKUP_SENDS,
  REFERENCE_SERVICE_URLS,
} from "./referenceRecords";
import { parseReference } from "./references";
import { BUILT_IN_RERANKING_MODEL, createRerankingModel } from "./reranking";
import { parseSearchScope, resolveSearchScope } from "./scope";
import { createSecrets } from "./secrets";
import { createSettings, isChatModelChoice } from "./settings";
import { createSkills } from "./skills";
import { createSkillDrafts } from "./skills/draft";
import { createScriptRunner } from "./skills/scripts";
import { createSources, FIND_SOURCES_SENDS } from "./sources";
import { createSourceHttp } from "./sources/http";
import { createPace, realTimers } from "./sources/pace";
import { createSourceProviders, sourceKeySecret } from "./sources/providers";
import { migrate, openDatabase } from "./storage";
import { createSuggestions } from "./suggestions";
import { createTags } from "./tags";
import { chatClassifier } from "./tags/classify";
import { createJevTagging } from "./tags/jev";
import { createTagger, TAGGING_FLOW_SENDS } from "./tags/tagger";
import { createWebSearch, WEB_SEARCH_SENDS } from "./tools/web";
import {
  createTranscriptionModel,
  QWEN3_ASR_0_6B,
  TRANSCRIPTION_SPEED,
} from "./transcription/model";
import { createUsageData } from "./usageData";
import { addedCounts, trackUsage } from "./usageTracking";
import { ensureWorkspace } from "./workspace";

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
   * The number a workbook's cell stores when its shown text isn't that
   * number written plainly (0.04213 for "4.2%"), from the Document's
   * current version; null when it shows its number plainly, isn't a number,
   * or isn't there, and for a CSV. `sheet` is the sheet's name (null for a
   * CSV). For the Tools that calculate over cells, so not on the bridge.
   * Throws NotFoundError for an unknown or deleted Document.
   */
  readCellNumber(documentId: string, sheet: string | null, ref: string): number | null;
  /**
   * Opens a live Document's file, where it is, in the default app for its
   * type, through the `shell` adapter; a web page opens live, at its
   * address, through the `browser` adapter. Throws NotFoundError as
   * `openDocumentFile` does.
   */
  openDocumentInApp(documentId: string): Promise<void>;
  /**
   * Shows a live Document's file selected in the system's file manager,
   * through the `shell` adapter. Throws NotFoundError as `openDocumentFile` does.
   */
  showDocumentInFolder(documentId: string): Promise<void>;
  /**
   * Shows a file IncarnaMind saved since it started (e.g. by
   * `saveMindExport`) selected in the system's file manager, through the
   * `shell` adapter. Throws NotFoundError for any other path.
   */
  showSavedFile(path: string): Promise<void>;
  /**
   * Opens a file IncarnaMind saved since it started in the default app for its
   * type, through the `shell` adapter. Throws NotFoundError for any other path.
   */
  openSavedFile(path: string): Promise<void>;
  /**
   * Reconciles every Linked folder and single file with the disk now, and
   * resolves once that, and any change already reported, has been taken in.
   */
  reconcileDocuments(): Promise<void>;
  /** The computer is about to sleep: a Meeting recording saves what it has and stops capturing. */
  systemSleeping(): void;
  /** The computer woke: a Meeting the sleep stopped records again, the sleep marked as a gap. */
  systemAwake(): void;
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
   * The versions of Documents that core features other than Citations still
   * need (#79): a check result's evidence, a saved page, an unfinished Task's
   * sources, a Checkpoint's Citations. A retained version keeps its text when
   * its file changes, and its Units retained when its Linked folder is
   * unlinked, as a quoted one does. Once its last owner releases it, and no
   * Citation quotes it, it goes at the next start.
   */
  readonly retention: VersionRetention;
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
  try {
    ensureWorkspace(db, now);
  } catch (error) {
    db.close();
    throw error;
  }
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
  /** The Mind, if this version may write to it: not one of a kind a newer version made. */
  const writableMind = (mindId: unknown): Mind => {
    const mind = minds.get(mindId);
    if (!isKnownKind(mind.kind)) {
      throw new MindReadOnlyError("A newer version of IncarnaMind made this. Update to open it.");
    }
    return mind;
  };
  const mindsChanged = () => events.emit("minds.changed", minds.list());
  // The assistant's conversations (ADR-0017); Answers a quit left being written are settled now.
  const conversations = createConversations(db, now);
  const conversationsChanged = () => events.emit("conversations.changed", null);
  const content = createMindContent(db, now, (mindId, update) => {
    events.emit("mind.update", { mindId, update });
    // Only notes the change; what became of AI text is worked out once the Mind is quiet.
    aiUse.contentStored(mindId);
    const renamed = minds.setAutoTitle(mindId, content.peek(mindId, firstLineOf));
    if (minds.markEdited(mindId) || renamed) mindsChanged();
  });
  // The record of AI use stays on this computer: it takes nothing from the network or the usage-data catalog.
  const aiUse = createAiUse({
    db,
    now,
    content,
    requireMind: (mindId) => void minds.get(mindId),
    changed: (status) => events.emit("aiUse.changed", status),
    reportError: (error) => console.error(error),
  });
  // A writing suggestion the User accepted (#282). It holds no Block ids: the words went into
  // the paragraph being written, so there is nothing to follow, only the use to record.
  events.on("suggestion.accepted", (accepted) => {
    aiUse.record([
      {
        mindId: accepted.mindId,
        kind: "suggestion-accepted",
        providerId: accepted.providerId,
        modelId: accepted.modelId,
        characters: accepted.characters,
        occurredAt: accepted.acceptedAt,
      },
    ]);
  });
  // The level Skill scripts run at: the given Executor's, else the local one's below ("none").
  const scriptSandbox = adapters.executor?.level ?? "none";
  const capture = adapters.audioCapture;
  const settings = createSettings(
    db,
    now,
    adapters.systemLanguages,
    scriptSandbox,
    !capture ? "none" : capture.systemAudio ? "both" : "microphone",
  );
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
  const batches = createBatches({
    db,
    emit: (batch) => events.emit("documents.batch", batch),
    file: (documentId, groupId) => library.file(documentId, groupId),
  });
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
  // Reads scanned pages (#397), when this copy has a recognizer to read them with.
  const textRecognitionModel = adapters.textRecognizer
    ? createTextRecognitionModel({
        definition: BUILT_IN_TEXT_RECOGNITION_MODEL,
        source: adapters.textRecognitionModelSource,
        dataDir,
        recognizer: adapters.textRecognizer,
        emitStatus: (status) => events.emit("textRecognitionModel.status", status),
      })
    : undefined;
  /** What `getTextRecognitionModel` says in a copy that has no recognizer: nothing to download. */
  const textRecognitionUnavailable: EmbeddingModelStatus = {
    name: BUILT_IN_TEXT_RECOGNITION_MODEL.name,
    host: new URL(BUILT_IN_TEXT_RECOGNITION_MODEL.source.baseUrl).host,
    state: "not-downloaded",
    downloadedBytes: 0,
    totalBytes: 0,
    error: null,
  };
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
  const retention = createVersionRetention(db, now);
  let documents: ReturnType<typeof createDocuments>;
  /** Set once created, so a failure after it can close it. */
  let createdDocuments: ReturnType<typeof createDocuments> | undefined;
  try {
    documents = createdDocuments = createDocuments({
      db,
      dataDir,
      now,
      model: embedding.model,
      textRecognition: textRecognitionModel,
      folders,
      emitStatus: (document) => {
        events.emit("document.status", document);
        libraryDocumentChanged(document.id);
        batches.changed([document.id]);
        rebuildMayHaveProgressed();
      },
      emitMoved: (moved) => events.emit("documents.moved", moved),
      emitRemoved: (ids) => {
        events.emit("documents.removed", ids);
        batches.changed(ids);
      },
      emitKept: (kept) => events.emit("keptCitationTexts.changed", kept),
      foldersChanged,
      linkedFoldersChanged: (list) => {
        events.emit("linkedFolders.changed", list);
        batches.linkedFoldersChanged(list);
      },
      onFound: (documentId) => batches.found(documentId),
      onReady: (documentId) => {
        // There is something to search now: the built-in reranking model downloads, if it reranks.
        prepareRerank();
        documentReady(documentId);
      },
      // Citations live in Minds, anywhere: in Answers, or copied into Notes; and in
      // conversations. Features other than Citations retain what they need too.
      citedUnits: (documentIds) => {
        const wanted = new Set(documentIds);
        const cited: CitedUnits[] = [];
        for (const mind of minds.list()) {
          content.peek(mind.id, (fragment) => addCitedUnits(fragment, wanted, cited));
        }
        conversations.addCitedUnits(wanted, cited);
        retention.addRetainedUnits(wanted, cited);
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
      say: (key, params) => translate(settings.get().language, key, params),
    });
    // Old versions' text, and the text kept of unlinked Documents, that no
    // Citation quotes and no feature retains any more goes, at startup only:
    // nothing is being written or undone then, so no Citation is in flight.
    // The retained versions are one query, whatever their kind.
    if (documents.hasOldVersions() || documents.keptCitationTexts().length > 0) {
      const cited: CitedVersions = new Map();
      for (const mind of minds.list()) {
        content.peek(mind.id, (fragment) => addCitedVersions(fragment, cited));
      }
      conversations.addCitedVersions(cited);
      retention.addRetainedVersions(cited);
      const quoted = (documentId: string, contentHash: string) =>
        isCited(cited, documentId, contentHash);
      documents.collectOldVersions(quoted);
      documents.releaseKeptText(quoted);
    }
    retention.forgetReleased();
  } catch (error) {
    lifetime.abort();
    // Created, but collecting old versions failed.
    createdDocuments?.close();
    embeddingModel.close();
    db.close();
    throw error;
  }
  const webPages = createWebPages({
    db,
    dataDir,
    now,
    documents,
    languages: () =>
      settings.get().language === "zh-CN" ? ["zh-CN", "zh;q=0.9", "en;q=0.8"] : ["en", "*;q=0.5"],
    ...adapters.webPages,
  });
  webPages.dropPurged().catch((error: unknown) => console.error(error));
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
    log: adapters.log ?? silentLogger,
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
  /** Set once outside agents exist: Local mode pauses them. */
  let agentsLocalModeChanged = () => {};
  /**
   * Local mode on or off: embeddings may switch back to the built-in model,
   * rerank pauses, usage data stops, and outside agents pause.
   */
  const setLocalOnly = async (enabled: unknown) => {
    const wasLocal = embedding.localOnly();
    await embedding.setLocalOnly(enabled);
    library.resume();
    if (wasLocal !== embedding.localOnly()) {
      usageData.localModeChanged();
      privacyChanged();
      agentsLocalModeChanged();
      await rerankChanged();
      // Questions to a cloud model are refused now (or allowed again): the chat model's readiness changed.
      await readinessChanged();
      // An Answer on its way to a cloud model stops at once, with the reason.
      if (embedding.localOnly()) await answers.stopCloud();
    }
    return embeddingSettings();
  };

  // Sign-in changes can happen in the middle of a request (a refresh that fails), so they report through events.
  let chatGptChanged = () => {};
  const chatGptStore = adapters.chatGptStore;
  const chatGpt = createChatGptPlan({
    endpoints: { ...CHATGPT_PLAN_ENDPOINTS, ...adapters.chatGptPlan },
    secrets,
    ...(chatGptStore && {
      sharedStore: {
        secrets: createSecrets(chatGptStore.keychain, settings),
        refreshLock: createFileLock(chatGptStore.lockFile),
      },
    }),
    settings,
    browser: adapters.browser,
    now: () => (adapters.now?.() ?? new Date()).getTime(),
    onChange: () => chatGptChanged(),
  });
  // What the experimental ChatGPT provider left on this device goes (see migration 53).
  chatGpt.forgetCodexSignIn().catch((error: unknown) => {
    console.error("The experimental ChatGPT provider's sign-in couldn't be removed.", error);
  });
  const meter = createMeter({
    db,
    now: () => adapters.now?.() ?? new Date(),
    cap: () => settings.get().user.monthlyCap,
  });
  // Nothing is in flight yet: what is still reserved was left by a crash.
  meter.settleOpen();
  const chat = createChat({
    db,
    now,
    settings,
    secrets,
    consent,
    createModel: adapters.createChatModel ?? createAiSdkChatModel,
    chatGpt,
    ollamaModels: adapters.ollamaModels ?? createOllamaModels(),
    localOnly: () => embedding.localOnly(),
    meter,
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

  // Requests to one service kept apart, as its operator asks: Look up and Find sources share them.
  const paceTimers = adapters.paceTimers ?? realTimers;
  const pace = createPace(paceTimers);
  // Find sources (#296): searches Source providers with consent, never in local mode, and adds
  // what the User chooses to the Library, as Documents (open copies) or reference records.
  const sources = createSources({
    db,
    now,
    clock: () => paceTimers.now(),
    consent,
    localOnly: () => embedding.localOnly(),
    secrets,
    providers: createSourceProviders(adapters.sourceProviders, (id) =>
      secrets.tryGet(sourceKeySecret(id)),
    ),
    http: createSourceHttp(pace, paceTimers),
    folder: join(dataDir, "Sources"),
    checkFolder: (folderId) => {
      library.target({ groupId: folderId });
    },
    async addFile(path, folderId, added) {
      const batchId = batches.begin(folderId);
      try {
        const result = await documents.add([path], (documentId) => {
          batches.added(batchId, documentId);
          added(documentId);
        });
        return result.documents[0] ?? null;
      } finally {
        batches.done(batchId);
      }
    },
    async savePage(address, folderId, added) {
      const result = await webPages.save({ address }, added);
      if (result.outcome === "failed") return null;
      if (!result.existing && folderId !== null) library.assign(result.document.id, folderId);
      return { document: documents.get(result.document.id), existing: result.existing };
    },
    document: (id) => documents.get(id),
    signal: lifetime.signal,
    reportError: (error) => console.error(error),
  });
  // A provider is asked only once the User searches it, as Look up's services are: the Privacy
  // page lists each one the User has decided on, to revoke.
  consent.registry.register({
    id: "find-sources",
    sends: FIND_SOURCES_SENDS,
    services: async () => [],
  });
  sources.dropPurged().catch((error: unknown) => console.error(error));
  /** The User's Documents searched as an Answer searches them: the search Tool, reranked as the User chose. */
  const searchDocuments = (
    query: string,
    documentIds: readonly string[] | null,
    signal?: AbortSignal,
  ) => {
    const options = {
      signal,
      rerank: adapters.reranker ?? rerank.active(),
      documentIds: documentIds ?? undefined,
    };
    const tuning = adapters.answerSearch;
    return tuning
      ? tunedSearch(tuning, query, options, (tuned) => documents.searchTool(query, tuned))
      : documents.searchTool(query, options);
  };
  const resolveScope = (scope: SearchScope) =>
    resolveSearchScope(scope, {
      folderTree: (folderId) => folders.subtree(folderId),
      organizedFolderDocuments: (folderId) => library.documentIds(folderId),
      documentIds: (filter) => documents.ids(filter),
    });
  // What checks and comparison tables search (#80): as Answers do, without IncarnaMind's own
  // files and the document being checked. Answers themselves search everything, as before.
  const evidence = createEvidence({
    db,
    leaveOutOwnFiles: () => settings.get().user.leaveOutOwnFiles,
    search: (query, options) =>
      documents.searchTool(query, { ...options, rerank: adapters.reranker ?? rerank.active() }),
  });

  // Web search through the Answer's model's provider (#128): asked for once per provider, as
  // Find sources' are, so the Privacy page lists each one the User has decided on.
  consent.registry.register({
    id: "web-search",
    sends: WEB_SEARCH_SENDS,
    services: async () => [],
  });
  const webSearch = createWebSearch({
    enabled: () => settings.get().user.webSearch && !embedding.localOnly(),
    readiness: (model) => chat.readiness(model),
    prepare: (model) => chat.prepareModel(model, "web-search", "off"),
    savePage: (address, fetch) => webPages.save({ address }, undefined, fetch),
    folderOf: (place) => {
      try {
        if (place.mindId !== null) return minds.get(place.mindId).folderId;
        return place.conversationId ? conversations.get(place.conversationId).folderId : null;
      } catch (error) {
        // Deleted while the Answer was written: the page stays in no Folder.
        if (error instanceof NotFoundError) return null;
        throw error;
      }
    },
    file: (documentId, folderId) => library.assign(documentId, folderId),
    document: (id) => documents.get(id),
    onStatus: (listener) => events.on("document.status", listener),
    ...(adapters.web && { fetch: adapters.web }),
  });

  const answers = createAnswers({
    content,
    events,
    engine: adapters.answerEngine ?? createAiSdkAnswerEngine({ runEngine: adapters.runEngine }),
    requireMind: (mindId) => writableMind(mindId).id,
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
      search: searchDocuments,
      citationSource: (passageId) => documents.citationSource(passageId),
      pageTexts: (documentId, contentHash, from, to) =>
        documents.pageTexts(documentId, contentHash, from, to),
    },
    toolProviders: [
      { kind: "connector", tools: (signal) => connectors.tools(signal) },
      { kind: "sources", tools: async () => sources.tools() },
    ],
    connectorsNeedingSignIn: () => connectors.needingSignIn(),
    web: webSearch,
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
    // Every window hears of each message as it is added or written.
    conversations: {
      add: (input, answerModel) => {
        const turn = conversations.add(input, answerModel);
        events.emit("conversation.message", turn.question);
        events.emit("conversation.message", turn.answer);
        conversationsChanged();
        return turn;
      },
      questionOf: (input) => conversations.questionOf(input),
      restartAnswer: (answerId, answerModel) => {
        const turn = conversations.restartAnswer(answerId, answerModel);
        events.emit("conversation.message", turn.answer);
        return turn;
      },
      writeAnswer: (answerId, write) => {
        const answer = conversations.writeAnswer(answerId, write);
        if (answer) events.emit("conversation.message", answer);
        return answer !== null;
      },
    },
    resolveScope,
    emptyScopeAnswer: () => translate(settings.get().language, "scope.answer.empty"),
    onWritingChange: (writing) => background.setAnswering(writing),
    reportError: (error) => console.error(error),
  });

  // Suggest as I write (#282): Passages found and quotes checked as Answers do (see ./answers/citing).
  const citing = createCiting({
    search: searchDocuments,
    citationSource: (passageId) => documents.citationSource(passageId),
    pageTexts: (documentId, contentHash, from, to) =>
      documents.pageTexts(documentId, contentHash, from, to),
  });
  const suggestions = createSuggestions({
    enabled: () => settings.get().user.suggestWhileWriting,
    model: () => settings.get().user.chatModel,
    chat,
    consent,
    citing,
    requireMind: (mindId) => {
      writableMind(mindId);
    },
    resolveScope,
    searchableCount: (documentIds) => documents.searchableCount(documentIds ?? undefined),
    localOnly: () => embedding.localOnly(),
    events,
    readValue: (key) => settings.readDeviceValue(key),
    writeValue: (key, value) => settings.writeDeviceValue(key, value),
    now: () => adapters.now?.() ?? new Date(),
    reportError: (error) => console.error(error),
  });

  // Citation styles (#289): formatted on their own thread, from the Documents' reference records.
  const citationStyles = createCitationStyles({
    folder: adapters.paths.citationStyles,
    document: (id): Document | undefined => {
      try {
        return documents.get(id);
      } catch (error) {
        if (error instanceof NotFoundError) return undefined;
        throw error;
      }
    },
  });
  const mindCitationStyle = (mindId: string): MindCitationStyle => {
    const { folderId } = minds.get(mindId);
    return content.readSettings(mindId, (mindSettings) => ({
      style: citationStyleIn(mindSettings),
      chosen: chosenCitationStyle(mindSettings) !== null,
      folderDefault: folderId === null ? null : library.citationStyle(folderId),
    }));
  };
  // Edit with AI (#93): the change comes back as Proposals; Citations it keeps are checked again.
  const edits = createEdits({
    content,
    requireMind: (mindId) => {
      writableMind(mindId);
    },
    defaultModel: () => settings.get().user.chatModel,
    chat,
    consent,
    localOnly: () => embedding.localOnly(),
    recheck: (attributes) => recheckKept(documents, attributes),
    now: () => adapters.now?.() ?? new Date(),
  });

  // "Add to note" into a Mind that isn't open (#201), with its Citations checked where they land.
  const noteAdding = createNoteAdding({
    content,
    requireMind: (mindId) => writableMind(mindId).id,
    recheck: (input) => recheckCitation(documents, input),
    recordAiUse: (inputs) => void aiUse.record(inputs),
    now: () => adapters.now?.() ?? new Date(),
  });

  // The fingerprints of the files IncarnaMind writes (#80). A Document already indexed with
  // the same bytes is marked at once, and every window hears of it.
  const ownFiles = createOwnFiles(db, now);
  const recordOwnFile = (data: Uint8Array, kind: OwnFileKind) => {
    const marked = ownFiles.record(data, kind);
    for (const document of documents.getMany(marked)) events.emit("document.status", document);
  };
  const mindExports = createExports({
    mind: (mindId) => minds.get(mindId),
    read: (mindId, look) => content.read(mindId, look),
    liveDocuments: () => documents.list(),
    keptCitationTexts: () => documents.keptCitationTexts(),
    language: () => settings.get().language,
    citationStyle: (mindId) => mindCitationStyle(mindId).style,
    bibliography: (mindId) => content.readSettings(mindId, bibliographyShown),
    formatCitations: (request) => citationStyles.format(request),
    printer: adapters.printer ?? null,
    paper: () => paperFor(systemRegion(adapters.systemRegion?.(), adapters.systemLanguages())),
    ...(adapters.pictures && { pictures: adapters.pictures }),
    log: adapters.log ?? silentLogger,
    now,
  });
  /** Usage data: which format, and whether Questions went with it. */
  const recordExport = (options: ExportMindOptions) =>
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
  // Make this a Skill: the chat model drafts it from the Mind, on the chat flow's consent.
  const skillDrafts = createSkillDrafts({
    mindMarkdown: async (mindId) => {
      const mind = new TextDecoder().decode(
        (await mindExports.export(mindId, { format: "markdown", includeQuestions: true })).data,
      );
      // Questions are asked in the assistant (#201): those asked from this Mind are its work too.
      const label = translate(settings.get().language, "export.question");
      const asked = conversations
        .askedFromMind(mindId)
        .map(({ question, answer }) => [`**${label}** ${question}`, answer].filter(Boolean));
      return [mind, ...asked.flat()].join("\n\n");
    },
    modelFor: (mindId) =>
      content.read(mindId, (blocks) => (blocks.doc ? mindModelOf(blocks.doc) : null)) ??
      settings.get().user.chatModel,
    chat,
  });
  // New files: in the Output location, or inside a Linked folder the User allowed.
  const outputs = createOutputFiles({
    db,
    events,
    now,
    settings,
    suggested: join(adapters.paths.documents ?? join(homedir(), "Documents"), "IncarnaMind"),
    linkedFolders: () => documents.linkedFolders.list(),
    recordOwnFile,
  });
  /** Saves an export in the Output location, as the window's save: what went wrong is a result, not an error. */
  const saveExport = async (
    name: string,
    data: Uint8Array,
    after: () => void = () => undefined,
  ): Promise<SavedExport> => {
    try {
      const file = await outputs.save({ name, data, ownFile: "export" });
      after();
      return { status: "saved", file };
    } catch (error) {
      if (error instanceof FileWriteCancelledError) return { status: "cancelled" };
      if (error instanceof LinkedFolderWritesDeclinedError) {
        return {
          status: "not-allowed",
          linkedFolderId: error.linkedFolderId,
          linkedFolderPath: error.linkedFolderPath,
        };
      }
      throw error;
    }
  };
  const unlinkFolder = (linkedFolderId: string) => {
    documents.linkedFolders.remove(linkedFolderId);
    outputs.forgetLinkedFolder(linkedFolderId);
  };

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
    changed: () => {
      events.emit("library.changed", null);
      batches.libraryChanged();
    },
    assigned: (assignments) => {
      events.emit("library.assignments", assignments);
      batches.changed(assignments.map((each) => each.documentId));
    },
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
      const images = readsImages(prepared.capabilities);
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

  // Reference records (#286): read from files as they're processed, edited by the User, and
  // filled by Look up, which sends a DOI or an arXiv id only, with consent, never in local mode.
  const references = createReferences({
    db,
    now,
    consent,
    localOnly: () => embedding.localOnly(),
    urls: { ...REFERENCE_SERVICE_URLS, ...adapters.referenceLookup },
    document: (id) => documents.get(id),
    changed: (list) => events.emit("references.changed", list),
    signal: lifetime.signal,
    pace,
  });
  // Importing references (#287) reads a file the User chose and sends nothing.
  const referenceImport = createReferenceImport({
    db,
    now,
    documents: () => documents.list({}),
    document: (id) => documents.get(id),
    addDocuments: (paths) => addDocuments(paths),
    changed: (list) => events.emit("references.changed", list),
    progress: (done, total) => events.emit("references.importProgress", { done, total }),
  });
  // Look up goes to Crossref or arXiv only when the User asks, so the flow goes nowhere until
  // then; the Privacy page lists each service once the User has decided on it.
  consent.registry.register({
    id: "reference-lookup",
    sends: REFERENCE_LOOKUP_SENDS,
    services: async () => [],
  });

  const recentlyDeleted = createRecentlyDeleted({
    db,
    now,
    // A Mind still waiting in Recently deleted comes back with its Citations, so they count too.
    documentText: (documentIds) => {
      const wanted = new Set(documentIds);
      const cited: CitedUnits[] = [];
      for (const { id } of db.all<{ id: string }>("SELECT id FROM minds WHERE purged_at IS NULL")) {
        content.peek(id, (fragment) => addCitedUnits(fragment, wanted, cited));
      }
      conversations.addCitedUnits(wanted, cited);
      retention.addRetainedUnits(wanted, cited);
      documents.purgeText(documentIds, cited);
    },
    say: (key, params) => translate(settings.get().language, key, params),
    minds,
    content,
    conversations,
    documents,
    tags,
    library,
    mindsChanged,
    conversationsChanged,
    tagsChanged,
    tagged: announceTagged,
  });
  try {
    recentlyDeleted.expire();
  } catch (error) {
    // The app still opens; what is due goes at the next start.
    console.error("Deleting what waited 30 days in Recently deleted failed.", error);
  }

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
  const textRecognitionSource =
    adapters.textRecognitionModelSource ?? BUILT_IN_TEXT_RECOGNITION_MODEL.source;
  const textRecognitionHost = new URL(textRecognitionSource.baseUrl);
  privacy.traffic.register({
    id: "text-recognition-model",
    service: {
      id: textRecognitionHost.origin,
      name:
        textRecognitionHost.host === "huggingface.co" ? "Hugging Face" : textRecognitionHost.host,
    },
    // Only a copy that reads scans downloads it.
    listed: () => textRecognitionModel !== undefined && textRecognitionSource.files.length > 0,
  });
  const transcriptionSource = adapters.transcriptionModelSource?.model ?? QWEN3_ASR_0_6B.source;
  const transcriptionHost = new URL(transcriptionSource.baseUrl);
  privacy.traffic.register({
    id: "transcription-model",
    service: {
      id: transcriptionHost.origin,
      name: transcriptionHost.host === "huggingface.co" ? "Hugging Face" : transcriptionHost.host,
    },
    // Where this copy transcribes, once Meetings are on.
    listed: () =>
      transcriber !== undefined &&
      transcriptionSource.files.length > 0 &&
      settings.get().device.meetingsEnabled,
  });
  privacy.traffic.register({ id: "ollama-pull", service: OLLAMA_REGISTRY });
  const chatGptSignIn = new URL(
    adapters.chatGptPlan?.authorizeUrl ?? CHATGPT_PLAN_ENDPOINTS.authorizeUrl,
  );
  privacy.traffic.register({
    id: "chatgpt-sign-in",
    service: { id: chatGptSignIn.origin, name: "OpenAI" },
    listed: () => chatGpt.used(),
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
  const layouts = createLayouts({
    readValue: (key) => settings.readDeviceValue(key),
    writeValue: (key, value) => settings.writeDeviceValue(key, value),
    mindExists: (mindId) => {
      try {
        minds.get(mindId);
        return true;
      } catch {
        return false;
      }
    },
    documentExists: (documentId) => documents.getMany([documentId]).length > 0,
    changed: (windowId, layout) => events.emit("layout.changed", { windowId, layout }),
  });
  // The example Mind: its Citations are checked once its Documents have been read.
  const examples = createExamples({
    source: adapters.paths.examples,
    dataDir,
    language: () => settings.get().language,
    readValue: (key) => settings.readDeviceValue(key),
    writeValue: (key, value) => settings.writeDeviceValue(key, value),
    createMind: (title) => {
      const mind = minds.create(parseCreateMindInput({ title }));
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
      layouts.forgetDeletedMinds();
    },
    editMind: (mindId, change) => content.edit(mindId, change),
    linkFolder: (path) => documents.linkedFolders.add(path),
    unlinkFolder,
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

  // Transcripts (#136): a Meeting is transcribed on this computer as it records, where this copy can.
  let listMeetings: (() => Meeting[]) | undefined;
  const meetingsChanged = () => {
    if (listMeetings) events.emit("meetings.changed", listMeetings());
  };
  const captureHealth = createCaptureHealth({
    dataDir,
    now,
    log: adapters.log ?? silentLogger,
    changed: meetingsChanged,
  });
  const transcriber = adapters.transcriber;
  const transcripts =
    transcriber &&
    createTranscripts({
      dataDir,
      transcriber,
      model: createTranscriptionModel({
        dataDir,
        transcriber,
        ...(adapters.transcriptionModelSource && { source: adapters.transcriptionModelSource }),
        speed: () => {
          const value = settings.readDeviceValue(TRANSCRIPTION_SPEED);
          return isRecord(value) ? (value as Record<string, number>) : {};
        },
        emitStatus: () => meetingsChanged(),
      }),
      documents: {
        add: (id, name, file) => documents.addSnapshot(id, name, file, () => {}, "transcript"),
        file: (documentId, folderId) => library.assign(documentId, folderId),
        check: (path) => documents.check(path),
        moveFile: (id, path) => documents.moveFile(id, path),
        find: (id) => {
          try {
            return documents.get(id);
          } catch {
            return null;
          }
        },
      },
      saveFile: (file) => outputs.save(file),
      name: (meetingId) => {
        const meeting = listMeetings?.().find((each) => each.id === meetingId);
        const mind = meeting ? minds.get(meeting.mindId) : null;
        const title = mind ? mind.title || mind.autoTitle : "";
        const language = settings.get().language;
        return title
          ? translate(language, "meetings.transcript.name", { meeting: title })
          : translate(language, "meetings.transcript.nameAlone");
      },
      folderOf: (meetingId) => {
        const meeting = listMeetings?.().find((each) => each.id === meetingId);
        return meeting ? minds.get(meeting.mindId).folderId : null;
      },
      partsOf: (meetingId) => listParts(meetingFolder(dataDir, meetingId)),
      recordSpeed: (modelId, ratio) => {
        const value = settings.readDeviceValue(TRANSCRIPTION_SPEED);
        settings.writeDeviceValue(TRANSCRIPTION_SPEED, {
          ...(isRecord(value) ? value : {}),
          [modelId]: ratio,
        });
      },
      health: captureHealth,
      changed: meetingsChanged,
      log: adapters.log ?? silentLogger,
    });

  // Meetings (#135): while audio is being captured, background model work holds back.
  const meetings = createMeetings({
    db,
    dataDir,
    now,
    capture: adapters.audioCapture,
    enabled: () => settings.get().device.meetingsEnabled,
    noticeAccepted: () => settings.get().device.meetingNoticeAccepted,
    systemAudioByDefault: () => settings.get().device.meetingSystemAudio,
    language: () => settings.get().language,
    createMind: (title) => {
      const mind = minds.create(parseCreateMindInput({ title }));
      mindsChanged();
      return mind;
    },
    mindExists: (mindId) => {
      try {
        minds.get(mindId);
        return true;
      } catch {
        return false;
      }
    },
    changed: (list) => events.emit("meetings.changed", list),
    onCapturingChange: (capturing) => {
      background.setRecording(capturing);
      documents.hold(capturing);
    },
    ...(transcripts && { transcripts }),
    health: captureHealth,
    log: adapters.log ?? silentLogger,
  });
  listMeetings = () => meetings.list();

  // Outside agents (#351): the User's own agent, paired in Settings, reads its Folders over the
  // socket and gets Citations IncarnaMind minted and checked. Off until the User turns it on.
  // Where agents keep their settings and skills.
  const home = adapters.agents?.home ?? homedir();
  const agents = createAgents({
    db,
    now,
    clock: () => performance.now(),
    secrets,
    consent,
    readValue: (key) => settings.readDeviceValue(key),
    writeValue: (key, value) => settings.writeDeviceValue(key, value),
    localOnly: () => embedding.localOnly(),
    socketPath: adapters.agents?.socketPath ?? agentSocketPath(dataDir),
    // Without a host's launcher there is no shim to start; tests and the evaluation give theirs.
    launcher: adapters.agents?.launcher ?? { command: process.execPath, args: [] },
    version: adapters.agents?.version ?? "0.0.0",
    access: {
      say: (key, params) => translate(settings.get().language, key, params),
      folders: () => library.folders().map((folder) => ({ id: folder.id, name: folder.name })),
      documentsIn: (folderIds) =>
        folderIds.length === 0
          ? []
          : (resolveScope({ folderIds: [...folderIds], tagIds: [], documentIds: [] }) ?? []),
      // Without the reranker: it costs about a second a search (ADR-0009), and an agent's
      // search must answer in well under 100 ms. `ask` reranks, as every Answer does.
      search: (query, documentIds, parameters, signal) =>
        documents.searchTool(query, { documentIds, parameters, signal }),
      passage: (passageId) => documents.passage(passageId),
      passagesBetween: (documentId, from, to) => documents.passagesBetween(documentId, from, to),
      citing,
      recheck: (input) => recheckCitation(documents, input),
      minds: () => minds.list().filter((mind) => isKnownKind(mind.kind)),
      mindMarkdown: (mindId, footnote) => {
        const mind = minds.get(mindId);
        const { language } = settings.get();
        const blocks = content.peek(mind.id, (fragment) =>
          readBlocks(fragment, { includeQuestions: true, footnote, trimBeforeCitation: true }),
        );
        return renderMarkdown(mind.title || mind.autoTitle, blocks, exportLabels(language));
      },
      answer: (question, documentIds, signal) =>
        answers.answerOutside(question, documentIds, signal),
      show: (place) => events.emit("agents.citationShown", place),
    },
    skillInstalled: (client) => skillInstalled(home, client),
    changed: (value) => events.emit("agents.changed", value),
    called: (activity) => events.emit("agents.activity", activity),
    reportError: (error) => console.error(error),
  });
  agentsLocalModeChanged = () => agents.localModeChanged();
  agents.start().catch((error: unknown) => console.error(error));
  // Setting IncarnaMind up in the User's agent, from Settings → Agents.
  const agentSetup = createAgentSetup({
    agents,
    home,
    applications:
      adapters.agents?.applications ??
      (platform === "darwin" ? ["/Applications", join(home, "Applications")] : []),
    platform,
    tempDir: adapters.paths.tempDir ?? tmpdir(),
    shimScript: adapters.agents?.shimScript ?? null,
    version: adapters.agents?.version ?? "0.0.0",
    openLink: adapters.agents?.openLink ?? null,
    openPath: adapters.shell ? (path) => adapters.shell?.openPath(path) ?? Promise.resolve() : null,
    say: (key, params) => translate(settings.get().language, key, params),
    reportError: (error) => console.error(error),
  });

  const addDocuments: CoreApi["addDocuments"] = async (paths, options) => {
    const batchId = batches.begin(library.target(options));
    const since = now();
    const result = await documents
      .add(paths, (documentId) => batches.added(batchId, documentId))
      .finally(() => batches.done(batchId));
    usageData.record({ event: "documents_added", fields: addedCounts(result, since) });
    return { ...result, batchId };
  };

  // Async on purpose: the renderer reaches these over IPC, and a future hosted core may be remote.
  return {
    createMind: async (input) => {
      const parsed = parseCreateMindInput(input);
      if (parsed.folderId !== null) library.folder(parsed.folderId);
      const mind = minds.create(parsed);
      // It starts in the style last chosen for a Mind in its Folder, as its own choice (#289).
      const style = parsed.folderId === null ? null : library.citationStyle(parsed.folderId);
      if (style !== null) {
        content.editSettings(mind.id, (mindSettings) => setCitationStyle(mindSettings, style));
      }
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
      const deleted = db.transaction(() => {
        const mind = minds.delete(mindId, at);
        content.remove(mind.id, at);
        return mind;
      });
      meetings.mindDeleted(deleted.id);
      mindsChanged();
      layouts.forgetDeletedMinds();
    },
    openMind: async (mindId) => {
      const mind = minds.get(mindId);
      // A Mind this version may only read is shown from what is stored: nothing is
      // settled, compacted or written, so its Yjs state stays exactly as it was.
      if (!isKnownKind(mind.kind)) {
        return { mind, access: "update-required", state: Y.encodeStateAsUpdate(new Y.Doc()) };
      }
      const access = accessFor(mind.kind, content.contentVersion(mind.id));
      if (access === "read-only") return { mind, access, state: content.storedState(mind.id) };
      answers.settleOrphans(mind.id);
      return { mind, access, state: content.state(mind.id) };
    },
    applyMindUpdate: async (mindId, update) => {
      content.apply(writableMind(mindId).id, update);
    },
    closeMind: async (mindId) => {
      content.close(parseMindId(mindId));
    },
    askInConversation: async (input) =>
      answers.askInConversation(conversations.parseQuestion(input)),
    regenerateConversationAnswer: async (input) => answers.regenerateInConversation(input),
    stopConversationAnswer: async (input) => answers.stopInConversation(input),
    listConversations: async (options) => conversations.list(options),
    getConversation: async (conversationId) => conversations.content(conversationId),
    searchConversations: async (query) => conversations.search(query),
    renameConversation: async (conversationId, title) => {
      const conversation = conversations.rename(conversationId, title);
      conversationsChanged();
      return conversation;
    },
    deleteConversation: async (conversationId) => {
      // What an Answer being written has is kept, before the conversation goes.
      answers.stopConversation(conversations.get(conversationId).id);
      conversations.delete(conversationId, now());
      conversationsChanged();
    },
    resolveReference: async (input) => {
      const reference = parseReference(input);
      if (!reference) throw new InvalidInputError("That isn't a reference IncarnaMind can follow.");
      const found =
        reference.artifactKind === "mind"
          ? minds.locate(reference.artifactId)
          : conversations.locate(reference.artifactId, reference.anchor.messageId);
      if (!found) return { status: "gone", reference };
      return found.deletedAt === null
        ? { status: "available", reference, name: found.name }
        : { status: "deleted", reference, name: found.name, deletedAt: found.deletedAt };
    },
    addToNote: (input) => noteAdding.add(input),
    undoAddToNote: async (input) => noteAdding.undo(input),
    getSettings: async () => settings.get(),
    getLayout: async (windowId) => layouts.get(windowId),
    changeLayout: async (windowId, change) => layouts.change(windowId, change),
    updateSettings: async (patch) => {
      // The default chat model must be on a saved provider.
      const chatModel = isRecord(patch) && isRecord(patch.user) ? patch.user.chatModel : undefined;
      if (isChatModelChoice(chatModel) && !chat.exists(chatModel.providerId)) {
        throw new InvalidInputError("That chat provider doesn't exist.");
      }
      const scriptsWere = settings.get().device.skillScriptsEnabled;
      let change = patch;
      // Suggestions turned on: the User allows what they send first. Declined, they stay as they were.
      if (isRecord(patch) && isRecord(patch.user) && patch.user.suggestWhileWriting === true) {
        if (!(await suggestions.allow())) {
          const { suggestWhileWriting: _declined, ...user } = patch.user;
          change = { ...patch, user };
        }
      }
      const updated = settings.update(change);
      if (!updated.user.suggestWhileWriting) suggestions.stop();
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
    addDocuments,
    saveWebPage: async (input) => {
      const folderId = isRecord(input) ? input.folderId : undefined;
      if (
        folderId !== undefined &&
        folderId !== null &&
        !library.snapshot().groups.some((group) => group.id === folderId)
      ) {
        throw new NotFoundError("There is no such Folder.");
      }
      const since = now();
      const result = await webPages.save(input);
      if (result.outcome === "saved" && !result.existing && typeof folderId === "string") {
        library.assign(result.document.id, folderId);
      }
      const added: Omit<AddDocumentsResult, "batchId"> =
        result.outcome === "saved"
          ? { documents: [result.document], skipped: [] }
          : { documents: [], skipped: [{ path: result.address, reason: "unreadable" }] };
      usageData.record({ event: "documents_added", fields: addedCounts(added, since) });
      return result.outcome === "saved"
        ? { ...result, document: documents.get(result.document.id) }
        : result;
    },
    fetchWebPageAgain: async (documentId) => webPages.fetchAgain(documentId),
    saveTextAsDocument: async (input) => {
      if (!isRecord(input) || typeof input.text !== "string" || input.text.trim() === "") {
        throw new InvalidInputError("There is no text to save.");
      }
      const title = typeof input.name === "string" ? input.name : "";
      const folder = join(dataDir, "Saved text");
      mkdirSync(folder, { recursive: true });
      const text = input.text.endsWith("\n") ? input.text : `${input.text}\n`;
      const name = await writeNewFile(
        folder,
        `${safeFileName(title, "Pasted text")}.md`,
        new TextEncoder().encode(text),
      );
      const { documents: added } = await documents.add([join(folder, name)]);
      const saved = added[0];
      if (!saved) throw new InvalidInputError("The text couldn't be saved.");
      return saved;
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
    readScannedText: async (id) => documents.readScannedText(id),
    readScannedQuote: async (input) => {
      if (!isRecord(input) || typeof input.quote !== "string") {
        throw new InvalidInputError(
          "readScannedQuote needs a Citation's Document, Units and quote.",
        );
      }
      const unit = (value: unknown) =>
        typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : null;
      return documents.readScannedQuote({
        documentId: typeof input.documentId === "string" ? input.documentId : "",
        contentHash: typeof input.contentHash === "string" ? input.contentHash : null,
        pageFrom: unit(input.pageFrom),
        pageTo: unit(input.pageTo),
        quote: input.quote,
      });
    },
    previewLinkedFolder: (path) => documents.linkedFolders.preview(path),
    addLinkedFolder: async (path, options) => {
      const groupId = library.target(options);
      const known = new Set(documents.linkedFolders.list().map((each) => each.id));
      const linked = await documents.linkedFolders.add(path);
      // Its first scan only starts adding Documents once this has returned.
      const batchId = batches.begin(groupId);
      // A folder inside one already linked adds nothing: its scans aren't this batch.
      if (known.has(linked.id)) batches.done(batchId);
      else batches.linking(batchId, linked);
      usageData.record({ event: "folder_linked", fields: {} });
      return { ...linked, batchId };
    },
    listLinkedFolders: async () => documents.linkedFolders.list(),
    locateLinkedFolder: (linkedFolderId, path) =>
      documents.linkedFolders.locate(linkedFolderId, path),
    removeLinkedFolder: async (linkedFolderId) => unlinkFolder(linkedFolderId),
    listKeptCitationTexts: async () => documents.keptCitationTexts(),
    updateReference: async (documentId, patch) => references.update(documentId, patch),
    restoreReference: async (documentId, before, fields) =>
      references.restore(documentId, before, fields),
    lookUpReference: (documentId) => references.lookUp(documentId),
    listSourceProviders: () => sources.list(),
    setSourceProviderKey: (providerId, key) => sources.setKey(providerId, key),
    findSources: (input) => sources.search(input),
    addFoundSource: (input) => sources.add(input),
    importReferences: (path) => referenceImport.run(path),
    undoReferenceImport: async (importId) => {
      referenceImport.undo(importId);
    },
    addImportedReferenceFiles: (importId) => referenceImport.addFiles(importId),
    listUnmatchedReferences: async () => referenceImport.unmatched(),
    addReferenceFile: (recordId, path) => referenceImport.addFile(recordId, path),
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
    searchEvidence: async (query, options) => {
      if (typeof query !== "string") throw new InvalidInputError("A search query must be text.");
      if (options !== undefined && !isRecord(options)) {
        throw new InvalidInputError("searchEvidence's options must be an object.");
      }
      const scope = parseSearchScope(options?.scope);
      const found = await evidence.search(
        query,
        evidence.scope(scope ? resolveScope(scope) : null, parseCheckedDocument(options?.checked)),
      );
      return { passages: found.passages.map(passageResult), excluded: found.excluded };
    },
    setCountedAsEvidence: async (documentId, counted) =>
      documents.setCountedAsEvidence(documentId, counted),
    getEmbeddingModel: async () => embeddingModel.status(),
    downloadEmbeddingModel: async () => embeddingModel.retry(),
    getTextRecognitionModel: async () =>
      textRecognitionModel?.status() ?? textRecognitionUnavailable,
    downloadTextRecognitionModel: async () =>
      textRecognitionModel?.retry() ?? textRecognitionUnavailable,

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
    getCostSummary: async () => meter.summary(),
    estimateRun: async (input) => {
      if (!isRecord(input)) throw new InvalidInputError("estimateRun expects an object.");
      const { items, model } = input;
      if (typeof items !== "number" || !Number.isInteger(items) || items < 0 || items > 100_000) {
        throw new InvalidInputError("items must be a whole number from 0 to 100,000.");
      }
      if (model !== undefined && !isChatModelChoice(model)) {
        throw new InvalidInputError("model must be a provider and a model.");
      }
      const metered = chat.metered(model);
      if (!metered) throw new InvalidInputError("There is no model to estimate for.");
      return meter.estimate(metered, items);
    },

    askQuestion: async (input) => answers.ask(input),
    regenerateAnswer: async (input) => answers.regenerate(input),
    stopAnswer: async (input) => answers.stop(input),
    suggestNext: (request) => suggestions.suggest(request),
    cancelSuggestion: async (requestId) => suggestions.cancel(requestId),
    acceptSuggestion: async (input) => suggestions.accept(input),
    getSuggestionStatus: () => suggestions.status(),
    editWithAi: (input) => edits.editWithAi(input),
    regenerateProposal: (input) => edits.regenerateProposal(input),
    cancelEdit: async (requestId) => edits.cancel(requestId),

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
      // One click means a model on this computer; a cloud model is chosen from the model menu.
      if (isCloudModelName(model) || hasOllamaModel(status.cloudModels, model)) {
        throw new InvalidInputError(
          `${model} runs on ollama.com, not on this computer. Choose a local model, or pick it in the model menu.`,
        );
      }
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
    signInToChatGpt: (input) =>
      chatGpt.signIn({ anotherAccount: isRecord(input) && input.anotherAccount === true }),
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
    recordAiUse: async (inputs) => aiUse.record(inputs),
    listAiUse: async (mindId) => aiUse.list(mindId),
    getAiUseStatus: async () => aiUse.status(),
    clearAiUse: async () => aiUse.clear(),
    restoreAiUse: async (clearedAt) => aiUse.restore(clearedAt),

    listFolders: async () => folders.list(),

    getLibrary: async () => library.snapshot(),
    createLibraryGroup: async (input) => library.create(input),
    updateLibraryGroup: async (id, input) => library.update(id, input),
    deleteLibraryGroup: async (id) => {
      const deletion = db.transaction(() => {
        const folder = library.folder(id);
        const left = minds.leaveFolder(folder.id);
        const leftConversations = conversations.leaveFolder(folder.id);
        const { assignments } = library.delete(folder.id, {
          minds: left.map((mind) => mind.id),
          conversations: leftConversations.map((conversation) => conversation.id),
        });
        const live = (items: { id: string; live: boolean }[]) =>
          items.filter((item) => item.live).map((item) => item.id);
        return { assignments, mindIds: live(left), conversationIds: live(leftConversations) };
      });
      if (deletion.mindIds.length > 0) mindsChanged();
      if (deletion.conversationIds.length > 0) conversationsChanged();
      return deletion;
    },
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
    restoreAssignment: async (id, previous) => library.restoreAssignment(id, previous),
    restore: async (refs) => recentlyDeleted.restore(refs),
    listDeleted: async () => recentlyDeleted.list(),
    deletePermanently: async (refs) => {
      recentlyDeleted.deletePermanently(refs);
      await Promise.all([webPages.dropPurged(), sources.dropPurged()]);
    },
    acceptAssignments: async (ids) => library.accept(ids),
    rejectAssignments: async (ids) => library.reject(ids),
    moveToFolder: async (input) => {
      if (!isRecord(input)) throw new InvalidInputError("moveToFolder expects an object.");
      const folderId = parseFolderChoice(input.folderId);
      const mindIds = parseIds(input.mindIds, "The Minds to move");
      const documentIds = parseIds(input.documentIds, "The Documents to move");
      const conversationIds = parseIds(input.conversationIds, "The conversations to move");
      // All or none: every one is checked before anything moves.
      if (folderId !== null) library.folder(folderId);
      for (const id of mindIds) minds.get(id);
      for (const id of conversationIds) conversations.get(id);
      const move = db.transaction(() => ({
        folderId,
        minds: minds.move(mindIds, folderId),
        documents: library.move(documentIds, folderId),
        conversations: conversations.move(conversationIds, folderId),
      }));
      // Said once it is saved, so a move that rolled back says nothing.
      if (move.minds.some((item) => item.from !== folderId)) mindsChanged();
      if (move.conversations.some((item) => item.from !== folderId)) conversationsChanged();
      library.moved(move.documents.map((item) => item.id));
      return move;
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
    addTagToDocuments: async (documentIds, tagId) =>
      tagMany(documentIds, (id) => tags.addToDocument(id, tagId)),
    removeTagFromDocuments: async (documentIds, tagId) =>
      tagMany(documentIds, (id) => tags.removeFromDocument(id, tagId)),
    revertDocumentTags: async (links) => {
      const changed = recentlyDeleted.revertTags(links);
      if (changed.length > 0) announceTagged(changed);
      return documents.getMany([...new Set(links.map((link) => link.documentId))]);
    },
    mergeTags: async (tagId, intoTagId) => {
      const { documentIds, links } = tags.merge(tagId, intoTagId, now());
      // Tags first, so a listener filtering by the merged Tag hears it's gone before it refreshes.
      tagsChanged();
      announceTagged(documentIds);
      return { tag: tags.get(intoTagId), merged: { kind: "tag", id: tagId, links } };
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
    getSkillDraft: async (skillId) => skills.draftOf(skillId),
    createSkill: async (draft) => {
      const result = await skills.create(draft);
      if (result.ok) skillsChanged();
      return result;
    },
    updateSkill: async (skillId, draft) => {
      const result = await skills.update(skillId, draft);
      if (result.ok) skillsChanged();
      return result;
    },
    draftSkillFromMind: (input) => skillDrafts.draft(input),
    cancelSkillDraft: async (requestId) => skillDrafts.cancel(requestId),
    listRemovedBuiltInSkills: async () => skills.removedBuiltIns(),
    restoreBuiltInSkills: async () => {
      const restored = await skills.restoreBuiltIns();
      if (restored.length > 0) skillsChanged();
      return restored;
    },

    previewMindExport: async (mindId, options) => mindExports.preview(mindId, options),
    exportMind: async (mindId, options) => {
      const exported = await mindExports.export(mindId, options);
      // The host saves these bytes as a file: IncarnaMind's, wherever it goes.
      recordOwnFile(exported.data, "export");
      recordExport(options);
      return exported;
    },
    saveMindExport: async (mindId, options) => {
      const { fileName, data, ...counts } = await mindExports.export(mindId, options);
      const saved = await saveExport(fileName, data, () => recordExport(options));
      return saved.status === "saved" ? { ...saved, ...counts } : saved;
    },
    saveSkillExport: async (skillId) => {
      const { fileName, data } = await skills.exportZip(skillId);
      return saveExport(fileName, data);
    },
    formatCitations: (request) => citationStyles.format(request),
    getMindCitationStyle: async (mindId) => mindCitationStyle(minds.get(mindId).id),
    setMindCitationStyle: async (mindId, style, options) => {
      const notAStyle = (value: unknown) =>
        value !== null && value !== undefined && !isCitationStyleId(value);
      if (style === undefined || notAStyle(style)) {
        throw new InvalidInputError(`"${String(style)}" isn't a citation style.`);
      }
      const folderDefault = options?.folderDefault;
      if (notAStyle(folderDefault)) {
        throw new InvalidInputError(`"${String(folderDefault)}" isn't a citation style.`);
      }
      const mind = writableMind(mindId);
      content.editSettings(mind.id, (mindSettings) => setCitationStyle(mindSettings, style));
      // The style chosen is the one new Minds in its Folder start in, as a document app keeps
      // the last template used; Undo puts back the one before.
      const { folderId } = minds.get(mind.id);
      if (folderId !== null) {
        if (folderDefault !== undefined) library.setCitationStyle(folderId, folderDefault);
        else if (style !== null) library.setCitationStyle(folderId, style);
      }
      return mindCitationStyle(mind.id);
    },
    getOutputSettings: async () => outputs.settings(),
    setOutputLocation: (path) => outputs.setLocation(path),
    setLinkedFolderWriting: async (linkedFolderId, allowed) =>
      outputs.setLinkedFolderWriting(linkedFolderId, allowed),
    listFileWriteRequests: async () => outputs.requests(),
    respondToFileWrite: (requestId, accept) => outputs.respond(requestId, accept),

    listMeetings: async () => meetings.list(),
    createMeeting: async (input) => meetings.create(input),
    discardMeeting: async (meetingId) => meetings.discard(meetingId),
    startMeeting: async (input) => meetings.start(input),
    pauseMeeting: async (meetingId) => meetings.pause(meetingId),
    resumeMeeting: async (meetingId) => meetings.resume(meetingId),
    stopMeeting: async (meetingId) => meetings.stop(meetingId),
    deleteMeetingAudio: async (meetingId) => meetings.deleteAudio(meetingId),
    retryTranscript: async (meetingId) => meetings.retryTranscript(meetingId),
    openAudioSettings: (source) => meetings.openSettings(source),

    openDocumentFile: (documentId) => documents.openFile(documentId),
    openDocumentImage: (documentId, path) => documents.openImage(documentId, path),
    readCellNumber: (documentId, sheet, ref) => documents.cellNumber(documentId, sheet, ref),
    openDocumentInApp: async (documentId) => {
      const address = documents.get(documentId).web?.address;
      if (address) return adapters.browser.open(address);
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
    showSavedFile: async (path) => {
      if (!outputs.saved(path))
        throw new NotFoundError("Only a file IncarnaMind saved can be shown.");
      if (!adapters.shell) throw new Error("This copy of IncarnaMind can't show files.");
      adapters.shell.showItemInFolder(path);
    },
    openSavedFile: async (path) => {
      if (!outputs.saved(path))
        throw new NotFoundError("Only a file IncarnaMind saved can be opened.");
      if (!adapters.shell)
        throw new Error("This copy of IncarnaMind can't open files in other apps.");
      await adapters.shell.openPath(path);
    },
    getAgents: async () => agents.settings(),
    setAgentsEnabled: (enabled) => agents.setEnabled(enabled),
    connectAgent: (input) => agents.connect(input),
    updateAgent: async (pairingId, input) => agents.update(pairingId, input),
    disconnectAgent: async (pairingId) => {
      await agentSetup.removeFromSettings(pairingId);
      await agents.disconnect(pairingId);
    },
    listAgentClients: async () => agentSetup.clients(),
    previewAgentSetup: async (client) => agentSetup.preview(client),
    setUpAgent: (input) => agentSetup.setUp(input),
    undoAgentSetup: (pairingId) => agentSetup.undo(pairingId),
    copyAgentSetup: (pairingId) => agentSetup.settingsFor(pairingId),
    getAgentLaunch: (pairingId) => agents.launch(pairingId),
    listAgentActivity: async (pairingId) => agents.activity(pairingId),
    reconcileDocuments: () => documents.linkedFolders.sync(),
    systemSleeping: () => meetings.sleeping(),
    systemAwake: () => meetings.awake(),
    on: (event, listener) => events.on(event, listener),
    onAnyEvent: (listener) => events.onAny(listener),
    dataFlows: consent.registry,
    networkTraffic: privacy.traffic,
    retention,
    prepareChatModel: (choice) => chat.prepareModel(choice),
    close: () => {
      if (lifetime.signal.aborted) return;
      lifetime.abort();
      // What a Meeting has captured is saved before anything else stops.
      meetings.close();
      transcripts?.close();
      // Outside agents can't reach the core while it closes.
      void agents.close().catch((error: unknown) => console.error(error));
      // First, so stopping the Answers below starts no background work.
      background.close();
      void chatGpt.cancelSignIn();
      // Answers being written keep what they have, marked "stopped"; Tool calls waiting for
      // the User's approval are denied, and Skill scripts running are stopped.
      answers.stopAll();
      void webSearch.close().catch((error: unknown) => console.error(error));
      suggestions.stop();
      edits.stopAll();
      skillDrafts.stopAll();
      scriptRunner.close();
      approvals.close();
      connectors.close();
      tagger.close();
      library.close();
      batches.close();
      skills.close();
      stopTrackingUsage();
      consent.close();
      outputs.close();
      citationStyles.close();
      documents.close();
      embeddingModel.close();
      rerankingModel.close();
      textRecognitionModel?.close();
      activity.stop();
      aiUse.close();
      events.clear();
      content.closeAll();
      db.close();
    },
  };
}
