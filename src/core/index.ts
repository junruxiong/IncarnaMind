/**
 * The IncarnaMind core: all app logic, with no Electron imports (ADR-0004, ADR-0006).
 * Hosts construct it with `createCore`, passing platform capabilities as adapters.
 */
export type {
  Browser,
  CoreAdapters,
  CrashReporter,
  CrossEncoder,
  Embedder,
  EmbeddingModelFiles,
  EmbeddingModelSource,
  ExecAllow,
  ExecFolder,
  ExecRequest,
  ExecResult,
  Executor,
  FileShell,
  Keychain,
  LinkedFolderOptions,
  LogFields,
  Logger,
  LogValue,
  ModelFile,
  Paths,
  ProcessLauncher,
  RerankingModelFiles,
  ScriptRuntimes,
  SpawnOptions,
} from "./adapters";
export * from "./api";
export { type Core, createCore, DATABASE_FILE } from "./core";
export type { DocumentFile, DocumentImage } from "./documents";
export type { Reranker } from "./documents/searchTool";
export type { FolderWatcher, WatchFolder, WatchListener } from "./documents/watcher";
export { BUILT_IN_EMBEDDING_MODEL } from "./embedding";
export {
  ChatNotReadyError,
  ConsentDeclinedError,
  EmbeddingModelNotReadyError,
  InvalidInputError,
  NotFoundError,
  SecretStorageError,
} from "./errors";
export * from "./language";
export type { ChatGptPlanEndpoints } from "./providers/chatgpt/plan";
export type { EmbeddingModelFactory, EmbeddingModelSpec } from "./providers/embeddings";
export {
  type ChatModelFactory,
  type ChatModelSpec,
  type ContextWindow,
  createAiSdkChatModel,
} from "./providers/models";
export { RECOMMENDED_OLLAMA_DOWNLOAD_GB, RECOMMENDED_OLLAMA_MODEL } from "./providers/ollama";
export {
  createOllamaModels,
  type OllamaModelProfile,
  type OllamaModelSettings,
  type OllamaModels,
} from "./providers/ollamaModels";
export type { RerankingModelFactory, RerankingModelSpec } from "./providers/rerank";
export {
  BUILT_IN_RERANKING_MODEL,
  downloadSize,
  RERANKING_MODEL_CANDIDATES,
  type RerankingModelDefinition,
} from "./reranking";
export { BUILT_IN_SKILLS_PACKAGED, BUILT_IN_SKILLS_SOURCE } from "./skills/builtIn";
