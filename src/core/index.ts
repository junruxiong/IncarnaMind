/**
 * The IncarnaMind core: all app logic, with no Electron imports (ADR-0004, ADR-0006).
 * Hosts construct it with `createCore`, passing platform capabilities as adapters.
 */
export type {
  Browser,
  CoreAdapters,
  Embedder,
  EmbeddingModelFiles,
  EmbeddingModelSource,
  Keychain,
  ModelFile,
  Paths,
  ProcessLauncher,
  SpawnOptions,
} from "./adapters";
export type {
  AnswerEngine,
  AnswerEngineEvent,
  AnswerMessage,
  AnswerRequest,
} from "./answers/engine";
export * from "./api";
export type { DataFlowDefinition, DataFlowRegistry } from "./consent";
export { type Core, createCore, DATABASE_FILE } from "./core";
export type { DocumentFile } from "./documents";
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
export type { PreparedChatModel } from "./providers/chat";
export type { ChatGptCredentials } from "./providers/chatgpt/codexEndpoint";
export { ChatGptPlanError, ChatGptSignInRequiredError } from "./providers/chatgpt/errors";
export type { ChatGptPlanEndpoints } from "./providers/chatgpt/plan";
export { OLLAMA_DEFAULT_URL } from "./providers/kinds";
export {
  type ChatLanguageModel,
  type ChatModelFactory,
  type ChatModelSpec,
  createAiSdkChatModel,
} from "./providers/models";
export { RECOMMENDED_OLLAMA_MODEL } from "./providers/ollama";
export { classifyProviderError } from "./providers/providerErrors";
