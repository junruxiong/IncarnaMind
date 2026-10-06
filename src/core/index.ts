/**
 * The IncarnaMind core: all app logic, with no Electron imports (ADR-0004, ADR-0006).
 * Hosts construct it with `createCore`, passing platform capabilities as adapters.
 */
export type {
  Browser,
  CoreAdapters,
  Keychain,
  Paths,
  ProcessLauncher,
  SpawnOptions,
} from "./adapters";
export * from "./api";
export type { DataFlowDefinition, DataFlowRegistry } from "./consent";
export { type Core, createCore, DATABASE_FILE } from "./core";
export {
  ChatNotReadyError,
  ConsentDeclinedError,
  InvalidInputError,
  SecretStorageError,
} from "./errors";
export * from "./language";
export type { PreparedChatModel } from "./providers/chat";
export { OLLAMA_DEFAULT_URL } from "./providers/kinds";
export type { ChatLanguageModel, ChatModelFactory, ChatModelSpec } from "./providers/models";
export { RECOMMENDED_OLLAMA_MODEL } from "./providers/ollama";
export { classifyProviderError } from "./providers/providerErrors";
