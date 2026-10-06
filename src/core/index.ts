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
export { type Core, createCore, DATABASE_FILE } from "./core";
export type { DocumentFile } from "./documents";
export { InvalidInputError, NotFoundError } from "./errors";
export * from "./language";
