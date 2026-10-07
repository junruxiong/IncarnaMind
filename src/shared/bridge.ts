/** Names shared by the main process and the preload script for the typed core bridge. */
import type { CoreApiMethod, ExportMindOptions } from "../core/api";

/** The renderer reaches the core as `window.incarnamind`. */
export const BRIDGE_KEY = "incarnamind";

export const channelFor = (method: CoreApiMethod): string => `core:${method}`;

/** The one channel the main process uses to push core events to the renderer: (event name, payload). */
export const EVENT_CHANNEL = "core:event";

/**
 * Helpers the preload script adds next to the core, as `window.incarnamindFiles`.
 * They need Electron (in the renderer's process, or the main process's dialogs
 * and shell), so they aren't part of the core's API.
 */
export const FILES_BRIDGE_KEY = "incarnamindFiles";

export interface FilesBridge {
  /** The absolute path of a dropped or picked file, or "" for a file that isn't on disk. */
  pathForFile(file: File): string;
  /**
   * Exports a Mind (see `CoreApi.exportMind`) to a file the User picks in the
   * system save dialog. Resolves with the file's path, or null if they cancelled.
   */
  saveMindExport(mindId: string, options: ExportMindOptions): Promise<string | null>;
  /** Shows a file `saveMindExport` wrote, selected in the system's file manager. */
  showExportInFolder(path: string): Promise<void>;
  /** Shows the data folder in the system's file manager, e.g. to back it up. */
  openDataFolder(): Promise<void>;
  /** Shows the data folder's `logs/` in the system's file manager, e.g. to attach the log to a bug report. */
  openLogsFolder(): Promise<void>;
  /**
   * Shows the system's open dialog for a Skill to import: a folder, or a zip
   * file. Resolves with its absolute path, or null if the User cancelled.
   */
  pickSkill(kind: SkillPickKind): Promise<string | null>;
  /**
   * Opens a Document's file, where the User keeps it, in the system's default
   * app for its type. Rejects for a deleted Document, or one whose file is
   * missing or can't be reached.
   */
  openDocumentExternally(documentId: string): Promise<void>;
  /**
   * Shows a Document's file selected in the system's file manager. Rejects
   * as `openDocumentExternally` does.
   */
  showDocumentInFolder(documentId: string): Promise<void>;
  /**
   * Shows the system's open dialog for a folder to link (see
   * `CoreApi.addLinkedFolder`). Resolves with its absolute path, or null if
   * the User cancelled.
   */
  pickLinkedFolder(): Promise<string | null>;
  /** Writes an error nothing in the window caught to the log, scrubbed of the User's content. */
  logError(report: RendererErrorReport): void;
  /** Calls `listener` with each command chosen in the application menu. Returns how to stop. */
  onMenuCommand(listener: (command: MenuCommand) => void): () => void;
}

/** What the application menu asks the window to do (see src/main/menu.ts). */
export type MenuCommand = "new-mind" | "close-tab" | "open-settings";

export type SkillPickKind = "folder" | "zip";

/** An error nothing in the window caught: a thrown error, or a rejected promise nobody handled. */
export interface RendererErrorReport {
  kind: "error" | "rejection";
  name: string;
  message: string;
  stack: string;
}

/** The main process's channels for the `FilesBridge` methods that need it. */
export const FILES_CHANNELS = {
  saveMindExport: "files:saveMindExport",
  showExportInFolder: "files:showExportInFolder",
  openDataFolder: "files:openDataFolder",
  openLogsFolder: "files:openLogsFolder",
  pickSkill: "files:pickSkill",
  openDocumentExternally: "files:openDocumentExternally",
  showDocumentInFolder: "files:showDocumentInFolder",
  pickLinkedFolder: "files:pickLinkedFolder",
  logError: "files:logError",
  /** From the main process: a command chosen in the application menu. */
  menuCommand: "app:menuCommand",
} as const;
