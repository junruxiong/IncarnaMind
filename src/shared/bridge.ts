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
  /** Shows the data folder in the system's file manager, e.g. to back it up. */
  openDataFolder(): Promise<void>;
  /**
   * Shows the system's open dialog for a Skill to import: a folder, or a zip
   * file. Resolves with its absolute path, or null if the User cancelled.
   */
  pickSkill(kind: SkillPickKind): Promise<string | null>;
  /**
   * Opens a Document's original file in the system's default app for its
   * type: a temporary copy, named after the Document. Rejects for a deleted Document.
   */
  openDocumentExternally(documentId: string): Promise<void>;
  /**
   * Saves a copy of a Document's original file where the User picks in the
   * system save dialog, suggesting the Document's name and extension.
   * Resolves with the copy's path, or null if they cancelled. Rejects for a
   * deleted Document.
   */
  saveDocumentCopy(documentId: string): Promise<string | null>;
}

export type SkillPickKind = "folder" | "zip";

/** The main process's channels for the `FilesBridge` methods that need it. */
export const FILES_CHANNELS = {
  saveMindExport: "files:saveMindExport",
  openDataFolder: "files:openDataFolder",
  pickSkill: "files:pickSkill",
  openDocumentExternally: "files:openDocumentExternally",
  saveDocumentCopy: "files:saveDocumentCopy",
} as const;
