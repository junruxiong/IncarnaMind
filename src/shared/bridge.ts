/** Names shared by the main process and the preload script for the typed core bridge. */
import type { CoreApiMethod } from "../core/api";
import type { MenuCommandId } from "./commands";
import type { TitleBar } from "./titleBar";

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
  /** Shows a file IncarnaMind saved (`CoreApi.saveMindExport`), selected in the system's file manager. */
  showSavedFile(path: string): Promise<void>;
  /** Opens a file IncarnaMind saved (`CoreApi.saveMindExport`) in the system's default app for it. */
  openSavedFile(path: string): Promise<void>;
  /**
   * Shows the system's open dialog for the Output location (see
   * `CoreApi.setOutputLocation`), starting at the current one. Resolves with
   * the folder's absolute path, or null if the User cancelled.
   */
  pickOutputLocation(): Promise<string | null>;
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
   * `CoreApi.addLinkedFolder`), or, given a Linked folder's id, for where
   * that folder is now (see `CoreApi.locateLinkedFolder`). Resolves with its
   * absolute path, or null if the User cancelled.
   */
  pickLinkedFolder(locating?: string): Promise<string | null>;
  /**
   * Shows the system's open dialog for Documents to add, as a sheet on the
   * window, starting in the folder last picked from (else Downloads).
   * Resolves with their absolute paths, or none if the User cancelled.
   */
  pickDocuments(): Promise<string[]>;
  /**
   * Shows the system's open dialog for a reference file to import (CSL JSON,
   * BibTeX or RIS: see `CoreApi.importReferences`), as a sheet on the window,
   * starting in the folder last picked from (else Downloads). Resolves with
   * its absolute path, or null if the User cancelled.
   */
  pickReferenceFile(): Promise<string | null>;
  /**
   * Shows a Linked folder selected in the system's file manager (Finder,
   * Explorer). Rejects for an unknown Linked folder, or one that can't be
   * reached.
   */
  showLinkedFolder(linkedFolderId: string): Promise<void>;
  /** Writes an error nothing in the window caught to the log, scrubbed of the User's content. */
  logError(report: RendererErrorReport): void;
  /** Calls `listener` with each command chosen in the application menu. Returns how to stop. */
  onMenuCommand(listener: (command: MenuCommand) => void): () => void;
  /** Tells the application menu whether the sidebar is hidden, so View says "Hide Sidebar" or "Show Sidebar". */
  setSidebarHidden(hidden: boolean): void;
  /** Tells the Edit menu where Undo and Redo go now, and what they would take back or do again. */
  setEditMenu(state: EditMenuState): void;
  /** The window's title bar (./titleBar.ts), for the page to keep room for the window's buttons. */
  readonly titleBar: TitleBar;
  /**
   * Calls `listener` with true when the window enters full screen, where
   * macOS hides its traffic lights, and false when it leaves. Returns how to stop.
   */
  onFullScreenChange(listener: (fullScreen: boolean) => void): () => void;
}

/** The command the application menu asks the window to run (see src/main/menu.ts). */
export type MenuCommand = MenuCommandId;

/**
 * What the Edit menu's Undo and Redo show for the window in front: whether the
 * focus is in a field or the note, whose own undo they reach, or neither (`text`
 * null), where they reach the app's changes. `undo` and `redo` name the change
 * each would reach ("Move 3 Documents"; "" for one without a name), or are null
 * when there is none. The window decides where they go when they are chosen.
 */
export interface EditMenuState {
  text: "field" | "note" | null;
  undo: string | null;
  redo: string | null;
}

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
  showSavedFile: "files:showSavedFile",
  openSavedFile: "files:openSavedFile",
  pickOutputLocation: "files:pickOutputLocation",
  openDataFolder: "files:openDataFolder",
  openLogsFolder: "files:openLogsFolder",
  pickSkill: "files:pickSkill",
  openDocumentExternally: "files:openDocumentExternally",
  showDocumentInFolder: "files:showDocumentInFolder",
  pickLinkedFolder: "files:pickLinkedFolder",
  pickDocuments: "files:pickDocuments",
  pickReferenceFile: "files:pickReferenceFile",
  showLinkedFolder: "files:showLinkedFolder",
  logError: "files:logError",
  /** From the page: the sidebar is hidden (true) or shown (false), for the View menu's label. */
  sidebarHidden: "app:sidebarHidden",
  /** From the page: where the Edit menu's Undo and Redo go now (`EditMenuState`). */
  editMenu: "app:editMenu",
  /** From the main process: a command chosen in the application menu. */
  menuCommand: "app:menuCommand",
  /** From the main process: the window entered (true) or left (false) full screen. */
  fullScreen: "app:fullScreen",
} as const;
