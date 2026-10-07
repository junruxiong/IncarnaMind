/** Names shared by the main process and the preload script for the typed core bridge. */
import type { CoreApiMethod } from "../core/api";

/** The renderer reaches the core as `window.incarnamind`. */
export const BRIDGE_KEY = "incarnamind";

export const channelFor = (method: CoreApiMethod): string => `core:${method}`;

/** The one channel the main process uses to push core events to the renderer: (event name, payload). */
export const EVENT_CHANNEL = "core:event";

/**
 * Helpers the preload script adds next to the core, as `window.incarnamindFiles`.
 * They need Electron in the renderer's process, so they aren't part of the core's API.
 */
export const FILES_BRIDGE_KEY = "incarnamindFiles";

export interface FilesBridge {
  /** The absolute path of a dropped or picked file, or "" for a file that isn't on disk. */
  pathForFile(file: File): string;
  /**
   * Shows the system's open dialog for a Skill to import: a folder, or a zip
   * file. Resolves with its absolute path, or null if the User cancelled.
   */
  pickSkill(kind: SkillPickKind): Promise<string | null>;
}

export type SkillPickKind = "folder" | "zip";

/** The main process shows the dialog `FilesBridge.pickSkill` asks for on this channel. */
export const PICK_SKILL_CHANNEL = "files:pickSkill";
