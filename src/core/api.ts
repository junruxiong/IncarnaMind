/**
 * The core's public interface: the only API the UI uses.
 *
 * The preload script exposes it to the renderer over IPC, and the tests drive it
 * directly. Everything that crosses it is plain data, so it survives IPC.
 *
 * This file must stay free of imports with side effects: the preload script and
 * the renderer import it.
 */
import type { Language, LanguagePreference } from "./language";

export interface Mind {
  /** A random UUID generated on this device. */
  id: string;
  /** May be empty: the UI shows an "Untitled" placeholder. */
  title: string;
  /** ISO 8601, UTC. */
  createdAt: string;
  /** ISO 8601, UTC. */
  updatedAt: string;
}

export interface CreateMindInput {
  title?: string;
}

/** Settings that belong to the User and will sync across their devices (ADR-0003). */
export interface UserSettings {
  language: LanguagePreference;
}

/** Settings that belong to this device and never sync (ADR-0003). */
export interface DeviceSettings {
  /** Width of the left sidebar, in CSS pixels. */
  sidebarWidth: number;
  /** Width of the right Document viewer pane, in CSS pixels. */
  viewerWidth: number;
}

export interface Settings {
  user: UserSettings;
  device: DeviceSettings;
  /** The interface language in effect: the User's choice, or the OS language for "system". */
  language: Language;
}

export interface SettingsPatch {
  user?: Partial<UserSettings>;
  device?: Partial<DeviceSettings>;
}

export interface CoreApi {
  createMind(input?: CreateMindInput): Promise<Mind>;
  /** Minds that are not deleted, most recently updated first. */
  listMinds(): Promise<Mind[]>;
  getSettings(): Promise<Settings>;
  /** Changes only the fields given and returns the settings now in effect. */
  updateSettings(patch: SettingsPatch): Promise<Settings>;
}

export type CoreApiMethod = keyof CoreApi;

// A Record over every method name: adding a method to CoreApi without listing it here fails the type-check.
const methods: Record<CoreApiMethod, true> = {
  createMind: true,
  listMinds: true,
  getSettings: true,
  updateSettings: true,
};

/** Every method of CoreApi, used to wire the IPC bridge. */
export const coreApiMethods = Object.keys(methods) as CoreApiMethod[];
