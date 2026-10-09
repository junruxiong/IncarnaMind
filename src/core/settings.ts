import { randomUUID } from "node:crypto";
import {
  type ChatModelChoice,
  type DeviceSettings,
  type GettingStarted,
  type SandboxLevel,
  type Settings,
  SKILL_SCRIPT_LIMITS,
  type UserSettings,
} from "./api";
import { InvalidInputError, isRecord } from "./errors";
import { isLanguagePreference, resolveLanguage } from "./language";
import type { Database } from "./storage";

/**
 * Settings are split into per-User (synced later) and per-device (never synced),
 * each in its own table (ADR-0003). A row holds one setting as JSON.
 */
interface Scope<T> {
  name: "user" | "device";
  table: "user_settings" | "device_settings";
  defaults: T;
  validators: { [K in keyof T]-?: (value: unknown) => value is T[K] };
}

const isPaneWidth = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 10_000;

const isBoolean = (value: unknown): value is boolean => typeof value === "boolean";

/** A Mind's ID, as a tab names it: any short non-blank text (the Mind may be gone since). */
const isMindId = (value: unknown): value is string =>
  typeof value === "string" && value.trim() !== "" && value.length <= 200;

/** Open Mind tabs: a list of distinct IDs, at most 100. */
const isMindIdList = (value: unknown): value is string[] =>
  Array.isArray(value) &&
  value.length <= 100 &&
  value.every(isMindId) &&
  new Set(value).size === value.length;

const isNonBlankText = (value: unknown): value is string =>
  typeof value === "string" && value.trim() !== "" && value.length <= 500;

/** Checks the shape only; the core checks that the provider exists. */
export const isChatModelChoice = (value: unknown): value is ChatModelChoice =>
  isRecord(value) &&
  Object.keys(value).length === 2 &&
  isNonBlankText(value.providerId) &&
  isNonBlankText(value.modelId);

const userScope: Scope<UserSettings> = {
  name: "user",
  table: "user_settings",
  defaults: { language: "system", chatModel: null },
  validators: {
    language: isLanguagePreference,
    chatModel: (value): value is ChatModelChoice | null =>
      value === null || isChatModelChoice(value),
  },
};

const GETTING_STARTED_STEPS = ["started", "citationChecked", "indexed", "askedOwn", "hidden"];

const isGettingStarted = (value: unknown): value is GettingStarted =>
  isRecord(value) &&
  Object.keys(value).length === GETTING_STARTED_STEPS.length &&
  GETTING_STARTED_STEPS.every((key) => isBoolean(value[key]));

const isScriptTimeout = (value: unknown): value is number =>
  typeof value === "number" &&
  Number.isInteger(value) &&
  value >= SKILL_SCRIPT_LIMITS.minTimeoutSeconds &&
  value <= SKILL_SCRIPT_LIMITS.maxTimeoutSeconds;

const deviceScope: Scope<DeviceSettings> = {
  name: "device",
  table: "device_settings",
  defaults: {
    // DESIGN.md: the sidebar is 248px by default; the viewer opens at about half the room beside the sidebar.
    sidebarWidth: 248,
    viewerWidth: null,
    openMinds: [],
    activeMind: null,
    chatSetupDismissed: false,
    gettingStarted: {
      started: false,
      citationChecked: false,
      indexed: false,
      askedOwn: false,
      hidden: false,
    },
    skillScriptsEnabled: true,
    skillScriptTimeoutSeconds: SKILL_SCRIPT_LIMITS.defaultTimeoutSeconds,
  },
  validators: {
    sidebarWidth: isPaneWidth,
    viewerWidth: (value): value is number | null => value === null || isPaneWidth(value),
    openMinds: isMindIdList,
    activeMind: (value): value is string | null => value === null || isMindId(value),
    chatSetupDismissed: isBoolean,
    gettingStarted: isGettingStarted,
    skillScriptsEnabled: isBoolean,
    skillScriptTimeoutSeconds: isScriptTimeout,
  },
};

function validatorFor<T>(scope: Scope<T>, key: string) {
  return Object.hasOwn(scope.validators, key)
    ? (scope.validators[key as keyof T] as (value: unknown) => boolean)
    : undefined;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function read<T>(db: Database, scope: Scope<T>): T {
  const values = { ...scope.defaults } as Record<string, unknown>;
  const rows = db.all<{ key: string; value: string }>(
    `SELECT key, value FROM ${scope.table} WHERE deleted_at IS NULL`,
  );
  for (const row of rows) {
    // Keys this version doesn't know (e.g. written by a newer version) are left alone.
    const isValid = validatorFor(scope, row.key);
    if (!isValid) continue;
    const value = parseJson(row.value);
    if (value !== undefined && isValid(value)) values[row.key] = value;
  }
  return values as T;
}

function parsePatch<T>(scope: Scope<T>, patch: unknown): [string, unknown][] {
  if (patch === undefined) return [];
  if (!isRecord(patch))
    throw new InvalidInputError(`The ${scope.name} settings must be an object.`);
  return Object.entries(patch).map(([key, value]) => {
    const isValid = validatorFor(scope, key);
    if (!isValid) throw new InvalidInputError(`Unknown ${scope.name} setting "${key}".`);
    if (!isValid(value)) {
      throw new InvalidInputError(`Invalid value for the ${scope.name} setting "${key}".`);
    }
    return [key, value];
  });
}

function write(db: Database, table: string, key: string, value: unknown, at: string): void {
  const json = JSON.stringify(value);
  const existing = db.get<{ id: string }>(
    `SELECT id FROM ${table} WHERE key = ? AND deleted_at IS NULL`,
    [key],
  );
  if (existing) {
    db.run(`UPDATE ${table} SET value = ?, updated_at = ? WHERE id = ?`, [json, at, existing.id]);
  } else {
    db.run(`INSERT INTO ${table} (id, key, value, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`, [
      randomUUID(),
      key,
      json,
      at,
      at,
    ]);
  }
}

export type SettingsStore = ReturnType<typeof createSettings>;

/** `scriptSandbox`: the level of the Executor Skill scripts run on (`Settings.scriptSandbox`). */
export function createSettings(
  db: Database,
  now: () => string,
  systemLanguages: () => readonly string[],
  scriptSandbox: SandboxLevel,
) {
  const get = (): Settings => {
    const user = read(db, userScope);
    const device = read(db, deviceScope);
    const language = resolveLanguage(user.language, systemLanguages());
    return { user, device, language, scriptSandbox };
  };

  return {
    get,

    update(patch: unknown): Settings {
      if (!isRecord(patch)) throw new InvalidInputError("updateSettings expects an object.");
      for (const key of Object.keys(patch)) {
        if (key !== "user" && key !== "device") {
          throw new InvalidInputError(`Unknown settings group "${key}".`);
        }
      }
      // Validate everything before writing anything.
      const changes = [
        ...parsePatch(userScope, patch.user).map((change) => [userScope.table, ...change] as const),
        ...parsePatch(deviceScope, patch.device).map(
          (change) => [deviceScope.table, ...change] as const,
        ),
      ];
      const at = now();
      db.transaction(() => {
        for (const [table, key, value] of changes) write(db, table, key, value, at);
      });
      return get();
    },

    /**
     * A per-device value the core keeps for itself, outside `DeviceSettings`,
     * so `updateSettings` can't change it (e.g. accepting plain-text secrets).
     */
    readDeviceValue(key: string): unknown {
      const row = db.get<{ value: string }>(
        `SELECT value FROM ${deviceScope.table} WHERE key = ? AND deleted_at IS NULL`,
        [key],
      );
      return row ? parseJson(row.value) : undefined;
    },

    writeDeviceValue(key: string, value: unknown): void {
      if (validatorFor(deviceScope, key)) throw new Error(`"${key}" is a device setting.`);
      write(db, deviceScope.table, key, value, now());
    },
  };
}
