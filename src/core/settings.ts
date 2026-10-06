import { randomUUID } from "node:crypto";
import type { DeviceSettings, Settings, UserSettings } from "./api";
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

const userScope: Scope<UserSettings> = {
  name: "user",
  table: "user_settings",
  defaults: { language: "system" },
  validators: { language: isLanguagePreference },
};

const deviceScope: Scope<DeviceSettings> = {
  name: "device",
  table: "device_settings",
  defaults: { sidebarWidth: 270, viewerWidth: 420 },
  validators: { sidebarWidth: isPaneWidth, viewerWidth: isPaneWidth },
};

function validatorFor<T>(scope: Scope<T>, key: string) {
  return Object.hasOwn(scope.validators, key)
    ? (scope.validators[key as keyof T] as (value: unknown) => boolean)
    : undefined;
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
    let value: unknown;
    try {
      value = JSON.parse(row.value);
    } catch {
      continue;
    }
    if (isValid(value)) values[row.key] = value;
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

export function createSettings(
  db: Database,
  now: () => string,
  systemLanguages: () => readonly string[],
) {
  const get = (): Settings => {
    const user = read(db, userScope);
    const device = read(db, deviceScope);
    return { user, device, language: resolveLanguage(user.language, systemLanguages()) };
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
  };
}
