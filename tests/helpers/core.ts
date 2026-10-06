import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { onTestFinished } from "vitest";
import { type Core, type CoreAdapters, createCore, type Keychain } from "../../src/core";

/** A fresh, empty data folder, deleted when the current test finishes. */
export async function createTempDataFolder(): Promise<string> {
  const dataDir = await mkdtemp(join(tmpdir(), "incarnamind-test-"));
  onTestFinished(() => rm(dataDir, { recursive: true, force: true }));
  return dataDir;
}

/** An in-memory stand-in for the OS keychain. */
export function createMemoryKeychain(): Keychain {
  const secrets = new Map<string, string>();
  return {
    get: async (name) => secrets.get(name) ?? null,
    set: async (name, secret) => {
      secrets.set(name, secret);
    },
    delete: async (name) => {
      secrets.delete(name);
    },
  };
}

/**
 * Starts the core on `dataDir` with test adapters: an English OS, an in-memory
 * keychain, and no browser or processes. Closed when the current test finishes;
 * call `close()` yourself to simulate quitting the app.
 */
export function startCore(dataDir: string, overrides: Partial<CoreAdapters> = {}): Core {
  const core = createCore({
    paths: { dataDir },
    systemLanguages: () => ["en-US"],
    keychain: createMemoryKeychain(),
    browser: {
      open: async () => {
        throw new Error("Tests can't open a browser.");
      },
    },
    processes: {
      spawn: () => {
        throw new Error("Tests can't spawn processes.");
      },
    },
    ...overrides,
  });
  onTestFinished(() => core.close());
  return core;
}

/** A clock that starts at `start` and moves on one second every time it is read. */
export function tickingClock(start = "2026-10-06T09:00:00.000Z"): () => Date {
  let time = Date.parse(start);
  return () => {
    const date = new Date(time);
    time += 1000;
    return date;
  };
}
