/**
 * Electron implementations of the core's adapters. This is the only place that
 * turns Electron APIs into capabilities the core can use.
 */
import { readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { app, safeStorage, shell } from "electron";
import type { Browser, CoreAdapters, Keychain, ProcessLauncher } from "../core";

/**
 * Secrets encrypted with `safeStorage` (backed by the OS keychain) and kept in
 * a small file in the data folder, never in SQLite. Nothing uses it yet.
 */
export function createSafeStorageKeychain(file: string): Keychain {
  const load = async (): Promise<Record<string, string>> => {
    try {
      return JSON.parse(await readFile(file, "utf8")) as Record<string, string>;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return {};
      throw error;
    }
  };
  const save = async (entries: Record<string, string>) => {
    const temporary = `${file}.tmp`;
    await writeFile(temporary, JSON.stringify(entries), { mode: 0o600 });
    await rename(temporary, file);
  };

  return {
    async get(name) {
      const encrypted = (await load())[name];
      return encrypted === undefined
        ? null
        : safeStorage.decryptString(Buffer.from(encrypted, "base64"));
    },
    async set(name, secret) {
      if (!safeStorage.isEncryptionAvailable()) {
        throw new Error("The OS keychain isn't available, so the secret can't be stored safely.");
      }
      const entries = await load();
      entries[name] = safeStorage.encryptString(secret).toString("base64");
      await save(entries);
    },
    async delete(name) {
      const entries = await load();
      if (!Object.hasOwn(entries, name)) return;
      delete entries[name];
      await save(entries);
    },
  };
}

export const systemBrowser: Browser = {
  async open(url) {
    const { protocol } = new URL(url);
    if (protocol !== "https:" && protocol !== "http:") {
      throw new Error(`Refusing to open a ${protocol} URL in the browser.`);
    }
    await shell.openExternal(url);
  },
};

/** Stub: Connectors and Skill scripts bring login-shell environment resolution and spawning. */
export const loginShellProcesses: ProcessLauncher = {
  spawn() {
    throw new Error("Starting processes isn't available yet.");
  },
};

/** Builds the core's adapters. Call after `app` is ready. */
export function createElectronAdapters(): CoreAdapters {
  const dataDir = app.getPath("userData");
  return {
    paths: { dataDir },
    systemLanguages: () => app.getPreferredSystemLanguages(),
    keychain: createSafeStorageKeychain(join(dataDir, "keychain.json")),
    browser: systemBrowser,
    processes: loginShellProcesses,
  };
}
