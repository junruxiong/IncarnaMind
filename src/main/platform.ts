/**
 * Electron implementations of the core's adapters. This is the only place that
 * turns Electron APIs into capabilities the core can use.
 */
import { join } from "node:path";
import { app, safeStorage, shell } from "electron";
import type { Browser, CoreAdapters, Keychain } from "../core";
import { createUtilityProcessEmbedder } from "./embedder";
import { createLoginShellProcesses } from "./processes";
import { createFileKeychain, SECRETS_FILE, type SecretCipher } from "./secretsFile";

/**
 * `safeStorage` encrypts with a key held by the OS secret store: the macOS
 * Keychain, Windows DPAPI, or GNOME Keyring / KWallet on Linux. On Linux with
 * no keyring running it falls back to "basic_text", a hard-coded key, which
 * the core treats as plain text.
 */
export const safeStorageCipher: SecretCipher = {
  protection() {
    if (process.platform === "linux" && safeStorage.getSelectedStorageBackend() === "basic_text") {
      return "plain-text";
    }
    return safeStorage.isEncryptionAvailable() ? "os" : "unavailable";
  },
  allowPlainText() {
    // Without this, safeStorage refuses to encrypt on Linux's basic_text backend.
    if (process.platform === "linux") safeStorage.setUsePlainTextEncryption(true);
  },
  encrypt: (plainText) => safeStorage.encryptString(plainText),
  decrypt: (cipherText) => safeStorage.decryptString(cipherText),
};

/** Secrets encrypted with `safeStorage`, kept in a secrets file in the data folder, never in SQLite. */
export function createSafeStorageKeychain(dataDir: string): Keychain {
  return createFileKeychain(join(dataDir, SECRETS_FILE), safeStorageCipher);
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

/**
 * Local Connectors (and later Skill scripts) start with the User's
 * login-shell environment, read once (see ./processes).
 */
export const loginShellProcesses = createLoginShellProcesses({
  reportError: (error) =>
    console.warn("Couldn't read the login shell's environment; using the app's own.", error),
});

/**
 * Test-only launch flag: the smoke tests run a deterministic fake embedding
 * model in the utility process, so no model is downloaded. Only a test build
 * (`electron-vite build --mode test`) honours it.
 */
const fakeEmbedder =
  import.meta.env.MODE === "test" && process.env.INCARNAMIND_TEST_EMBEDDER === "fake";

/** Builds the core's adapters. Call after `app` is ready. */
export function createElectronAdapters(): CoreAdapters {
  const dataDir = app.getPath("userData");
  return {
    paths: { dataDir },
    systemLanguages: () => app.getPreferredSystemLanguages(),
    keychain: createSafeStorageKeychain(dataDir),
    browser: systemBrowser,
    processes: loginShellProcesses,
    embedder: createUtilityProcessEmbedder({ fake: fakeEmbedder }),
    // The fake model has no files to download.
    ...(fakeEmbedder && { embeddingModelSource: { baseUrl: "http://localhost/", files: [] } }),
  };
}
