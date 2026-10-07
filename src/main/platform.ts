/**
 * Electron implementations of the core's adapters. This is the only place that
 * turns Electron APIs into capabilities the core can use.
 */
import { join } from "node:path";
import { app, safeStorage, shell } from "electron";
import {
  type Browser,
  BUILT_IN_SKILLS_PACKAGED,
  BUILT_IN_SKILLS_SOURCE,
  type CoreAdapters,
  type CrashReporter,
  type FileShell,
  type Keychain,
  type Logger,
} from "../core";
import { createSentryCrashReporter } from "./crashReports";
import { createUtilityProcessEmbedder } from "./embedder";
import { createLoginShellProcesses } from "./processes";
import { createFileKeychain, SECRETS_FILE, type SecretCipher } from "./secretsFile";

/**
 * `safeStorage` encrypts with a key held by the OS secret store: the macOS
 * Keychain, Windows DPAPI, or GNOME Keyring / KWallet on Linux. On Linux with
 * no keyring running it falls back to "basic_text", a hard-coded key, which
 * the core treats as plain text.
 */
const safeStorageCipher: SecretCipher = {
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
function createSafeStorageKeychain(dataDir: string): Keychain {
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
 * Documents' files, opened in their default app or shown in the file
 * manager, where the User keeps them. `shell` is looked up at each call, so
 * the smoke tests can stand in for it.
 */
const fileShell: FileShell = {
  async openPath(path) {
    const error = await shell.openPath(path);
    if (error) throw new Error(error);
  },
  showItemInFolder(path) {
    shell.showItemInFolder(path);
  },
};

/**
 * Local Connectors (and later Skill scripts) start with the User's
 * login-shell environment, read once (see ./processes).
 */
const loginShellProcesses = createLoginShellProcesses({
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

/**
 * The built-in Skills: in a packaged app, where electron-builder copied them
 * into its resources (`extraResources` in electron-builder.yml); otherwise
 * (development, the smoke tests) in the repository.
 */
function builtInSkillsFolder(): string {
  return app.isPackaged
    ? join(process.resourcesPath, BUILT_IN_SKILLS_PACKAGED)
    : join(app.getAppPath(), BUILT_IN_SKILLS_SOURCE);
}

/**
 * Where crash reports go: the Sentry DSN this copy was built with
 * (`MAIN_VITE_SENTRY_DSN`), if any. A test build ignores it, so the smoke
 * tests never reach Sentry; they may point it at a local server instead.
 */
function crashReportDsn(): string | undefined {
  const dsn =
    import.meta.env.MODE === "test"
      ? process.env.INCARNAMIND_TEST_SENTRY_DSN
      : import.meta.env.MAIN_VITE_SENTRY_DSN;
  return dsn?.trim() || undefined;
}

/** Sentry crash reports, if this copy can send them. The core starts them only once the User opts in. */
function createCrashReporter(dataDir: string): CrashReporter | undefined {
  const dsn = crashReportDsn();
  if (!dsn) return undefined;
  return createSentryCrashReporter({
    dsn,
    paths: {
      homeDir: app.getPath("home"),
      dataDir,
      others: [app.getPath("temp")],
    },
  });
}

/** Builds the core's adapters, with the log the core writes to. Call after `app` is ready. */
export function createElectronAdapters(log: Logger): CoreAdapters {
  const dataDir = app.getPath("userData");
  const crashReporter = createCrashReporter(dataDir);
  return {
    ...(crashReporter && { crashReporter }),
    log,
    paths: { dataDir, builtInSkills: builtInSkillsFolder() },
    systemLanguages: () => app.getPreferredSystemLanguages(),
    keychain: createSafeStorageKeychain(dataDir),
    browser: systemBrowser,
    shell: fileShell,
    processes: loginShellProcesses,
    // JavaScript Skill scripts run on Electron's own Node, as plain Node: nothing to install.
    scriptRuntimes: {
      node: { command: process.execPath, env: { ELECTRON_RUN_AS_NODE: "1" } },
    },
    embedder: createUtilityProcessEmbedder({ fake: fakeEmbedder }),
    // The fake model has no files to download.
    ...(fakeEmbedder && { embeddingModelSource: { baseUrl: "http://localhost/", files: [] } }),
  };
}
