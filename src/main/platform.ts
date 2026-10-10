/**
 * Electron implementations of the core's adapters. This is the only place that
 * turns Electron APIs into capabilities the core can use.
 */
import { tmpdir } from "node:os";
import { join } from "node:path";
import { app, safeStorage, shell } from "electron";
import {
  type Browser,
  BUILT_IN_SKILLS_PACKAGED,
  BUILT_IN_SKILLS_SOURCE,
  type CoreAdapters,
  type CrashReporter,
  type Executor,
  type FileShell,
  type Keychain,
  type Logger,
  type UsageDataSender,
} from "../core";
import { createSentryCrashReporter } from "./crashReports";
import { createUtilityProcessEmbedder } from "./embedder";
import { createLoginShellProcesses } from "./processes";
import { createUtilityProcessCrossEncoder } from "./reranker";
import { loadOsSandbox } from "./sandbox";
import { createFileKeychain, SECRETS_FILE, type SecretCipher } from "./secretsFile";
import { createPostHogSender } from "./usageData";

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
 * Local Connectors and Skill scripts start with the User's login-shell
 * environment, read once (see ./processes).
 */
const loginShellProcesses = createLoginShellProcesses({
  reportError: (error) =>
    console.warn("Couldn't read the login shell's environment; using the app's own.", error),
});

/**
 * In a packaged Linux app, sandbox-runtime's seccomp helper outside app.asar
 * (electron-builder.yml unpacks it), where bubblewrap can run it.
 */
function seccompHelper(): string | undefined {
  if (process.platform !== "linux" || !app.isPackaged) return undefined;
  const runtime = join("node_modules", "@anthropic-ai", "sandbox-runtime");
  return join(
    process.resourcesPath,
    "app.asar.unpacked",
    runtime,
    "vendor",
    "seccomp",
    process.arch,
    "apply-seccomp",
  );
}

/**
 * The Executor Skill scripts run on (#65): the OS sandbox's, at level "os",
 * where it can start (macOS; Linux with bubblewrap). Elsewhere none is
 * given, and the core runs them at level "none", where each run asks first.
 * Found out once, at startup: the log says which, and why not "os". Call
 * once `userData` is set.
 */
export async function chooseExecutor(log: Logger): Promise<Executor | undefined> {
  try {
    const sandbox = await loadOsSandbox({ seccompHelper: seccompHelper() });
    if (!sandbox.available) {
      log.info("scripts.sandbox", { level: "none", reason: sandbox.reason });
      return undefined;
    }
    log.info("scripts.sandbox", { level: "os" });
    return sandbox.createExecutor({
      processes: loginShellProcesses,
      environment: () => loginShellProcesses.environment(),
      tempDir: tmpdir(),
      denyRead: [app.getPath("home"), app.getPath("userData")],
      reportError: (error) => console.error(error),
    });
  } catch (error) {
    log.error("scripts.sandbox", {
      level: "none",
      reason: error instanceof Error ? error.message : String(error),
    });
    return undefined;
  }
}

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
 * The example Documents for the example Mind, found like the built-in Skills.
 * The smoke tests start without them (each would otherwise begin with the
 * example Mind), unless one asks for them with INCARNAMIND_TEST_EXAMPLES=1.
 */
function examplesFolder(): string | undefined {
  if (import.meta.env.MODE === "test" && process.env.INCARNAMIND_TEST_EXAMPLES !== "1") {
    return undefined;
  }
  return app.isPackaged
    ? join(process.resourcesPath, "examples")
    : join(app.getAppPath(), "resources", "examples");
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

/**
 * The PostHog project this copy sends usage data to, if any: its key and host
 * (`MAIN_VITE_POSTHOG_KEY`, `MAIN_VITE_POSTHOG_HOST`), both needed, the host
 * over https; and whether it is a test build (`MAIN_VITE_TESTER_BUILD=1`, the
 * alpha). A smoke-test build ignores them, so the smoke tests never reach
 * PostHog; they may point it at a local server instead
 * (`INCARNAMIND_TEST_POSTHOG_KEY`, `INCARNAMIND_TEST_POSTHOG_HOST`,
 * `INCARNAMIND_TEST_TESTER_BUILD`).
 */
function usageDataProject(): { key: string; host: string; testerBuild: boolean } | undefined {
  const test = import.meta.env.MODE === "test";
  const key = (
    test ? process.env.INCARNAMIND_TEST_POSTHOG_KEY : import.meta.env.MAIN_VITE_POSTHOG_KEY
  )?.trim();
  const host = (
    test ? process.env.INCARNAMIND_TEST_POSTHOG_HOST : import.meta.env.MAIN_VITE_POSTHOG_HOST
  )?.trim();
  const tester = test
    ? process.env.INCARNAMIND_TEST_TESTER_BUILD
    : import.meta.env.MAIN_VITE_TESTER_BUILD;
  if (!key || !host) return undefined;
  let url: URL;
  try {
    url = new URL(host);
  } catch {
    return undefined;
  }
  const local = url.hostname === "127.0.0.1" || url.hostname === "localhost";
  if (url.protocol !== "https:" && !(test && local && url.protocol === "http:")) return undefined;
  return { key, host, testerBuild: tester?.trim() === "1" };
}

/** Sends usage data, if this copy was built to. The core starts it only while the User agrees. */
function createUsageDataSender(): UsageDataSender | undefined {
  const project = usageDataProject();
  if (!project) return undefined;
  return createPostHogSender({
    projectKey: project.key,
    host: project.host,
    testerBuild: project.testerBuild,
    appVersion: app.getVersion(),
    // The smoke tests don't wait half a minute for events.
    ...(import.meta.env.MODE === "test" && { flushIntervalMs: 500 }),
  });
}

/** Builds the core's adapters, with the log the core writes to. Call after `app` is ready. */
export function createElectronAdapters(log: Logger): CoreAdapters {
  const dataDir = app.getPath("userData");
  const crashReporter = createCrashReporter(dataDir);
  const usageData = createUsageDataSender();
  return {
    ...(crashReporter && { crashReporter }),
    ...(usageData && { usageData }),
    log,
    paths: { dataDir, builtInSkills: builtInSkillsFolder(), examples: examplesFolder() },
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
    crossEncoder: createUtilityProcessCrossEncoder({ fake: fakeEmbedder }),
    // The fake models have no files to download.
    ...(fakeEmbedder && {
      embeddingModelSource: { baseUrl: "http://localhost/", files: [] },
      rerankingModelSource: { baseUrl: "http://localhost/", files: [] },
    }),
  };
}
