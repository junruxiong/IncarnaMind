/**
 * Electron implementations of the core's adapters. This is the only place that
 * turns Electron APIs into capabilities the core can use.
 */
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { app, safeStorage, shell } from "electron";
import {
  type AudioCapture,
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
import { createFakeCapture, type FakeCapture } from "../core/meetings/fakeCapture";
import type { FakeTranscriberOptions } from "../core/transcription/fake";
import { createSentryCrashReporter } from "./crashReports";
import { createUtilityProcessEmbedder } from "./embedder";
import { createElectronAudioCapture } from "./meetings/capture";
import { createElectronPictures } from "./pictures";
import { createLoginShellProcesses } from "./processes";
import { createUtilityProcessCrossEncoder } from "./reranker";
import { loadOsSandbox } from "./sandbox";
import { createUtilityProcessTextRecognizer } from "./scanReader";
import { createFileKeychain, SECRETS_FILE, type SecretCipher } from "./secretsFile";
import { createUtilityProcessTranscriber } from "./transcriber";
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

/** The apps whose install links Settings → Agents opens (#351), and no others. */
const AGENT_LINK_SCHEMES = new Set(["cursor:"]);

/** Opens an agent's install link in its app: its app shows what it installs, and asks. */
async function openAgentLink(url: string): Promise<void> {
  const { protocol } = new URL(url);
  if (!AGENT_LINK_SCHEMES.has(protocol)) throw new Error(`Refusing to open a ${protocol} link.`);
  await shell.openExternal(url);
}

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

/** With the fake models, what the fake text recognition model reads on each scanned page (#397). */
const fakeScans = fakeEmbedder ? (process.env.INCARNAMIND_TEST_SCANS ?? null) : null;

/**
 * Test-only launch flags: the smoke tests transcribe Meetings with a fake
 * speech-to-text engine in the utility process, so no model is downloaded
 * (#136), slow, failing to read stretches or failing over as
 * INCARNAMIND_TEST_TRANSCRIBER_OPTIONS (JSON) asks (#161). Only a test build
 * honours them.
 */
const fakeTranscriber: FakeTranscriberOptions | undefined =
  import.meta.env.MODE === "test" && process.env.INCARNAMIND_TEST_TRANSCRIBER === "fake"
    ? JSON.parse(process.env.INCARNAMIND_TEST_TRANSCRIBER_OPTIONS ?? "{}")
    : undefined;

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
 * The citation-style engine, styles and locales (ADR-0016), found like the built-in Skills: in
 * the packaged app's resources, otherwise in the repository.
 */
function citationStylesFolder(): string {
  return app.isPackaged
    ? join(process.resourcesPath, "citation-styles")
    : join(app.getAppPath(), "resources", "citation-styles");
}

/**
 * The User's Documents folder, where the first save offers to put the Output
 * location. A test build uses one in its data folder, so the smoke tests never
 * write into the User's.
 */
function documentsFolder(dataDir: string): string {
  return import.meta.env.MODE === "test" ? join(dataDir, "Documents") : app.getPath("documents");
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

/**
 * Test-only launch flag: where Look up asks Crossref and arXiv (#286), a
 * local server standing in for both, so the smoke tests reach neither. Only
 * a test build honours it.
 */
const referenceLookupUrl =
  import.meta.env.MODE === "test" ? process.env.INCARNAMIND_TEST_REFERENCE_LOOKUP : undefined;

/**
 * Test-only launch flag: where Find sources asks its providers (#296), a
 * local server standing in for each provider (OpenAlex, arXiv, PubMed, Crossref, openFDA, DailyMed, ClinicalTrials.gov, the eCFR and GovInfo), each
 * under its own path, so the smoke tests reach none of them. Only a test
 * build honours it.
 */
const sourceProvidersUrl =
  import.meta.env.MODE === "test" ? process.env.INCARNAMIND_TEST_SOURCES : undefined;

/**
 * Test-only launch flag: a local server standing in for OpenAI (#122), at
 * this origin, where the ChatGPT plan signs in and asks, so the smoke tests
 * can sign in and reach neither OpenAI's sign-in nor its API. Only a test
 * build honours it.
 */
function chatGptStandIn(): CoreAdapters["chatGptPlan"] {
  const origin = import.meta.env.MODE === "test" ? process.env.INCARNAMIND_TEST_CHATGPT : undefined;
  if (!origin) return undefined;
  return {
    issuer: origin,
    authorizeUrl: `${origin}/api/accounts/authorize`,
    tokenUrl: `${origin}/api/accounts/oauth/token`,
    revokeUrl: `${origin}/api/accounts/oauth/revoke`,
    jwksUrl: `${origin}/.well-known/jwks.json`,
    apiBaseUrl: `${origin}/v1`,
  };
}

/**
 * What records Meetings: the hidden capture window (ADR-0015). Test-only
 * launch flags: the smoke tests record a fake that makes a tone in real time,
 * so no device is ever opened (INCARNAMIND_TEST_AUDIO=fake), its computer's
 * sound allowed, denied or silent (INCARNAMIND_TEST_AUDIO_SYSTEM). They reach
 * it as `globalThis.incarnamindFakeAudio`, e.g. to change the microphone.
 */
function audioCapture(log: Logger): AudioCapture {
  if (import.meta.env.MODE === "test" && process.env.INCARNAMIND_TEST_AUDIO === "fake") {
    const system = process.env.INCARNAMIND_TEST_AUDIO_SYSTEM;
    const fake = createFakeCapture({
      realTime: 100,
      system: system === "denied" || system === "silent" ? system : "allowed",
    });
    (globalThis as { incarnamindFakeAudio?: FakeCapture }).incarnamindFakeAudio = fake;
    return fake;
  }
  return createElectronAudioCapture({ log, rendererUrl: process.env.ELECTRON_RENDERER_URL });
}

/**
 * A development or test build may keep the ChatGPT sign-in in a ChatGPT test
 * profile, the folder in INCARNAMIND_CHATGPT_PROFILE, instead of the data
 * folder: the owner signs in there once, and coding agents running with
 * temporary data folders use the same sign-in for checks by hand
 * (docs/agents/testing.md). Its tokens stay encrypted by safeStorage, in that
 * folder only, and a lock file keeps the processes' refreshes apart. A
 * release build never reads the variable.
 */
function chatGptTestProfile(log: Logger): CoreAdapters["chatGptStore"] {
  const folder = process.env.INCARNAMIND_CHATGPT_PROFILE;
  if (import.meta.env.MODE === "production" || !folder) return undefined;
  const dir = resolve(folder);
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  log.info("chatgpt.testProfile", { folder: dir });
  return {
    keychain: createFileKeychain(join(dir, SECRETS_FILE), safeStorageCipher),
    lockFile: join(dir, "refresh.lock"),
  };
}

/** Builds the core's adapters, with the log the core writes to. Call after `app` is ready. */
export function createElectronAdapters(log: Logger): CoreAdapters {
  const dataDir = app.getPath("userData");
  const crashReporter = createCrashReporter(dataDir);
  const usageData = createUsageDataSender();
  const chatGptStore = chatGptTestProfile(log);
  const chatGptPlan = chatGptStandIn();
  return {
    ...(chatGptStore && { chatGptStore }),
    ...(chatGptPlan && { chatGptPlan }),
    ...(crashReporter && { crashReporter }),
    ...(usageData && { usageData }),
    ...(referenceLookupUrl && {
      referenceLookup: { crossref: referenceLookupUrl, arxiv: referenceLookupUrl },
    }),
    ...(sourceProvidersUrl && {
      sourceProviders: {
        openalex: `${sourceProvidersUrl}/openalex`,
        arxiv: `${sourceProvidersUrl}/arxiv`,
        pubmed: `${sourceProvidersUrl}/pubmed`,
        crossref: `${sourceProvidersUrl}/crossref`,
        openfda: `${sourceProvidersUrl}/openfda`,
        dailymed: `${sourceProvidersUrl}/dailymed`,
        clinicaltrials: `${sourceProvidersUrl}/clinicaltrials`,
        ecfr: `${sourceProvidersUrl}/ecfr`,
        govinfo: `${sourceProvidersUrl}/govinfo`,
      },
    }),
    log,
    audioCapture: audioCapture(log),
    paths: {
      dataDir,
      builtInSkills: builtInSkillsFolder(),
      examples: examplesFolder(),
      citationStyles: citationStylesFolder(),
      documents: documentsFolder(dataDir),
    },
    systemLanguages: () => app.getPreferredSystemLanguages(),
    keychain: createSafeStorageKeychain(dataDir),
    browser: systemBrowser,
    shell: fileShell,
    pictures: createElectronPictures(log),
    processes: loginShellProcesses,
    // JavaScript Skill scripts run on Electron's own Node, as plain Node: nothing to install.
    scriptRuntimes: {
      node: { command: process.execPath, env: { ELECTRON_RUN_AS_NODE: "1" } },
    },
    // An outside agent starts the shim (#351) the same way: this app's binary, as plain Node,
    // running the shim built beside this file (inside app.asar, which Node mode reads).
    agents: {
      launcher: {
        command: process.execPath,
        args: [join(__dirname, "agentShim.js")],
        env: { ELECTRON_RUN_AS_NODE: "1" },
      },
      shimScript: join(__dirname, "agentShim.js"),
      version: app.getVersion(),
      openLink: openAgentLink,
    },
    embedder: createUtilityProcessEmbedder({ fake: fakeEmbedder }),
    transcriber: createUtilityProcessTranscriber(fakeTranscriber ? { fake: fakeTranscriber } : {}),
    ...(fakeTranscriber && {
      transcriptionModelSource: {
        model: { baseUrl: "http://localhost/", files: [] },
        vad: { baseUrl: "http://localhost/", files: [] },
      },
    }),
    crossEncoder: createUtilityProcessCrossEncoder({ fake: fakeEmbedder }),
    textRecognizer: createUtilityProcessTextRecognizer(
      fakeEmbedder ? { fake: { script: fakeScans } } : {},
    ),
    // The fake models have no files to download.
    ...(fakeEmbedder && {
      embeddingModelSource: { baseUrl: "http://localhost/", files: [] },
      rerankingModelSource: { baseUrl: "http://localhost/", files: [] },
      textRecognitionModelSource: { baseUrl: "http://localhost/", files: [] },
    }),
  };
}
