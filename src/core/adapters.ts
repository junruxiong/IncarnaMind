/**
 * Capabilities the host passes into the core when it constructs it.
 *
 * The desktop app builds these from Electron in `src/main/platform.ts`; tests
 * build fakes; a future hosted server would pass its own. The core never
 * imports Electron (ADR-0004, ADR-0006).
 */
import type { ChildProcess } from "node:child_process";
import type { AnswerEngine } from "./answers/engine";
import type { ExternalService, SandboxLevel, SecretProtection } from "./api";
import type { Reranker } from "./documents/searchTool";
import type { WatchFolder } from "./documents/watcher";
import type { ChatGptPlanEndpoints } from "./providers/chatgpt/plan";
import type { EmbeddingModelFactory } from "./providers/embeddings";
import type { ChatModelFactory } from "./providers/models";
import type { OllamaModels } from "./providers/ollamaModels";
import type { RerankingModelFactory } from "./providers/rerank";
import type { RunEngine } from "./runs/engine";
import type { UsageEventName, UsageValue } from "./usageEvents";

export interface Paths {
  /**
   * The app data folder. It holds the SQLite database, the secrets file,
   * Skills, the embedding model (`models/`) and logs. Documents stay where
   * the User keeps them (ADR-0010): backing up this folder backs up the
   * Minds, the index and settings, not the Documents' files.
   */
  dataDir: string;
  /**
   * The built-in Skills the app ships, one SKILL.md folder each, named like
   * the Skill: `resources/skills/` in the repository, copied into the packaged
   * app's resources. The core installs them into the data folder at startup.
   * Not given: no built-in Skills (tests that aren't about them).
   */
  builtInSkills?: string;
  /**
   * The example Documents the app ships for its example Mind (onboarding):
   * `resources/examples/` in the repository, copied into the packaged app's
   * resources. Not given: no examples (tests that aren't about them).
   */
  examples?: string;
  /**
   * Where each program a Tool runs (a Skill script) gets its own temporary
   * working folder from the local `Executor`, removed when the run ends.
   * Defaults to the OS's temporary folder.
   */
  tempDir?: string;
}

/** A value in a log entry. */
export type LogValue = string | number | boolean | null;

/** What a log entry says about its event. Undefined fields are left out. */
export type LogFields = Readonly<Record<string, LogValue | undefined>>;

/**
 * IncarnaMind's log, for working out what went wrong on the User's computer.
 * The desktop app writes it to `logs/` in the data folder (src/main/log.ts).
 *
 * The core logs what happened, never what the User wrote or keeps secret:
 * ids, statuses, kinds and counts. Never Document text or names, Mind
 * content, Questions, Answers, providers' messages, API keys or tokens.
 */
export interface Logger {
  /** `event` is a short dotted name, e.g. "document.status". */
  info(event: string, fields?: LogFields): void;
  warn(event: string, fields?: LogFields): void;
  error(event: string, fields?: LogFields): void;
}

/**
 * Secrets (API keys, OAuth tokens) stay on this device, outside the database
 * (ADR-0003). The desktop app encrypts them with Electron `safeStorage` and
 * writes the ciphertext to a secrets file in the data folder.
 *
 * The core decides whether a secret may be stored (see `src/core/secrets.ts`);
 * use it through that module, not directly.
 */
export interface Keychain {
  /** How secrets are protected on this device right now. */
  protection(): SecretProtection;
  /**
   * Lets `get` and `set` work when `protection()` is "plain-text". The core
   * calls this only after the User has accepted the risk.
   */
  allowPlainText(): void;
  get(name: string): Promise<string | null>;
  /** Throws if secrets can't be encrypted, or if they would be plain text and `allowPlainText` wasn't called. */
  set(name: string, secret: string): Promise<void>;
  delete(name: string): Promise<void>;
}

/** Opens a URL in the User's default browser, e.g. for a Connector's OAuth sign-in. */
export interface Browser {
  open(url: string): Promise<void>;
}

/**
 * Hands the User's own files to the OS: a Document's file opened in its
 * default app, or shown selected in the file manager. The desktop app uses
 * Electron's `shell`. The core passes only paths of live Documents.
 */
export interface FileShell {
  /** Opens a file in the default app for its type. Rejects if the OS can't. */
  openPath(path: string): Promise<void>;
  /** Shows a file selected in the system's file manager (Finder, Explorer). */
  showItemInFolder(path: string): void;
}

/** How Linked folders are watched and checked (see ./documents/library). The defaults suit the app. */
export interface LinkedFolderOptions {
  /** How long a changed path must stay unchanged before it is read, in ms. Defaults to 300. */
  settleMs?: number;
  /** How often unreachable folders and unreadable files are tried again, in ms. Defaults to 30 seconds. */
  retryMs?: number;
  /**
   * Whether files with a size but no block stored on disk count as cloud
   * placeholders (dataless files). Defaults to true on macOS only.
   */
  detectDatalessFiles?: boolean;
  /**
   * Starts watching a folder: a Linked folder at any depth, a folder holding
   * files added on their own without its subfolders. Defaults to `fs.watch`;
   * tests may pass a fake.
   */
  watch?: WatchFolder;
}

export interface SpawnOptions {
  cwd?: string;
  /** Added to (and overriding) the login-shell environment. */
  env?: Readonly<Record<string, string>>;
  /**
   * Leaves out the login-shell environment's variables whose names this
   * matches: the Executor's, those that look like secrets. `env` is added
   * afterwards, whatever its names.
   */
  omitEnv?: (name: string) => boolean;
  /**
   * On macOS and Linux, starts the process as the leader of a new process
   * group, so it can be stopped together with every process it starts (see
   * `stopProcessTree` in ./execution). Windows has no process groups; there the tree is
   * found by parent process instead.
   */
  processGroup?: boolean;
}

/**
 * Starts child processes with the User's login-shell environment, so `npx` and
 * `uvx` resolve even when the app was opened from the Dock or Start menu. The
 * command is looked up on that environment's PATH. Used by local Connectors,
 * and by the local `Executor` for the programs Tools run (Skill scripts).
 */
export interface ProcessLauncher {
  /**
   * Resolves once the process is running, with its standard input, output and
   * error piped. Rejects if it can't start: a command that isn't found rejects
   * with an error whose `code` is "ENOENT".
   */
  spawn(command: string, args: readonly string[], options?: SpawnOptions): Promise<ChildProcess>;
}

/** How the programs an `Executor` runs are confined (see `SandboxLevel` in ./api, where the UI reads it). */
export type { SandboxLevel } from "./api";

/**
 * A folder an `ExecRequest` allows: an absolute path, or the run's own
 * working folder (its `cwd`, or the new one the Executor makes), whose path
 * the caller can't know beforehand. `WORKING_FOLDER` in ./execution.
 */
export type ExecFolder = string | { readonly kind: "working-folder" };

/** What a program may touch. A folder allowed covers everything inside it. */
export interface ExecAllow {
  read: readonly ExecFolder[];
  write: readonly ExecFolder[];
  /** Any host, none, or only these hosts. */
  network: "none" | "any" | readonly string[];
}

/** A program for an `Executor` to run, and what it may touch. */
export interface ExecRequest {
  /** The program: looked up on the PATH of the environment it runs with, unless a path. */
  command: string;
  /** Its arguments, as they are: no shell comes in between. */
  args: readonly string[];
  /** Its working folder. Not given: a new, empty temporary folder, removed when the run ends. */
  cwd?: string;
  /**
   * Added to (and overriding) the environment programs get here: the login
   * shell's, on the desktop, without the variables whose names look like
   * secrets (see `looksLikeSecret` in ./execution).
   */
  env: Readonly<Record<string, string>>;
  /**
   * What it may read, write and reach. Enforced from "os" up; at "none" it is
   * only declared, for approvals (see `declaredAccess` in ./execution).
   */
  allow: ExecAllow;
  /** How long it may run before it is stopped, with every process it started. */
  timeoutMs: number;
  /** How much of each output is kept: the start of its standard output, the end of its error output. */
  maxOutputBytes: number;
  /** Stops it at once, with every process it started (e.g. the User stops the Answer). */
  signal: AbortSignal;
}

/** How a program an `Executor` ran ended, and what it wrote (see `SkillScriptRun`, which adds `error`). */
export interface ExecResult {
  /** Its exit code; null when it was stopped (its timeout, or its signal). */
  exitCode: number | null;
  /** It ran longer than its timeout, so it was stopped with every process it started. */
  timedOut: boolean;
  /** The start of its standard output, up to `maxOutputBytes`; a character cut in half is left out. */
  stdout: string;
  /** The end of its error output, up to `maxOutputBytes`; a character cut in half is left out. */
  stderr: string;
  /** It wrote more to its standard output than is kept. */
  stdoutTruncated: boolean;
  /** It wrote more to its error output than is kept. */
  stderrTruncated: boolean;
}

/**
 * Runs the programs Tools start (Skill scripts now; later a shell or a
 * converter), as confined as this host can. The core never starts a Tool's
 * process itself, so a sandbox can come later without changing the Tools.
 * The core defaults to the local one at level "none" (./execution); the
 * desktop app gives it the OS sandbox's, at "os", where that can start
 * (src/main/sandbox.ts). Connectors' own server processes don't come here:
 * they use `ProcessLauncher`.
 */
export interface Executor {
  readonly level: SandboxLevel;
  /**
   * Runs a program to its end, with no input, and resolves with how it ended
   * and what it wrote. Rejects if it can't start: a command that isn't found
   * rejects with an error whose `code` is "ENOENT", and a signal stopped
   * already with its reason, before anything runs.
   */
  run(request: ExecRequest): Promise<ExecResult>;
}

/**
 * What Skill scripts run with (see `src/core/skills/scripts.ts`). Python and
 * bash come from the User's login-shell PATH; JavaScript runs on the app's
 * own Node.js, so the User needn't install one.
 */
export interface ScriptRuntimes {
  /**
   * The Node.js that runs JavaScript scripts. Defaults to the one running the
   * core (`process.execPath`) with `ELECTRON_RUN_AS_NODE=1`: in the desktop
   * app that is Electron's bundled Node, running as plain Node.
   */
  node?: { command: string; env?: Readonly<Record<string, string>> };
  /**
   * The OS whose interpreters are used: `python` instead of `python3`, and no
   * shell scripts, on Windows. Defaults to `process.platform`; tests pretend.
   */
  platform?: NodeJS.Platform;
}

/** The built-in embedding model's downloaded files, as the core hands them to an `Embedder`. */
export interface EmbeddingModelFiles {
  /** The ONNX model, an absolute path. */
  model: string;
  /** The tokenizer's `tokenizer.json` and `tokenizer_config.json`, absolute paths. */
  tokenizer: string;
  tokenizerConfig: string;
  /** Texts are cut to this many tokens, special tokens included. */
  maxTokens: number;
}

/**
 * Runs the built-in embedding model, one text at a time, off the core's
 * thread: the desktop app runs it in an Electron utility process, and tests
 * pass a deterministic fake. The core downloads and checks the files first,
 * and adds the model's "passage: " or "query: " prefix to each text.
 */
export interface Embedder {
  /**
   * Loads the model, if it isn't loaded already: call before `embed`, and
   * again after `embed` fails (e.g. the process running it crashed). Rejects
   * if the model can't start.
   */
  load(files: EmbeddingModelFiles): Promise<void>;
  /** The text's vector. Rejects if the model isn't loaded or fails on it. */
  embed(text: string): Promise<Float32Array>;
  /** Stops the model and frees its memory. Loading again starts it afresh. */
  close(): void;
}

/** The built-in reranking model's downloaded files: the same kinds as the embedding model's. */
export type RerankingModelFiles = EmbeddingModelFiles;

/**
 * Runs the built-in reranking model, a cross-encoder, off the core's thread:
 * the desktop app runs it in an Electron utility process of its own, and tests
 * pass a deterministic fake. The core downloads and checks the files first.
 */
export interface CrossEncoder {
  /** Loads the model, if it isn't loaded already. Rejects if the model can't start. */
  load(files: RerankingModelFiles): Promise<void>;
  /**
   * How well each text answers the query, in the order given: the model's own
   * scores, higher is better. Rejects if the model isn't loaded or fails.
   */
  score(query: string, texts: readonly string[]): Promise<number[]>;
  /** Stops the model and frees its memory. Loading again starts it afresh. */
  close(): void;
}

/** One file of the built-in embedding model, as recorded when the app was built. */
export interface ModelFile {
  /** Relative to the source's base URL, and to the model's folder in the data folder. */
  path: string;
  size: number;
  /** SHA-256 of the file, in hex. */
  sha256: string;
}

/** Where the built-in embedding model's files are downloaded from. */
export interface EmbeddingModelSource {
  /** Ends with "/"; each file's path is resolved against it. */
  baseUrl: string;
  files: readonly ModelFile[];
}

/**
 * Sends crash reports, scrubbed of the User's content, to IncarnaMind's
 * developers. The desktop app passes one only when it was built with a
 * crash-report address (a Sentry DSN). The core turns it on only once the User
 * has opted in, at startup or when they opt in, and off the moment they opt out.
 */
export interface CrashReporter {
  /** Starts reporting, or stops it at once. Must not throw: a reporter that can't start says so in the log. */
  setEnabled(enabled: boolean): void;
}

/** One usage event, checked against the catalog (./usageEvents), ready to send. */
export interface UsageEventMessage {
  /** The install's random ID, which the User can reset. */
  installId: string;
  event: UsageEventName;
  /** The event's fields, then the common fields (`COMMON_FIELDS`). Nothing else. */
  properties: Readonly<Record<string, UsageValue>>;
}

/**
 * Sends usage data (#187): anonymous product events, never the User's files,
 * Questions or Answers. The desktop app passes one only when it was built with
 * an analytics project (PostHog); the core turns it on only while the User
 * agrees, and off the moment they don't.
 */
export interface UsageDataSender {
  /** Where events go, for the Privacy page. */
  readonly service: ExternalService;
  /**
   * A test build (the alpha): usage data is on until the User turns it off,
   * as testers agree to when they join, and a first-run notice says so.
   * Otherwise it is off until the User agrees.
   */
  readonly testerBuild: boolean;
  /** This build's version, sent with every event. */
  readonly appVersion: string;
  /** Starts sending, or stops at once and drops everything queued. Must not throw. */
  setEnabled(enabled: boolean): void;
  /** Queues an event. Called only while sending is on. Must not throw, offline or not. */
  capture(message: UsageEventMessage): void;
}

export interface CoreAdapters {
  paths: Paths;
  /** The OS's preferred languages, most preferred first, as BCP 47 tags such as "zh-Hans-CN". */
  systemLanguages(): readonly string[];
  keychain: Keychain;
  browser: Browser;
  /**
   * Opens Documents' files in other apps and shows them in the file manager.
   * Absent (tests that don't need it): both are refused.
   */
  shell?: FileShell;
  /** How Linked folders are watched (see `LinkedFolderOptions`). */
  linkedFolders?: LinkedFolderOptions;
  processes: ProcessLauncher;
  /**
   * Runs the programs Tools start, such as Skill scripts (see `Executor`).
   * Defaults to the local one at sandbox level "none", starting them through
   * `processes`, with working folders in `paths.tempDir`. The desktop app
   * passes the OS sandbox's where it can start.
   */
  executor?: Executor;
  /** What Skill scripts run with (see `ScriptRuntimes`); the defaults suit the desktop app. */
  scriptRuntimes?: ScriptRuntimes;
  /** Runs the built-in embedding model (see `Embedder`). */
  embedder: Embedder;
  /**
   * Where the built-in embedding model's files come from. Defaults to the
   * pinned Hugging Face revision; tests point it at a local server, or give no
   * files for a fake embedder that needs none.
   */
  embeddingModelSource?: EmbeddingModelSource;
  /** Runs the built-in reranking model (see `CrossEncoder`), when the User turns it on. */
  crossEncoder: CrossEncoder;
  /**
   * Where the built-in reranking model's files come from. Defaults to the
   * pinned Hugging Face revision; tests give no files for a fake that needs none.
   */
  rerankingModelSource?: EmbeddingModelSource;
  /** Defaults to the system clock. Tests may pass a fake one. */
  now?: () => Date;
  /**
   * Builds chat models from provider settings. Defaults to the AI SDK
   * providers; tests pass AI SDK mock models.
   */
  createChatModel?: ChatModelFactory;
  /**
   * Looks up models in Ollama (`/api/tags`, `/api/show`) for the settings
   * their requests carry and how they cite. Defaults to asking Ollama, with
   * this computer's memory; tests pass a stub.
   */
  ollamaModels?: OllamaModels;
  /**
   * Where the experimental ChatGPT plan provider signs in and sends Questions.
   * Defaults to OpenAI's servers and the Codex CLI's callback port; tests
   * point it at a local fake authorization server and endpoint.
   */
  chatGptPlan?: Partial<ChatGptPlanEndpoints>;
  /**
   * How long a remote Connector's browser sign-in waits for the User.
   * Defaults to five minutes; tests shorten it.
   */
  connectorSignInTimeoutMs?: number;
  /**
   * Where requests to TypeSafe's hosted Jev go. Defaults to
   * https://api.typesafe.ai; tests point it at a local fake server. Consent
   * still treats them as going to TypeSafe.
   */
  jevHostedUrl?: string;
  /**
   * Turns Question context into a streamed Answer. Defaults to the AI SDK
   * engine; an alternative agent layer plugs in here.
   */
  answerEngine?: AnswerEngine;
  /**
   * Runs the Tool-calling loop of each Run, such as an Answer's (see
   * `RunEngine`). Defaults to the one on AI SDK 7; another loop library
   * plugs in here, and must pass the same contract suite.
   */
  runEngine?: RunEngine;
  /**
   * Builds embedding models for the providers the User can choose instead of
   * the built-in model. Defaults to the AI SDK providers; tests pass AI SDK
   * mock models.
   */
  createEmbeddingModel?: EmbeddingModelFactory;
  /**
   * Builds Cohere and Voyage reranking models. Defaults to the AI SDK
   * providers; tests pass AI SDK mock models.
   */
  createRerankingModel?: RerankingModelFactory;
  /**
   * Reorders the document-search Tool's hybrid hits before they are grouped,
   * replacing the rerank the User sets up, built in or with a Cohere or
   * Voyage key (an alternative search layer plugs in here). None by default.
   */
  reranker?: Reranker;
  /**
   * Sends crash reports once the User opts in (see `CrashReporter`). Absent
   * when this copy can't send any: Settings then doesn't offer them.
   */
  crashReporter?: CrashReporter;
  /**
   * Sends usage data while the User agrees (see `UsageDataSender`). Absent
   * when this copy can't send any: nothing is sent or asked, and Settings
   * says IncarnaMind collects no usage data.
   */
  usageData?: UsageDataSender;
  /** Where the core logs what happens (see `Logger`). None by default. */
  log?: Logger;
}
