/**
 * Capabilities the host passes into the core when it constructs it.
 *
 * The desktop app builds these from Electron in `src/main/platform.ts`; tests
 * build fakes; a future hosted server would pass its own. The core never
 * imports Electron (ADR-0004, ADR-0006).
 */
import type { ChildProcess } from "node:child_process";
import type { SecretProtection } from "./api";
import type { ChatGptPlanEndpoints } from "./providers/chatgpt/plan";
import type { ChatModelFactory } from "./providers/models";

export interface Paths {
  /**
   * The app data folder. It holds the SQLite database, the secrets file and, in
   * later tickets, Document files, Skills, the embedding model and logs.
   * Backing up means copying this one folder.
   */
  dataDir: string;
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

export interface SpawnOptions {
  cwd?: string;
  env?: Readonly<Record<string, string>>;
}

/**
 * Starts child processes with the User's login-shell environment, so `npx` and
 * `uvx` resolve even when the app was opened from the Dock or Start menu.
 * Used by Connectors and Skill scripts in later tickets.
 */
export interface ProcessLauncher {
  spawn(command: string, args: readonly string[], options?: SpawnOptions): ChildProcess;
}

export interface CoreAdapters {
  paths: Paths;
  /** The OS's preferred languages, most preferred first, as BCP 47 tags such as "zh-Hans-CN". */
  systemLanguages(): readonly string[];
  keychain: Keychain;
  browser: Browser;
  processes: ProcessLauncher;
  /** Defaults to the system clock. Tests may pass a fake one. */
  now?: () => Date;
  /**
   * Builds chat models from provider settings. Defaults to the AI SDK
   * providers; tests pass AI SDK mock models.
   */
  createChatModel?: ChatModelFactory;
  /**
   * Where the experimental ChatGPT plan provider signs in and sends Questions.
   * Defaults to OpenAI's servers and the Codex CLI's callback port; tests
   * point it at a local fake authorization server and endpoint.
   */
  chatGptPlan?: Partial<ChatGptPlanEndpoints>;
}
