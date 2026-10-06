/**
 * Capabilities the host passes into the core when it constructs it.
 *
 * The desktop app builds these from Electron in `src/main/platform.ts`; tests
 * build fakes; a future hosted server would pass its own. The core never
 * imports Electron (ADR-0004, ADR-0006).
 */
import type { ChildProcess } from "node:child_process";

export interface Paths {
  /**
   * The app data folder. It holds the SQLite database and, in later tickets,
   * Document files, Skills, the embedding model and logs. Backing up means
   * copying this one folder.
   */
  dataDir: string;
}

/**
 * Secrets (API keys, OAuth tokens) stay on this device, outside the database
 * (ADR-0003). The desktop app backs this with Electron `safeStorage`.
 */
export interface Keychain {
  get(name: string): Promise<string | null>;
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
}
