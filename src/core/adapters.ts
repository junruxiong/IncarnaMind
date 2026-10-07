/**
 * Capabilities the host passes into the core when it constructs it.
 *
 * The desktop app builds these from Electron in `src/main/platform.ts`; tests
 * build fakes; a future hosted server would pass its own. The core never
 * imports Electron (ADR-0004, ADR-0006).
 */
import type { ChildProcess } from "node:child_process";
import type { AnswerEngine } from "./answers/engine";
import type { SecretProtection } from "./api";
import type { Reranker } from "./documents/searchTool";
import type { ChatGptPlanEndpoints } from "./providers/chatgpt/plan";
import type { EmbeddingModelFactory } from "./providers/embeddings";
import type { ChatModelFactory } from "./providers/models";
import type { RerankingModelFactory } from "./providers/rerank";

export interface Paths {
  /**
   * The app data folder. It holds the SQLite database, the secrets file and, in
   * later tickets, Document files, Skills, the embedding model (`models/`) and logs.
   * Backing up means copying this one folder.
   */
  dataDir: string;
  /**
   * The built-in Skills the app ships, one SKILL.md folder each, named like
   * the Skill: `resources/skills/` in the repository, copied into the packaged
   * app's resources. The core installs them into the data folder at startup.
   * Not given: no built-in Skills (tests that aren't about them).
   */
  builtInSkills?: string;
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
  /** Added to (and overriding) the login-shell environment. */
  env?: Readonly<Record<string, string>>;
}

/**
 * Starts child processes with the User's login-shell environment, so `npx` and
 * `uvx` resolve even when the app was opened from the Dock or Start menu. The
 * command is looked up on that environment's PATH. Used by local Connectors,
 * and by Skill scripts in a later ticket.
 */
export interface ProcessLauncher {
  /**
   * Resolves once the process is running, with its standard input, output and
   * error piped. Rejects if it can't start: a command that isn't found rejects
   * with an error whose `code` is "ENOENT".
   */
  spawn(command: string, args: readonly string[], options?: SpawnOptions): Promise<ChildProcess>;
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

export interface CoreAdapters {
  paths: Paths;
  /** The OS's preferred languages, most preferred first, as BCP 47 tags such as "zh-Hans-CN". */
  systemLanguages(): readonly string[];
  keychain: Keychain;
  browser: Browser;
  processes: ProcessLauncher;
  /** Runs the built-in embedding model (see `Embedder`). */
  embedder: Embedder;
  /**
   * Where the built-in embedding model's files come from. Defaults to the
   * pinned Hugging Face revision; tests point it at a local server, or give no
   * files for a fake embedder that needs none.
   */
  embeddingModelSource?: EmbeddingModelSource;
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
   * replacing the rerank the User sets up with a Cohere or Voyage key (an
   * alternative search layer plugs in here). None by default.
   */
  reranker?: Reranker;
}
