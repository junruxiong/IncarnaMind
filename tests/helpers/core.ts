import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { onTestFinished } from "vitest";
import {
  CHOSEN_PROVIDER_FLOWS,
  type ChatProvider,
  type Core,
  type CoreAdapters,
  type CoreEventName,
  type CoreEvents,
  createCore,
  DATABASE_FILE,
  type Keychain,
  type SaveChatProviderInput,
  type SecretProtection,
} from "../../src/core";
import { createFakeEmbedder } from "../../src/core/embedding/fake";
import { createFakeCrossEncoder } from "../../src/core/reranking/fake";
import { openDatabase, type SqlValue } from "../../src/core/storage";

/** The citation-style engine, styles and locales the app ships (ADR-0016). */
export const CITATION_STYLES = fileURLToPath(
  new URL("../../resources/citation-styles", import.meta.url),
);

/** A model source with nothing to download: the fake embedding model needs no files. */
export const NO_MODEL_FILES = { baseUrl: "http://127.0.0.1/", files: [] } as const;

/** An address nothing answers at (the discard port), for services a test doesn't stand in for. */
const NOWHERE = "http://127.0.0.1:9";

/** A fresh, empty data folder, deleted when the current test finishes. */
export async function createTempDataFolder(): Promise<string> {
  const dataDir = await mkdtemp(join(tmpdir(), "incarnamind-test-"));
  onTestFinished(() => rm(dataDir, { recursive: true, force: true }));
  return dataDir;
}

export interface MemoryKeychain extends Keychain {
  /** What is stored, for assertions. */
  readonly secrets: ReadonlyMap<string, string>;
}

/**
 * An in-memory stand-in for the OS keychain. Like the real one, it refuses to
 * store secrets when `protection` is "unavailable", or "plain-text" until
 * `allowPlainText` is called (Linux without a keyring).
 */
export function createMemoryKeychain(protection: SecretProtection = "os"): MemoryKeychain {
  const secrets = new Map<string, string>();
  let plainTextAllowed = false;
  const assertProtected = () => {
    if (protection === "unavailable" || (protection === "plain-text" && !plainTextAllowed)) {
      throw new Error(`The fake keychain refuses to store secrets (${protection}).`);
    }
  };
  return {
    secrets,
    protection: () => protection,
    allowPlainText: () => {
      plainTextAllowed = true;
    },
    get: async (name) => secrets.get(name) ?? null,
    set: async (name, secret) => {
      assertProtected();
      secrets.set(name, secret);
    },
    delete: async (name) => {
      secrets.delete(name);
    },
  };
}

/**
 * Starts the core on `dataDir` with test adapters: an English OS, an in-memory
 * keychain, no browser, shell or processes, the deterministic fake embedding
 * and reranking models, which have no files to download, Linked folders that settle quickly,
 * and no lookups of models in Ollama. Closed when the current test finishes; call `close()` yourself to
 * simulate quitting the app.
 */
export function startCore(dataDir: string, overrides: Partial<CoreAdapters> = {}): Core {
  const core = createCore({
    // A Documents folder of its own, so a test that saves a file never writes into the User's.
    paths: { dataDir, documents: join(dataDir, "Documents"), citationStyles: CITATION_STYLES },
    systemLanguages: () => ["en-US"],
    systemUserName: () => "Priya Raman",
    keychain: createMemoryKeychain(),
    browser: {
      open: async () => {
        throw new Error("Tests can't open a browser.");
      },
    },
    processes: {
      spawn: async () => {
        throw new Error("This test didn't provide a process launcher.");
      },
    },
    createChatModel: () => {
      throw new Error("This test didn't provide a chat model.");
    },
    // Nothing asks the Ollama on this computer: it runs, and its models get the default settings.
    ollamaModels: {
      describe: async () => null,
      loaded: async () => null,
      running: async () => true,
    },
    embedder: createFakeEmbedder(),
    embeddingModelSource: NO_MODEL_FILES,
    crossEncoder: createFakeCrossEncoder(),
    rerankingModelSource: NO_MODEL_FILES,
    // Find sources never reaches the real services from a test: a test that searches gives its stand-in.
    sourceProviders: {
      openalex: NOWHERE,
      arxiv: NOWHERE,
      pubmed: NOWHERE,
      crossref: NOWHERE,
      openfda: NOWHERE,
      dailymed: NOWHERE,
      clinicaltrials: NOWHERE,
      ecfr: NOWHERE,
      govinfo: NOWHERE,
    },
    ...overrides,
    // Linked folders settle and retry quickly in tests.
    linkedFolders: { settleMs: 30, retryMs: 300, ...overrides.linkedFolders },
  });
  onTestFinished(() => core.close());
  return core;
}

/**
 * Reads the data folder's database directly, for checking how things are
 * stored (e.g. that a delete is soft). Everything else goes through the core.
 */
export function queryDatabase<Row>(
  dataDir: string,
  sql: string,
  params: readonly SqlValue[] = [],
): Row[] {
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    return db.all<Row>(sql, params);
  } finally {
    db.close();
  }
}

/** A clock that only moves when the test moves it. */
export function manualClock(start = "2026-10-06T09:00:00.000Z") {
  let time = Date.parse(start);
  return {
    now: () => new Date(time),
    advance(ms: number) {
      time += ms;
    },
  };
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

/**
 * Returns on the next turn of the event loop: by then every promise chain that needs no timer,
 * thread or process has run to its end. For checking what such work did or didn't do.
 */
export const nextTurn = () => new Promise<void>((resolve) => setImmediate(resolve));

/** Resolves with the payload of the next `event` the core emits. */
export function nextEvent<E extends CoreEventName>(core: Core, event: E): Promise<CoreEvents[E]> {
  return new Promise((resolve) => {
    const stop = core.on(event, (payload) => {
      stop();
      resolve(payload);
    });
  });
}

/**
 * Saves a chat provider as it was left when choosing it wasn't yet the
 * consent (ADR-0022): nothing allowed or declined, so each flow's first send asks.
 */
export async function saveProviderUnasked(
  core: Core,
  input: SaveChatProviderInput,
): Promise<ChatProvider> {
  const provider = await core.saveChatProvider(input);
  const service = (await core.listChatProviders()).find((each) => each.id === provider.id)?.service;
  if (service) {
    for (const flow of CHOSEN_PROVIDER_FLOWS) await core.revokeConsent(flow, service.id);
  }
  return provider;
}
