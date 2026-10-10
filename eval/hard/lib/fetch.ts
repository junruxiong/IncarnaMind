/**
 * Fetches the hard tier's library into the evaluation's cache
 * (`<cache>/hard/`), once: a file already there with the right SHA-256 isn't
 * fetched again. Each file is checked against the SHA-256 the manifest
 * records; one that has changed at its source, or can't be fetched, is left
 * out of the run and reported, with the Questions about it, rather than
 * failing the run.
 *
 * Requests name no person: the User-Agent is the evaluation's own, with no
 * contact details, and nothing else identifies anyone. Sources whose terms
 * require contact details (SEC EDGAR's fair-access policy, Wikimedia's API)
 * aren't in the library. Each host gets at most one request at a time, and
 * arXiv one every 3 seconds, as its terms ask.
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { ZipArchive } from "../../../src/core/documents/formats/zip";
import type { Log } from "../../lib/log";
import { cachePathOf, type HardArchive, type HardDocument, type Manifest } from "./manifest";

/** The only identification any request carries. */
export const USER_AGENT = "IncarnaMind document evaluation";

/** The pause between two requests to one host, in milliseconds. */
export function pauseFor(host: string): number {
  return host === "arxiv.org" || host.endsWith(".arxiv.org") ? 3000 : 500;
}

/** The largest file the library fetches: anything bigger isn't what the manifest describes. */
const MAX_FILE_BYTES = 200_000_000;

export type FetchStatus = "ready" | "changed" | "unavailable";

export interface FetchedDocument {
  key: string;
  /** The file to add; null unless it is ready. */
  path: string | null;
  status: FetchStatus;
  /** Why it isn't ready. */
  reason?: string;
}

export interface FetchResult {
  documents: FetchedDocument[];
  /** Bytes fetched by this run: none when the cache had everything. */
  fetchedBytes: number;
  seconds: number;
}

export interface FetchOptions {
  root: string;
  cacheDir: string;
  log: Log;
  /** For tests: the network. */
  fetch?: typeof globalThis.fetch;
  /** For tests: waiting between requests. */
  sleep?: (ms: number) => Promise<void>;
}

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

const sleepFor = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** Writes a file whole or not at all. */
async function writeAtomically(path: string, bytes: Uint8Array): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const partial = `${path}.partial`;
  await writeFile(partial, bytes);
  await rename(partial, path);
}

/** The file's bytes if it is there with this hash; null otherwise. */
async function cached(path: string, hash: string): Promise<Uint8Array | null> {
  if (!existsSync(path)) return null;
  const bytes = await readFile(path);
  return sha256(bytes) === hash ? bytes : null;
}

/** Fetches with the evaluation's User-Agent, one request per host at a time, paced. */
function createFetcher(options: FetchOptions) {
  const fetch = options.fetch ?? globalThis.fetch;
  const sleep = options.sleep ?? sleepFor;
  const last = new Map<string, number>();
  return async (url: string): Promise<Uint8Array> => {
    const host = new URL(url).hostname;
    const wait = (last.get(host) ?? 0) + pauseFor(host) - Date.now();
    if (wait > 0) await sleep(wait);
    try {
      let lastError: unknown;
      for (let attempt = 1; attempt <= 3; attempt++) {
        try {
          const response = await fetch(url, {
            headers: { "User-Agent": USER_AGENT },
            redirect: "follow",
            signal: AbortSignal.timeout(300_000),
          });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const length = Number(response.headers.get("content-length") ?? 0);
          if (length > MAX_FILE_BYTES) throw new Error(`${length} bytes is too large`);
          const bytes = new Uint8Array(await response.arrayBuffer());
          if (bytes.byteLength > MAX_FILE_BYTES) throw new Error("too large");
          return bytes;
        } catch (error) {
          lastError = error;
          // A missing or refused file won't come back on a second try.
          if (error instanceof Error && /^HTTP 4\d\d/.test(error.message)) break;
          if (attempt < 3) await sleep(attempt * 2000);
        }
      }
      throw lastError;
    } finally {
      last.set(host, Date.now());
    }
  };
}

const describe = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * Makes sure every Document is in the cache, checked, and says where each is.
 * Archives are fetched only when a Document in them is missing.
 */
export async function fetchLibrary(
  manifest: Manifest,
  options: FetchOptions,
): Promise<FetchResult> {
  const started = Date.now();
  const get = createFetcher(options);
  const { log } = options;
  let fetchedBytes = 0;
  const results = new Map<string, FetchedDocument>();
  const ready = (document: HardDocument, path: string) =>
    results.set(document.key, { key: document.key, path, status: "ready" });
  const notReady = (document: HardDocument, status: FetchStatus, reason: string) => {
    results.set(document.key, { key: document.key, path: null, status, reason });
    log(`Left out ${document.key}: ${reason}`);
  };

  // In the repository: checked, never fetched.
  for (const document of manifest.documents.filter((each) => each.path)) {
    const path = join(options.root, document.path as string);
    if (await cached(path, document.sha256)) ready(document, path);
    else notReady(document, "changed", `${document.path} isn't the version recorded.`);
  }

  // From archives: each archive fetched (or read from the cache) once, if a member is missing.
  const archived = manifest.documents.filter((each) => each.archive);
  for (const archive of manifest.archives) {
    const members = archived.filter((document) => document.archive === archive.key);
    const missing: HardDocument[] = [];
    for (const document of members) {
      const path = cachePathOf(options.cacheDir, document);
      if (await cached(path, document.sha256)) ready(document, path);
      else missing.push(document);
    }
    if (missing.length === 0) continue;
    const zip = await openArchive(archive, options, get, (bytes) => {
      fetchedBytes += bytes;
    });
    if (typeof zip === "string") {
      for (const document of missing) notReady(document, "unavailable", zip);
      continue;
    }
    for (const document of missing) {
      try {
        const bytes = await zip.read(document.member as string, 100_000_000);
        if (!bytes) {
          notReady(document, "changed", `${document.member} isn't in ${archive.key}.`);
        } else if (sha256(bytes) !== document.sha256) {
          notReady(document, "changed", `${document.member} isn't the version recorded.`);
        } else {
          const path = cachePathOf(options.cacheDir, document);
          await writeAtomically(path, bytes);
          ready(document, path);
        }
      } catch (error) {
        notReady(document, "unavailable", describe(error));
      }
    }
  }

  // From their URLs, one at a time.
  const remote = manifest.documents.filter((each) => each.url);
  let done = 0;
  for (const document of remote) {
    done++;
    const path = cachePathOf(options.cacheDir, document);
    if (await cached(path, document.sha256)) {
      ready(document, path);
      continue;
    }
    try {
      const bytes = await get(document.url as string);
      fetchedBytes += bytes.byteLength;
      if (sha256(bytes) !== document.sha256) {
        notReady(
          document,
          "changed",
          `the file at ${document.url} isn't the version recorded (it has changed at its source).`,
        );
        continue;
      }
      await writeAtomically(path, bytes);
      ready(document, path);
      if (done % 25 === 0) log(`Fetched ${done} of ${remote.length} Documents`);
    } catch (error) {
      notReady(document, "unavailable", `${document.url}: ${describe(error)}`);
    }
  }

  return {
    documents: manifest.documents.map(
      (document) =>
        results.get(document.key) ?? {
          key: document.key,
          path: null,
          status: "unavailable",
          reason: "not fetched",
        },
    ),
    fetchedBytes,
    seconds: (Date.now() - started) / 1000,
  };
}

/** An archive from the cache, or fetched into it; a reason when it can't be had. */
async function openArchive(
  archive: HardArchive,
  options: FetchOptions,
  get: (url: string) => Promise<Uint8Array>,
  counted: (bytes: number) => void,
): Promise<ZipArchive | string> {
  const path = join(options.cacheDir, "hard", "archives", `${archive.key}.zip`);
  let bytes = await cached(path, archive.sha256);
  if (!bytes) {
    options.log(`Fetching ${archive.key} (${Math.round(archive.bytes / 1e6)} MB)`);
    try {
      bytes = await get(archive.url);
      counted(bytes.byteLength);
    } catch (error) {
      return `${archive.url}: ${describe(error)}`;
    }
    if (sha256(bytes) !== archive.sha256) {
      return `the archive at ${archive.url} isn't the version recorded.`;
    }
    await writeAtomically(path, bytes);
  }
  try {
    return new ZipArchive(bytes);
  } catch (error) {
    return `${archive.key}: ${describe(error)}`;
  }
}
