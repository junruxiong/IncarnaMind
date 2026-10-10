/**
 * Web pages saved as Documents (#261): the snapshot, in the data folder's
 * "Web pages", is the Document's file and is indexed like any file. "Fetch
 * again" writes over it, so a changed page is a new version (ADR-0010).
 */
import { randomUUID } from "node:crypto";
import { mkdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { normaliseAddress } from "../../../shared/webAddresses";
import type { Document, SaveWebPageInput, WebPageResult } from "../../api";
import { InvalidInputError, isRecord, NotFoundError } from "../../errors";
import type { Database } from "../../storage";
import { hashBytes } from "../files";
import { WebPageError } from "./errors";
import { type FetchLimits, fetchPage } from "./fetch";
import { type Snapshot, snapshotOf } from "./snapshot";
import { documentAtAddress, recordVersion, type WebPageVersion, webPageInfo } from "./store";

export { WebPageError } from "./errors";

export interface SnapshotFile {
  path: string;
  contentHash: string;
  size: number;
  mtimeMs: number;
}

export interface WebPagesOptions {
  db: Database;
  dataDir: string;
  now: () => string;
  documents: {
    get(id: string): Document;
    addSnapshot(id: string, name: string, file: SnapshotFile, record: () => void): Document;
    check(path: string): Promise<void>;
  };
  /** Most preferred first, as Accept-Language lists them. */
  languages(): readonly string[];
  fetch?: typeof fetch;
  limits?: FetchLimits;
}

function parseSaveInput(input: unknown): SaveWebPageInput {
  if (!isRecord(input) || typeof input.address !== "string") {
    throw new InvalidInputError("saveWebPage expects an address.");
  }
  const { address, html, folderId } = input;
  if (html !== undefined && typeof html !== "string") {
    throw new InvalidInputError("A web page's HTML must be text.");
  }
  if (folderId !== undefined && folderId !== null && typeof folderId !== "string") {
    throw new InvalidInputError("A Folder id must be text, or null.");
  }
  return { address, ...(html === undefined ? {} : { html }), folderId: folderId ?? null };
}

export function createWebPages(options: WebPagesOptions) {
  const { db, now, documents } = options;
  const folder = join(options.dataDir, "Web pages");
  /** Asked twice at once, a save or a fetch is done once. */
  const underWay = new Map<string, Promise<WebPageResult>>();

  const once = (key: string, task: () => Promise<WebPageResult>): Promise<WebPageResult> => {
    const running = underWay.get(key);
    if (running) return running;
    const started = task().finally(() => underWay.delete(key));
    underWay.set(key, started);
    return started;
  };

  const failed = (address: string, error: unknown): WebPageResult => {
    if (error instanceof WebPageError)
      return { outcome: "failed", address, failure: error.failure };
    throw error;
  };

  async function snapshotFor(address: string, html: string | undefined): Promise<Snapshot> {
    const page =
      html === undefined
        ? await fetchPage(address, {
            fetch: options.fetch,
            limits: options.limits,
            languages: options.languages(),
          })
        : { html, url: address };
    return snapshotOf(page.html, page.url, address);
  }

  /** Writes a snapshot over `path`, whole or not at all. */
  async function write(path: string, xhtml: string): Promise<SnapshotFile> {
    await mkdir(folder, { recursive: true });
    const bytes = Buffer.from(xhtml, "utf8");
    const partial = `${path}.${randomUUID()}.part`;
    try {
      await writeFile(partial, bytes);
      await rename(partial, path);
    } catch (error) {
      await rm(partial, { force: true });
      throw error;
    }
    const { mtimeMs } = await stat(path);
    return { path, contentHash: hashBytes(bytes), size: bytes.byteLength, mtimeMs };
  }

  const version = (
    documentId: string,
    file: SnapshotFile,
    address: string,
    snapshot: Snapshot,
  ): WebPageVersion => ({
    documentId,
    contentHash: file.contentHash,
    address,
    siteName: snapshot.siteName,
    author: snapshot.author,
    published: snapshot.published,
    fetchedAt: now(),
  });

  return {
    /** See `CoreApi.saveWebPage`; the caller files it in its Folder. */
    save(input: unknown): Promise<WebPageResult> {
      const { address: typed, html } = parseSaveInput(input);
      const address = normaliseAddress(typed);
      if (!address) {
        return Promise.resolve(
          failed(typed, new WebPageError("invalid-address", "Not an http or https address.")),
        );
      }
      const saved = (id: string): WebPageResult => ({
        outcome: "saved",
        document: documents.get(id),
        existing: true,
      });
      const known = documentAtAddress(db, address);
      if (known) return Promise.resolve(saved(known));
      return once(address, async () => {
        try {
          const snapshot = await snapshotFor(address, html);
          const meanwhile = documentAtAddress(db, address);
          if (meanwhile) return saved(meanwhile);
          const id = randomUUID();
          const file = await write(join(folder, `${id}.html`), snapshot.xhtml);
          const document = documents.addSnapshot(id, snapshot.title, file, () =>
            recordVersion(db, version(id, file, address, snapshot), now()),
          );
          return { outcome: "saved", document, existing: false };
        } catch (error) {
          return failed(address, error);
        }
      });
    },

    /** See `CoreApi.fetchWebPageAgain`. */
    fetchAgain(documentId: unknown): Promise<WebPageResult> {
      if (typeof documentId !== "string" || documentId === "") {
        throw new InvalidInputError("A Document id is required.");
      }
      const document = documents.get(documentId);
      const info = webPageInfo(db, document.id);
      if (document.kind !== "web" || !info) {
        throw new InvalidInputError("Only a web page can be fetched again.");
      }
      return once(`document:${document.id}`, async () => {
        try {
          const snapshot = await snapshotFor(info.address, undefined);
          const current = documents.get(document.id);
          const file = await write(current.path, snapshot.xhtml);
          recordVersion(db, version(current.id, file, info.address, snapshot), now());
          await documents.check(current.path);
          return { outcome: "saved", document: documents.get(current.id), existing: true };
        } catch (error) {
          if (error instanceof NotFoundError) throw error;
          return failed(info.address, error);
        }
      });
    },
  };
}

export type WebPages = ReturnType<typeof createWebPages>;
