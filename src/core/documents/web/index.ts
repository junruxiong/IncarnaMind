/**
 * Web pages saved as Documents (#261): the snapshot, in the data folder's
 * "Web pages", is the Document's file and is indexed like any file. "Fetch
 * again" writes over it, so a changed page is a new version (ADR-0010).
 *
 * The article's pictures are saved with it (#756): fetched once through the
 * fetcher that reaches only the public internet, kept in "Web pages/pictures"
 * as files named by their content, so versions share them and none is
 * written over, and shown by the snapshot from there.
 */
import { randomUUID } from "node:crypto";
import { access, mkdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { normaliseAddress } from "../../../shared/webAddresses";
import type { Document, SaveWebPageInput, WebPageResult } from "../../api";
import { InvalidInputError, isRecord, NotFoundError } from "../../errors";
import type { Database } from "../../storage";
import { hashBytes } from "../files";
import { WebPageError } from "./errors";
import { type FetchLimits, fetchPage, type PageFetch } from "./fetch";
import { type FetchedPicture, fetchPictures, type PictureLimits } from "./pictures";
import { type ShownPicture, type Snapshot, snapshotOf, withPictures } from "./snapshot";
import {
  documentAtAddress,
  picturesOnlyPurgedShow,
  recordPictures,
  recordVersion,
  type WebPagePicture,
  type WebPageVersion,
  webPageInfo,
} from "./store";

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
  fetch?: PageFetch;
  limits?: FetchLimits;
  /** Fetches the pictures: one that reaches only the public internet, whatever fetched the page. */
  pictureFetch: PageFetch;
  pictureLimits?: PictureLimits;
  reportError?: (error: unknown) => void;
}

/** Where a picture's file is, as the snapshot names it beside itself. */
const pictureName = (picture: { pictureHash: string; mediaType: string }) =>
  `${picture.pictureHash}.${PICTURE_EXTENSIONS[picture.mediaType] ?? "bin"}`;

const PICTURE_EXTENSIONS: Readonly<Record<string, string>> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/avif": "avif",
};

/** A page ready to write: its snapshot with its pictures in place, and the pictures it shows. */
interface PageToWrite {
  snapshot: Snapshot;
  xhtml: string;
  pictures: WebPagePicture[];
  leftOut: number;
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
  const pictureFolder = join(folder, "pictures");
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

  /** Writes a picture's file, unless one with its content is there already. */
  async function keepPicture(picture: FetchedPicture): Promise<WebPagePicture> {
    const kept = {
      pictureHash: hashBytes(picture.bytes),
      mediaType: picture.type.mediaType,
    };
    const path = join(pictureFolder, pictureName(kept));
    const there = await access(path).then(
      () => true,
      () => false,
    );
    if (!there) {
      await mkdir(pictureFolder, { recursive: true });
      const partial = `${path}.${randomUUID()}.part`;
      try {
        await writeFile(partial, picture.bytes);
        await rename(partial, path);
      } catch (error) {
        await rm(partial, { force: true });
        throw error;
      }
    }
    return {
      number: 0,
      ...kept,
      width: picture.width,
      height: picture.height,
      bytes: picture.bytes.byteLength,
      address: picture.address,
    };
  }

  async function pageFor(
    address: string,
    html: string | undefined,
    fetch = options.fetch,
  ): Promise<PageToWrite> {
    const page =
      html === undefined
        ? await fetchPage(address, {
            fetch,
            limits: options.limits,
            languages: options.languages(),
          })
        : { html, url: address };
    const snapshot = snapshotOf(page.html, page.url, address);
    // HTML given is saved as it is: nothing is fetched, its pictures neither.
    const fetched =
      html === undefined
        ? await fetchPictures(snapshot.pictures, options.pictureFetch, {
            ...(options.pictureLimits ? { limits: options.pictureLimits } : {}),
            ...(options.reportError ? { reportError: options.reportError } : {}),
          })
        : { kept: [], leftOut: 0 };
    const shown = new Map<number, ShownPicture>();
    const pictures: WebPagePicture[] = [];
    for (const picture of fetched.kept) {
      const kept = { ...(await keepPicture(picture)), number: pictures.length + 1 };
      pictures.push(kept);
      shown.set(picture.index, {
        src: `pictures/${pictureName(kept)}`,
        width: kept.width,
        height: kept.height,
      });
    }
    return {
      snapshot,
      xhtml: withPictures(snapshot.xhtml, shown),
      pictures,
      leftOut: fetched.leftOut,
    };
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

  /** Records a version written, and the pictures it shows. */
  const record = (documentId: string, file: SnapshotFile, address: string, page: PageToWrite) => {
    const at = now();
    const version: WebPageVersion = {
      documentId,
      contentHash: file.contentHash,
      address,
      siteName: page.snapshot.siteName,
      author: page.snapshot.author,
      published: page.snapshot.published,
      fetchedAt: at,
      picturesLeftOut: page.leftOut,
    };
    recordVersion(db, version, at);
    recordPictures(db, documentId, file.contentHash, page.pictures, at);
  };

  return {
    /**
     * Removes the snapshots of web pages deleted permanently, their text going
     * with them, and the pictures no other page shows.
     */
    async dropPurged(): Promise<void> {
      const rows = db.all<{ path: string }>(
        "SELECT path FROM documents WHERE kind = 'web' AND purged_at IS NOT NULL",
      );
      await Promise.all(rows.map((row) => rm(row.path, { force: true })));
      await Promise.all(
        picturesOnlyPurgedShow(db).map((picture) =>
          rm(join(pictureFolder, pictureName(picture)), { force: true }),
        ),
      );
    },

    /**
     * See `CoreApi.saveWebPage`; the caller files it in its Folder. `added`
     * hears a new Document's id as it is stored, before it is processed.
     * `fetch` fetches the page instead of the usual one: web search's (#128).
     */
    save(
      input: unknown,
      added?: (documentId: string) => void,
      fetch?: PageFetch,
    ): Promise<WebPageResult> {
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
          const page = await pageFor(address, html, fetch);
          const meanwhile = documentAtAddress(db, address);
          if (meanwhile) return saved(meanwhile);
          const id = randomUUID();
          const file = await write(join(folder, `${id}.html`), page.xhtml);
          const document = documents.addSnapshot(id, page.snapshot.title, file, () => {
            record(id, file, address, page);
            added?.(id);
          });
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
          const page = await pageFor(info.address, undefined);
          const current = documents.get(document.id);
          const file = await write(current.path, page.xhtml);
          record(current.id, file, info.address, page);
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
