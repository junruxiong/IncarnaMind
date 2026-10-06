import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { Core, Document, DocumentStatus } from "../../src/core";

/** Writes a file the User might add, in a folder outside the data folder, and returns its path. */
export async function writeSourceFile(
  folder: string,
  name: string,
  contents: string | Uint8Array,
): Promise<string> {
  const path = join(folder, name);
  await writeFile(path, contents);
  return path;
}

export const sha256 = (contents: string | Uint8Array) =>
  createHash("sha256").update(contents).digest("hex");

/** Where the data folder keeps a Document's file. */
export const storedFile = (dataDir: string, contentHash: string) =>
  join(dataDir, "documents", contentHash);

const FINISHED: ReadonlySet<DocumentStatus> = new Set(["ready", "failed", "no-text"]);

/** Resolves with the Documents, in the order given, once each has finished processing. */
export function waitForProcessing(
  core: Core,
  ids: readonly string[],
  timeout = 15_000,
): Promise<Document[]> {
  return new Promise((resolve, reject) => {
    const wanted = new Set(ids);
    const finished = new Map<string, Document>();
    const settle = (document: Document) => {
      if (!wanted.has(document.id) || !FINISHED.has(document.status)) return;
      finished.set(document.id, document);
      if (finished.size < wanted.size) return;
      clearTimeout(timer);
      unsubscribe();
      resolve(ids.map((id) => finished.get(id) as Document));
    };
    const timer = setTimeout(() => {
      unsubscribe();
      reject(new Error(`Processing didn't finish within ${timeout} ms.`));
    }, timeout);
    const unsubscribe = core.on("document.status", settle);
    void core.listDocuments().then((documents) => {
      for (const document of documents) settle(document);
    }, reject);
  });
}

/** Adds files and waits until they have all finished processing. */
export async function addAndProcess(core: Core, paths: string[]): Promise<Document[]> {
  const { documents } = await core.addDocuments(paths);
  return waitForProcessing(
    core,
    documents.map((document) => document.id),
  );
}
