/**
 * Document files in the data folder. Each is stored once, named by the
 * SHA-256 of its content: `documents/<hash>`.
 */
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream, mkdirSync, readdirSync, rmSync } from "node:fs";
import { access, rename, rm } from "node:fs/promises";
import { extname, join } from "node:path";
import { pipeline } from "node:stream/promises";
import type { DocumentKind } from "../api";

const KINDS: Readonly<Record<string, DocumentKind>> = {
  ".pdf": "pdf",
  ".txt": "text",
  ".md": "markdown",
  ".markdown": "markdown",
};

/** The kind of Document a file would be, from its extension; undefined if it isn't supported. */
export const kindOf = (path: string): DocumentKind | undefined =>
  KINDS[extname(path).toLowerCase()];

const HASH = /^[0-9a-f]{64}$/;

const exists = (path: string) =>
  access(path).then(
    () => true,
    () => false,
  );

export interface ImportedFile {
  contentHash: string;
  size: number;
}

export function createDocumentFiles(dataDir: string) {
  const directory = join(dataDir, "documents");
  // Copies in progress, so a half-written file never sits under a hash.
  const incoming = join(directory, ".incoming");
  const pathFor = (contentHash: string) => join(directory, contentHash);

  return {
    pathFor,

    /**
     * Prepares the folder at startup: removes copies a quit interrupted, and
     * files no live Document uses (e.g. when removing one failed earlier).
     */
    prepare(isInUse: (contentHash: string) => boolean): void {
      rmSync(incoming, { recursive: true, force: true });
      mkdirSync(incoming, { recursive: true });
      for (const name of readdirSync(directory)) {
        if (HASH.test(name) && !isInUse(name)) rmSync(pathFor(name), { force: true });
      }
    },

    /**
     * Copies a file into the folder, hashing it in the same pass, so the name
     * always matches what was copied even if the original changes meanwhile.
     */
    async import(source: string): Promise<ImportedFile> {
      const temporary = join(incoming, randomUUID());
      const hash = createHash("sha256");
      let size = 0;
      try {
        await pipeline(
          createReadStream(source),
          async function* (chunks: AsyncIterable<Buffer>) {
            for await (const chunk of chunks) {
              hash.update(chunk);
              size += chunk.length;
              yield chunk;
            }
          },
          createWriteStream(temporary, { flags: "wx" }),
        );
        const contentHash = hash.digest("hex");
        const target = pathFor(contentHash);
        // Already stored: same hash, same bytes. Keep the existing copy, which may be in use.
        if (await exists(target)) await rm(temporary, { force: true });
        else await rename(temporary, target);
        return { contentHash, size };
      } catch (error) {
        await rm(temporary, { force: true });
        throw error;
      }
    },

    /** Removes a stored file. A failure (e.g. Windows locks it) leaves it for `prepare` next time. */
    async remove(contentHash: string): Promise<void> {
      try {
        await rm(pathFor(contentHash), { force: true });
      } catch (error) {
        console.error(error);
      }
    },
  };
}

export type DocumentFiles = ReturnType<typeof createDocumentFiles>;
