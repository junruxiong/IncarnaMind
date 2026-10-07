/**
 * Copies of Documents' files outside the data folder, named after the
 * Document rather than by its content hash: one to open in another app, or
 * one saved where the User chooses.
 */
import { mkdir, mkdtemp, readdir, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import type { Document, DocumentKind } from "../api";
import { safeFileName } from "../fileNames";

/** The folder in the temporary folder where copies to open in another app go, one folder each. */
export const OPEN_COPIES_FOLDER = "incarnamind-documents";

/** At startup, copies to open older than this are removed; newer ones may still be open. */
const OPEN_COPY_LIFETIME_MS = 24 * 60 * 60 * 1000;

/** Each kind's extensions, the one a copy gets first. */
const EXTENSIONS: Readonly<Record<DocumentKind, readonly string[]>> = {
  pdf: ["pdf"],
  text: ["txt"],
  markdown: ["md", "markdown"],
};

/** The extension a copy of a Document of this kind gets, without the dot. */
const copyExtension = (kind: DocumentKind): string => EXTENSIONS[kind][0] ?? kind;

/**
 * The file name a copy of a Document gets: its name, without characters file
 * systems refuse, and its kind's extension, unless the name ends with it already.
 */
export function copyFileName({ name, kind }: Pick<Document, "name" | "kind">): string {
  const base = safeFileName(name, "Document");
  const lower = base.toLowerCase();
  if (EXTENSIONS[kind].some((extension) => lower.endsWith(`.${extension}`))) return base;
  return `${base}.${copyExtension(kind)}`;
}

export function createOpenCopies(tempDir: string) {
  const root = join(tempDir, OPEN_COPIES_FOLDER);
  return {
    /**
     * A new, empty folder for one copy, so copies of Documents with the same
     * name, or of one Document opened twice, never replace one another (a copy
     * may be open, and on Windows locked).
     */
    async folder(): Promise<string> {
      await mkdir(root, { recursive: true });
      return mkdtemp(join(root, "copy-"));
    },

    /** Removes copies made more than a day before `now`. Never throws: a copy still open stays. */
    async tidy(now = Date.now()): Promise<void> {
      let names: string[];
      try {
        names = await readdir(root);
      } catch {
        return; // nothing copied yet
      }
      await Promise.all(
        names.map(async (name) => {
          const folder = join(root, name);
          try {
            if (now - (await stat(folder)).mtimeMs < OPEN_COPY_LIFETIME_MS) return;
            await rm(folder, { recursive: true, force: true });
          } catch {
            // In use, or gone already: tried again next time.
          }
        }),
      );
    },
  };
}
