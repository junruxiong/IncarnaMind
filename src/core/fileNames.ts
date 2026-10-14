import { randomUUID } from "node:crypto";
import { open, rename, rm, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";

/** Longest a file name made from a title gets, before its extension. */
const MAX_NAME_LENGTH = 120;

/** Names Windows keeps for devices, with or without an extension. */
const WINDOWS_DEVICE = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i;

/** How many names `writeNewFile` tries ("Tides 2.docx", "Tides 3.docx"…) before giving up. */
const MAX_TRIES = 1000;

/**
 * A file name (without its extension) from a title, e.g. a Mind's or a
 * Document's: without characters file systems refuse, and not too long.
 * `fallback` if nothing is left.
 */
export function safeFileName(title: string, fallback: string): string {
  const name = title
    // biome-ignore lint/suspicious/noControlCharactersInRegex: control characters can't be in file names.
    .replace(/[\\/:*?"<>|\u0000-\u001f\u007f]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\.+/, "")
    .slice(0, MAX_NAME_LENGTH)
    .replace(/[\s.]+$/, "");
  if (!name) return fallback;
  return WINDOWS_DEVICE.test(name) ? `${name}_` : name;
}

/**
 * Writes a new file in `folder` under `name` ("Tides.docx"), or under the
 * next free name, as Finder does ("Tides 2.docx", "Tides 3.docx"), and
 * returns the name used. Each name is created exclusively, so nothing already
 * there, a file or a link, is ever replaced or followed. The bytes go to a
 * temporary file beside it first, which then takes the name at once, so the
 * file is never seen half written; if writing fails, neither is left.
 */
export async function writeNewFile(
  folder: string,
  name: string,
  data: Uint8Array,
): Promise<string> {
  const extension = extname(name);
  const stem = name.slice(0, name.length - extension.length);
  for (let n = 1; n <= MAX_TRIES; n++) {
    const candidate = n === 1 ? name : `${stem} ${n}${extension}`;
    const path = join(folder, candidate);
    try {
      await (await open(path, "wx")).close();
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") continue;
      throw error;
    }
    const temporary = join(folder, `.${candidate}.${randomUUID()}.partial`);
    try {
      await writeFile(temporary, data, { flag: "wx" });
      await rename(temporary, path);
    } catch (error) {
      await rm(temporary, { force: true });
      await rm(path, { force: true });
      throw error;
    }
    return candidate;
  }
  throw new Error(`Every name like "${name}" is taken in that folder.`);
}
