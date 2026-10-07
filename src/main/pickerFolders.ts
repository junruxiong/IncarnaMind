/**
 * Where the system's open dialogs start: the folder the User last picked
 * from, kept per device in the data folder (pickers.json), else a folder on
 * this computer. Never a cloud folder by default: with "Desktop & Documents
 * in iCloud", or Google Drive and OneDrive syncing hard, the macOS open panel
 * waits on their File Provider before it shows, and the app looks frozen.
 */
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const FILE = "pickers.json";

export type PickerKind = "documents" | "folders";

function read(dataDir: string): Partial<Record<PickerKind, string>> {
  try {
    const saved = JSON.parse(readFileSync(join(dataDir, FILE), "utf8")) as Record<string, unknown>;
    const out: Partial<Record<PickerKind, string>> = {};
    for (const kind of ["documents", "folders"] as const) {
      const value = saved[kind];
      if (typeof value === "string") out[kind] = value;
    }
    return out;
  } catch {
    return {};
  }
}

const isFolder = (path: string) => {
  try {
    return existsSync(path) && statSync(path).isDirectory();
  } catch {
    return false;
  }
};

/** Where a dialog of this kind starts: the last folder picked from, if it's still there, else `fallback`. */
export function startFolder(dataDir: string, kind: PickerKind, fallback: string): string {
  const last = read(dataDir)[kind];
  return last && isFolder(last) ? last : fallback;
}

/** Remembers where the User picked from: the folder holding the picked file, or the picked folder's parent. */
export function rememberPick(dataDir: string, kind: PickerKind, picked: string): void {
  const saved = read(dataDir);
  saved[kind] = dirname(picked);
  try {
    writeFileSync(join(dataDir, FILE), JSON.stringify(saved));
  } catch {
    // Not worth failing over: the dialog starts at the default next time.
  }
}
