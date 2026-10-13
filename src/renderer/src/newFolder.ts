import type { Translate } from "./objects";

/**
 * "Untitled Folder", or the first of "Untitled Folder 2", "Untitled Folder 3"…
 * that no Folder has: Folder names are unique, whatever their case.
 */
export function untitledFolderName(t: Translate, taken: readonly string[]): string {
  const used = new Set(taken.map((name) => name.toLowerCase()));
  const plain = t("folder.untitled");
  if (!used.has(plain.toLowerCase())) return plain;
  for (let number = 2; ; number++) {
    const name = t("folder.untitled.numbered", { number });
    if (!used.has(name.toLowerCase())) return name;
  }
}
