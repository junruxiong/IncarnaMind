/**
 * A pack style's file, with the locales it is parsed with, for the engine: the style's own
 * locale and the request's, and the ones every request may need (English for the second render
 * of a Chinese style, `formatJob`). The pack's chunks are kept as `PackFiles` keeps them.
 */
import { join } from "node:path";
import { PackFiles, type PackIndex, type PackStyle, readPackIndex } from "./pack";
import type { EngineStyle } from "./protocol";
import { StyleError } from "./wasm";

const ALWAYS = ["en-US", "en-GB", "zh-CN"];

export function createPackStyles(folder: string) {
  const path = join(folder, "styles.pack");
  let read: {
    index: PackIndex & { dataOffset: number };
    byId: Map<string, PackStyle>;
    files: PackFiles;
  } | null = null;

  const pack = () => {
    if (!read) {
      const index = readPackIndex(path);
      read = {
        index,
        byId: new Map(index.styles.map((style) => [style.id, style])),
        files: new PackFiles(path, index),
      };
    }
    return read;
  };

  return {
    style(style: EngineStyle): { text: string; locales: string[] } {
      const { index, byId, files } = pack();
      const entry = byId.get(style.packId);
      if (!entry?.at) throw new StyleError(`The style pack has no style ${style.packId}.`, null);
      const codes = [...new Set([...ALWAYS, entry.locale, style.locale])].flatMap((code) => {
        const slot = code ? index.locales[code] : undefined;
        return slot ? [slot] : [];
      });
      return { text: files.text(entry.at), locales: codes.map((slot) => files.text(slot)) };
    },

    decompressions: () => pack().files.decompressions,
  };
}
