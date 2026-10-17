/**
 * The citation-style pack (ADR-0016): every style and locale of the CSL repositories at a pinned
 * commit, unmodified, in one file of brotli-compressed chunks behind an index, so opening a style
 * decompresses its chunk alone. scripts/citationStylePack.ts writes it; the core reads its index
 * when the picker or a Mind's style needs it, and the style thread reads chunks. Pure Node.
 *
 * The file: "IMCSLPK1", the index's length (4 bytes, little-endian), the index as brotli JSON,
 * then the chunks, each a brotli JSON list of files' texts, at offsets from the end of the index.
 */
import { closeSync, openSync, readSync } from "node:fs";
import { brotliDecompressSync } from "node:zlib";
import type { CitationFormat } from "../../shared/citationStyles";

export const PACK_MAGIC = "IMCSLPK1";
export const PACK_HEADER_BYTES = PACK_MAGIC.length + 4;

/** Where a file is in the pack: its chunk, and its place in that chunk. */
export type PackSlot = readonly [chunk: number, slot: number];

/** A style as the pack lists it, from its `<info>` and `<style>` element. */
export interface PackStyle {
  /** Its file's name without ".csl": "apa", "nature", "the-lancet". */
  id: string;
  title: string;
  /** Its `<title-short>`, if it has one. */
  short?: string;
  format: CitationFormat;
  /** Its citations are notes (`class="note"`). */
  note?: true;
  fields?: string[];
  /** Its `default-locale`, if it has one: "zh-CN", "de-DE". */
  locale?: string;
  /** An independent style: where its file is. */
  at?: PackSlot;
  /** A journal (dependent) style: the style it follows, which it formats with. */
  parent?: string;
}

export interface PackIndex {
  version: 1;
  /** The commits of the CSL styles and locales repositories it was built from. */
  source: { styles: string; locales: string };
  chunks: { offset: number; length: number }[];
  styles: PackStyle[];
  /** Each locale's file, by its code: "en-US". */
  locales: Record<string, PackSlot>;
  /** The licence notice made from every style's and locale's `<info>`. */
  notice: PackSlot;
}

function readAt(path: string, offset: number, length: number): Buffer {
  const file = openSync(path, "r");
  try {
    const bytes = Buffer.alloc(length);
    let read = 0;
    while (read < length) {
      const got = readSync(file, bytes, read, length - read, offset + read);
      if (got === 0) throw new Error(`The style pack ${path} ends early.`);
      read += got;
    }
    return bytes;
  } finally {
    closeSync(file);
  }
}

/** The pack's index, read and decompressed. Throws for a file that isn't a pack. */
export function readPackIndex(path: string): PackIndex & { dataOffset: number } {
  const header = readAt(path, 0, PACK_HEADER_BYTES);
  if (header.subarray(0, PACK_MAGIC.length).toString("latin1") !== PACK_MAGIC) {
    throw new Error(`${path} isn't a citation-style pack.`);
  }
  const length = header.readUInt32LE(PACK_MAGIC.length);
  const index = JSON.parse(
    brotliDecompressSync(readAt(path, PACK_HEADER_BYTES, length)).toString("utf8"),
  ) as PackIndex;
  return { ...index, dataOffset: PACK_HEADER_BYTES + length };
}

/**
 * The pack's files, each read by decompressing its chunk; the last few chunks are kept, so
 * switching between two styles doesn't decompress either again.
 */
export class PackFiles {
  /** How many chunks have been decompressed: what the cache saves shows here. */
  decompressions = 0;
  private readonly kept = new Map<number, string[]>();
  private readonly path: string;
  private readonly index: PackIndex & { dataOffset: number };
  private readonly cap: number;

  constructor(path: string, index: PackIndex & { dataOffset: number }, cap = 4) {
    this.path = path;
    this.index = index;
    this.cap = cap;
  }

  text([chunk, slot]: PackSlot): string {
    let files = this.kept.get(chunk);
    if (files) {
      this.kept.delete(chunk);
    } else {
      const where = this.index.chunks[chunk];
      if (!where) throw new Error(`The style pack has no chunk ${chunk}.`);
      const bytes = readAt(this.path, this.index.dataOffset + where.offset, where.length);
      files = JSON.parse(brotliDecompressSync(bytes).toString("utf8")) as string[];
      this.decompressions++;
    }
    this.kept.set(chunk, files);
    for (const oldest of this.kept.keys()) {
      if (this.kept.size <= this.cap) break;
      this.kept.delete(oldest);
    }
    const text = files[slot];
    if (text === undefined) throw new Error(`The style pack's chunk ${chunk} has no file ${slot}.`);
    return text;
  }
}
