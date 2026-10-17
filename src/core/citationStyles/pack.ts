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
import { CITATION_FORMATS, type CitationFormat } from "../../shared/citationStyles";

export const PACK_MAGIC = "IMCSLPK1";
export const PACK_HEADER_BYTES = PACK_MAGIC.length + 4;

/** Where a file is in the pack: its chunk, and its place in that chunk. */
export type PackSlot = readonly [chunk: number, slot: number];

/** A style as the pack lists it, from its `<info>` and `<style>` element. */
export interface PackStyle {
  /** Its file's name without ".csl": "apa", "nature", "the-lancet". */
  id: string;
  title: string;
  short?: string;
  format: CitationFormat;
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
  locales: Record<string, PackSlot>;
  /** The licence notice made from every style's and locale's `<info>`. */
  notice: PackSlot;
}

// Not errors.ts's isRecord: the build script imports this file, and Node can't run errors.ts.
const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isText = (value: unknown) => typeof value === "string";
const isCount = (value: unknown) => Number.isInteger(value) && Number(value) >= 0;

const isSlot = (value: unknown): value is PackSlot =>
  Array.isArray(value) && value.length === 2 && value.every(isCount);

function isPackStyle(value: unknown): value is PackStyle {
  if (!isObject(value) || !isText(value.id) || !isText(value.title)) return false;
  return (
    CITATION_FORMATS.some((format) => format === value.format) &&
    (value.short === undefined || isText(value.short)) &&
    (value.note === undefined || value.note === true) &&
    (value.fields === undefined || (Array.isArray(value.fields) && value.fields.every(isText))) &&
    (value.locale === undefined || isText(value.locale)) &&
    (value.at === undefined || isSlot(value.at)) &&
    (value.parent === undefined || isText(value.parent))
  );
}

function isPackIndex(value: unknown): value is PackIndex {
  if (!isObject(value) || value.version !== 1) return false;
  const { source, chunks, styles, locales } = value;
  return (
    isObject(source) &&
    isText(source.styles) &&
    isText(source.locales) &&
    Array.isArray(chunks) &&
    chunks.every((chunk) => isObject(chunk) && isCount(chunk.offset) && isCount(chunk.length)) &&
    Array.isArray(styles) &&
    styles.every(isPackStyle) &&
    isObject(locales) &&
    Object.values(locales).every(isSlot) &&
    isSlot(value.notice)
  );
}

const isTexts = (value: unknown): value is string[] => Array.isArray(value) && value.every(isText);

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
  const index: unknown = JSON.parse(
    brotliDecompressSync(readAt(path, PACK_HEADER_BYTES, length)).toString("utf8"),
  );
  if (!isPackIndex(index)) throw new Error(`${path} has an index this version can't read.`);
  return { ...index, dataOffset: PACK_HEADER_BYTES + length };
}

/**
 * The pack's files, each read by decompressing its chunk; the last few chunks are kept, so
 * switching between two styles doesn't decompress either again.
 */
export class PackFiles {
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
      const read: unknown = JSON.parse(brotliDecompressSync(bytes).toString("utf8"));
      if (!isTexts(read)) throw new Error(`The style pack's chunk ${chunk} isn't a list of files.`);
      files = read;
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
