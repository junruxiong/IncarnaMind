/**
 * A small ZIP reader for Office packages (.docx, .pptx, .xlsx): the central
 * directory, and stored or deflated entries. No encryption, no multi-disk, no
 * ZIP64. It inflates with `DecompressionStream("deflate-raw")`, so it runs in
 * the processing worker and in the renderer alike, with no dependency.
 * Ported from the office-formats spike (ADR-0011).
 *
 * Untrusted input: an entry is inflated against a size limit, so a ZIP bomb
 * fails instead of filling memory, and large parts (a sheet) can be read as
 * a stream of text and dropped part-way.
 *
 * Password-protected Office files and the legacy binary formats (.doc, .xls,
 * .ppt) aren't ZIP files but OLE compound files: `openPackage` tells which,
 * so processing can say "needs a password" rather than "not a ZIP file".
 */
import { ExtractionError } from "./errors";

interface ZipEntry {
  name: string;
  method: number;
  compressedSize: number;
  size: number;
  localHeaderOffset: number;
}

/** The most entries a package may have. */
const MAX_ENTRIES = 20_000;
/** The largest a part read whole may inflate to: an XML part parsed into a tree. */
export const MAX_PART_BYTES = 64 * 1024 * 1024;

const EOCD = 0x06054b50;
const ZIP64_EOCD_LOCATOR = 0x07064b50;
const CENTRAL = 0x02014b50;
const LOCAL = 0x04034b50;

/** The signature of an OLE compound file. */
const CFB_SIGNATURE = [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1];

/** "EncryptedPackage" in UTF-16LE: the stream an encrypted Office file keeps its package in. */
const ENCRYPTED_PACKAGE = (() => {
  const name = "EncryptedPackage";
  const bytes = new Uint8Array(name.length * 2);
  for (let index = 0; index < name.length; index++) bytes[index * 2] = name.charCodeAt(index);
  return bytes;
})();

function startsWith(bytes: Uint8Array, prefix: readonly number[]): boolean {
  return prefix.every((byte, index) => bytes[index] === byte);
}

function contains(bytes: Uint8Array, needle: Uint8Array): boolean {
  const first = needle[0] as number;
  for (let at = bytes.indexOf(first); at >= 0 && at <= bytes.length - needle.length; ) {
    let index = 1;
    while (index < needle.length && bytes[at + index] === needle[index]) index++;
    if (index === needle.length) return true;
    at = bytes.indexOf(first, at + 1);
  }
  return false;
}

export class ZipArchive {
  private readonly entries = new Map<string, ZipEntry>();
  private readonly view: DataView;

  constructor(private readonly bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.readCentralDirectory();
  }

  private readCentralDirectory(): void {
    const { view, bytes } = this;
    // The end record is in the last 22 + 65,535 (its comment) bytes.
    let eocd = -1;
    for (let at = bytes.length - 22; at >= Math.max(0, bytes.length - 22 - 65_535); at--) {
      if (view.getUint32(at, true) === EOCD) {
        eocd = at;
        break;
      }
    }
    if (eocd < 0) throw new ExtractionError("unreadable", "Not a ZIP package: no end record.");
    if (eocd >= 20 && view.getUint32(eocd - 20, true) === ZIP64_EOCD_LOCATOR) {
      throw new ExtractionError("unreadable", "ZIP64 packages aren't supported.");
    }
    const count = view.getUint16(eocd + 10, true);
    let offset = view.getUint32(eocd + 16, true);
    if (count > MAX_ENTRIES) {
      throw new ExtractionError("unreadable", `The package has too many parts (${count}).`);
    }
    const decoder = new TextDecoder();
    for (let index = 0; index < count; index++) {
      if (offset + 46 > bytes.length || view.getUint32(offset, true) !== CENTRAL) {
        throw new ExtractionError("unreadable", "The package's directory is corrupt.");
      }
      const nameLength = view.getUint16(offset + 28, true);
      const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength));
      this.entries.set(name, {
        name,
        method: view.getUint16(offset + 10, true),
        compressedSize: view.getUint32(offset + 20, true),
        size: view.getUint32(offset + 24, true),
        localHeaderOffset: view.getUint32(offset + 42, true),
      });
      offset +=
        46 + nameLength + view.getUint16(offset + 30, true) + view.getUint16(offset + 32, true);
    }
  }

  has(name: string): boolean {
    return this.entries.has(name);
  }

  /** The names of the parts, in the directory's order. */
  names(): string[] {
    return [...this.entries.keys()];
  }

  /** An entry's compressed bytes and method; undefined if there's no such entry. */
  private raw(name: string): { data: Uint8Array; method: number } | undefined {
    const entry = this.entries.get(name);
    if (!entry) return undefined;
    const { view, bytes } = this;
    const at = entry.localHeaderOffset;
    if (at + 30 > bytes.length || view.getUint32(at, true) !== LOCAL) {
      throw new ExtractionError("unreadable", `A part of the package is corrupt: ${name}`);
    }
    const start = at + 30 + view.getUint16(at + 26, true) + view.getUint16(at + 28, true);
    if (start + entry.compressedSize > bytes.length) {
      throw new ExtractionError("unreadable", `A part of the package is cut short: ${name}`);
    }
    if (entry.method !== 0 && entry.method !== 8) {
      throw new ExtractionError("unreadable", `Unsupported compression in ${name}.`);
    }
    return { data: bytes.subarray(start, start + entry.compressedSize), method: entry.method };
  }

  /** An entry's bytes, inflated, up to `limit` bytes; undefined if there's no such entry. */
  async read(name: string, limit = MAX_PART_BYTES): Promise<Uint8Array | undefined> {
    const raw = this.raw(name);
    if (!raw) return undefined;
    if (raw.method === 0) {
      if (raw.data.length > limit) throw tooLarge(name);
      return raw.data;
    }
    const chunks: Uint8Array[] = [];
    let total = 0;
    for await (const chunk of inflate(raw.data)) {
      total += chunk.length;
      if (total > limit) throw tooLarge(name);
      chunks.push(chunk);
    }
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }

  async readText(name: string, limit = MAX_PART_BYTES): Promise<string | undefined> {
    const data = await this.read(name, limit);
    return data && new TextDecoder().decode(data);
  }

  /**
   * An entry's text, a chunk at a time, as it inflates, up to `limit` bytes:
   * for parts too large to hold whole. Stopping early (a `break`) stops inflating.
   */
  async *textChunks(name: string, limit: number): AsyncGenerator<string> {
    const raw = this.raw(name);
    if (!raw) return;
    const decoder = new TextDecoder();
    if (raw.method === 0) {
      if (raw.data.length > limit) throw tooLarge(name);
      yield decoder.decode(raw.data);
      return;
    }
    let total = 0;
    for await (const chunk of inflate(raw.data)) {
      total += chunk.length;
      if (total > limit) throw tooLarge(name);
      yield decoder.decode(chunk, { stream: true });
    }
    const rest = decoder.decode();
    if (rest) yield rest;
  }
}

const tooLarge = (name: string) =>
  new ExtractionError("unreadable", `A part of the package is too large to read: ${name}`);

async function* inflate(data: Uint8Array): AsyncGenerator<Uint8Array> {
  const stream = new Blob([data as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream("deflate-raw"));
  const reader = stream.getReader();
  try {
    for (;;) {
      let result: ReadableStreamReadResult<Uint8Array>;
      try {
        result = await reader.read();
      } catch (error) {
        throw new ExtractionError(
          "unreadable",
          `A part of the package couldn't be inflated: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
      if (result.done) return;
      yield result.value;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}

/**
 * Opens an Office package. An OLE compound file instead of a ZIP is either
 * encrypted (it holds an "EncryptedPackage" stream: the file needs a
 * password) or a legacy binary file renamed; both are refused with a reason.
 */
export function openPackage(bytes: Uint8Array): ZipArchive {
  if (startsWith(bytes, CFB_SIGNATURE)) {
    if (contains(bytes, ENCRYPTED_PACKAGE)) {
      throw new ExtractionError("password-protected", "The file is encrypted with a password.");
    }
    throw new ExtractionError(
      "unreadable",
      "The file is an older binary Office file (97–2003), not an Office Open XML package.",
    );
  }
  return new ZipArchive(bytes);
}

/** Resolves a relationship target against the part that owns it: ("ppt/slides/slide1.xml", "../media/a.png"). */
export function resolvePart(owner: string, target: string): string {
  if (target.startsWith("/")) return target.slice(1);
  const parts = owner.split("/").slice(0, -1);
  for (const segment of target.split("/")) {
    if (segment === "..") parts.pop();
    else if (segment !== "." && segment !== "") parts.push(segment);
  }
  return parts.join("/");
}
