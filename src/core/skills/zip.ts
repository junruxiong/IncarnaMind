/**
 * Reads a zip file held in memory, with `node:zlib` alone: stored and deflated
 * entries, which is what zip tools write by default. Encrypted entries, ZIP64
 * archives (only needed past 4 GB or 65,535 entries) and other compression
 * methods are refused. Sizes are checked against the limits before anything is
 * inflated, and inflating stops at each entry's declared size, so a small zip
 * can't expand into a huge one.
 */
import * as zlib from "node:zlib";

export interface ZipEntry {
  /** As stored in the zip: "/" between folders, not yet checked for safety. */
  name: string;
  kind: "file" | "directory" | "symlink";
  /** A file's bytes, or a symbolic link's target. Empty for a directory. */
  data: Buffer;
}

export type ZipErrorKind = "invalid" | "too-large" | "too-many-files";

export class ZipError extends Error {
  override name = "ZipError";
  constructor(
    readonly kind: ZipErrorKind,
    message: string,
  ) {
    super(message);
  }
}

export interface ZipLimits {
  /** Uncompressed bytes of all entries together. */
  maxBytes: number;
  /** Entries that are files or links. */
  maxFiles: number;
}

const END_OF_CENTRAL_DIRECTORY = 0x06054b50;
const CENTRAL_HEADER = 0x02014b50;
const LOCAL_HEADER = 0x04034b50;
const STORED = 0;
const DEFLATED = 8;
/** The zip's own comment can be this long, so the end record is at most this far from the end. */
const MAX_COMMENT = 0xffff;
const UNIX = 3;
const S_IFMT = 0o170000;
const S_IFLNK = 0o120000;
const S_IFDIR = 0o040000;
/** The MS-DOS directory attribute. */
const DOS_DIRECTORY = 0x10;

/** zlib.crc32 arrived in Node 22.2; without it the check is skipped. */
const crc32 = (zlib as { crc32?: (data: Uint8Array) => number }).crc32;

const invalid = (message: string) => new ZipError("invalid", message);

function findEndOfCentralDirectory(bytes: Buffer): number {
  const last = bytes.length - 22;
  const first = Math.max(0, last - MAX_COMMENT);
  for (let at = last; at >= first; at--) {
    if (bytes.readUInt32LE(at) === END_OF_CENTRAL_DIRECTORY) return at;
  }
  throw invalid("This isn't a zip file.");
}

/** Every entry of the zip, in the order of its central directory. */
export function readZip(bytes: Buffer, limits: ZipLimits): ZipEntry[] {
  if (bytes.length < 22) throw invalid("This isn't a zip file.");
  const end = findEndOfCentralDirectory(bytes);
  const count = bytes.readUInt16LE(end + 10);
  const directorySize = bytes.readUInt32LE(end + 12);
  const directoryOffset = bytes.readUInt32LE(end + 16);
  if (count === 0xffff || directoryOffset === 0xffffffff || directorySize === 0xffffffff) {
    throw invalid("ZIP64 archives aren't supported.");
  }
  if (directoryOffset + directorySize > end) throw invalid("The zip's directory is damaged.");

  const entries: ZipEntry[] = [];
  let totalBytes = 0;
  let files = 0;
  let at = directoryOffset;
  for (let index = 0; index < count; index++) {
    if (at + 46 > end || bytes.readUInt32LE(at) !== CENTRAL_HEADER) {
      throw invalid("The zip's directory is damaged.");
    }
    const madeBy = bytes.readUInt16LE(at + 4);
    const flags = bytes.readUInt16LE(at + 8);
    const method = bytes.readUInt16LE(at + 10);
    const crc = bytes.readUInt32LE(at + 16);
    const compressedSize = bytes.readUInt32LE(at + 20);
    const size = bytes.readUInt32LE(at + 24);
    const nameLength = bytes.readUInt16LE(at + 28);
    const extraLength = bytes.readUInt16LE(at + 30);
    const commentLength = bytes.readUInt16LE(at + 32);
    const external = bytes.readUInt32LE(at + 38);
    const localOffset = bytes.readUInt32LE(at + 42);
    const name = bytes.toString("utf8", at + 46, at + 46 + nameLength);
    at += 46 + nameLength + extraLength + commentLength;

    if (flags & 1) throw invalid(`${name} is encrypted.`);
    const mode = madeBy >> 8 === UNIX ? external >>> 16 : 0;
    const kind: ZipEntry["kind"] =
      (mode & S_IFMT) === S_IFLNK
        ? "symlink"
        : name.endsWith("/") || (mode & S_IFMT) === S_IFDIR || (!mode && external & DOS_DIRECTORY)
          ? "directory"
          : "file";
    if (kind === "directory") {
      entries.push({ name, kind, data: Buffer.alloc(0) });
      continue;
    }

    files++;
    if (files > limits.maxFiles) {
      throw new ZipError("too-many-files", `The zip has more than ${limits.maxFiles} files.`);
    }
    // Declared sizes first: nothing is inflated past the limit.
    totalBytes += size;
    if (totalBytes > limits.maxBytes) {
      throw new ZipError(
        "too-large",
        `The zip's files add up to more than ${limits.maxBytes} bytes.`,
      );
    }

    if (localOffset + 30 > bytes.length || bytes.readUInt32LE(localOffset) !== LOCAL_HEADER) {
      throw invalid(`${name}'s entry is damaged.`);
    }
    const dataStart =
      localOffset +
      30 +
      bytes.readUInt16LE(localOffset + 26) +
      bytes.readUInt16LE(localOffset + 28);
    if (dataStart + compressedSize > bytes.length) throw invalid(`${name} is cut short.`);
    const compressed = bytes.subarray(dataStart, dataStart + compressedSize);

    let data: Buffer;
    if (method === STORED) {
      data = Buffer.from(compressed);
    } else if (method === DEFLATED) {
      try {
        // One byte more than declared: an entry that inflates further lied about its size.
        data = zlib.inflateRawSync(compressed, { maxOutputLength: size + 1 });
      } catch {
        throw invalid(`${name} can't be decompressed.`);
      }
    } else {
      throw invalid(`${name} uses a compression method that isn't supported (${method}).`);
    }
    if (data.length !== size) throw invalid(`${name} isn't the size the zip says.`);
    if (crc32 && crc32(data) >>> 0 !== crc)
      throw invalid(`${name} is damaged (its checksum is wrong).`);
    entries.push({ name, kind, data });
  }
  return entries;
}
