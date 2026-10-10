import { createHash } from "node:crypto";
import { crc32, deflateRawSync, inflateRawSync } from "node:zlib";

/** One entry of a zip to build. */
export interface ZipInput {
  /** As stored, unchecked: tests use it for "../" and absolute paths too. */
  name: string;
  /** A file's contents. */
  data?: string | Buffer;
  /** Makes the entry a symbolic link to this target (Unix permissions, as zip on macOS and Linux stores links). */
  link?: string;
  /** Makes the entry a folder; its name should end with "/". */
  folder?: boolean;
  /** Stored as is rather than deflated. */
  store?: boolean;
  /** The uncompressed size to declare, if not the real one (a lie, for zip bombs). */
  declaredSize?: number;
}

/** A zip file as zip tools write one: local headers, the central directory, and its end record. */
export function buildZip(entries: readonly ZipInput[]): Buffer {
  const locals: Buffer[] = [];
  const centrals: Buffer[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name, "utf8");
    const raw = Buffer.from(entry.link ?? entry.data ?? "");
    const compress = !entry.store && !entry.folder && !entry.link;
    const body = compress ? deflateRawSync(raw) : raw;
    const crc = crc32(raw) >>> 0;
    const size = entry.declaredSize ?? raw.length;
    const mode = entry.link ? 0o120777 : entry.folder ? 0o040755 : 0o100644;

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x0800, 6); // UTF-8 names
    local.writeUInt16LE(compress ? 8 : 0, 8);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(size, 22);
    local.writeUInt16LE(name.length, 26);
    locals.push(local, name, body);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE((3 << 8) | 30, 4); // made by Unix
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0x0800, 8);
    central.writeUInt16LE(compress ? 8 : 0, 10);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(body.length, 20);
    central.writeUInt32LE(size, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(((mode << 16) | (entry.folder ? 0x10 : 0)) >>> 0, 38);
    central.writeUInt32LE(offset, 42);
    centrals.push(central, name);
    offset += local.length + name.length + body.length;
  }
  const directory = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, directory, end]);
}

const LOCAL_HEADER = 0x04034b50;
const CENTRAL_HEADER = 0x02014b50;
const END_OF_DIRECTORY = 0x06054b50;

/**
 * A fingerprint of a ZIP archive as its writer made it: the SHA-256 of all its
 * bytes but the compressed ones, which depend on the build of zlib. Node's own
 * zlib (in CI, and in the app's Electron) and the system's (which some Node
 * installs use instead) deflate the same text to different bytes. Each
 * deflated entry must be what this Node's `deflateRawSync` makes of its
 * content, so with the fingerprint every byte is pinned all the same.
 *
 * It throws unless the archive is consistent: each central record matches its
 * local header and points at it, the entries follow one another, and the end
 * record points at the directory that follows them. What is hashed is the
 * archive with each entry's content in place of its compressed bytes, and the
 * sizes and offsets that follow from that.
 */
export function zipFingerprint(data: Uint8Array): string {
  const bytes = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  const end = bytes.length - 22;
  if (end < 0 || bytes.readUInt32LE(end) !== END_OF_DIRECTORY) {
    throw new Error("No end record where an archive without a comment has it.");
  }
  const count = bytes.readUInt16LE(end + 10);
  const directoryStart = bytes.readUInt32LE(end + 16);
  if (bytes.readUInt16LE(end + 8) !== count) throw new Error("The entry counts differ.");

  const hash = createHash("sha256");
  const centrals: Buffer[] = [];
  let local = 0;
  let central = directoryStart;
  let canonical = 0;
  for (let index = 0; index < count; index++) {
    if (bytes.readUInt32LE(central) !== CENTRAL_HEADER) throw new Error("Bad central record.");
    if (bytes.readUInt32LE(local) !== LOCAL_HEADER) throw new Error("Bad local header.");
    if (bytes.readUInt32LE(central + 42) !== local) throw new Error("A record points elsewhere.");
    const nameLength = bytes.readUInt16LE(central + 28);
    const centralLength =
      46 + nameLength + bytes.readUInt16LE(central + 30) + bytes.readUInt16LE(central + 32);
    const localLength = 30 + nameLength + bytes.readUInt16LE(local + 28);
    // Version needed, flags, method, time, date, CRC-32, sizes and name length agree.
    if (!bytes.subarray(local + 4, local + 28).equals(bytes.subarray(central + 6, central + 30))) {
      throw new Error("A local header and its central record differ.");
    }
    const name = bytes.subarray(central + 46, central + 46 + nameLength);
    if (!name.equals(bytes.subarray(local + 30, local + 30 + nameLength))) {
      throw new Error("A local header and its central record name different entries.");
    }
    const method = bytes.readUInt16LE(local + 8);
    const compressed = bytes.subarray(
      local + localLength,
      local + localLength + bytes.readUInt32LE(local + 18),
    );
    const content = method === 8 ? inflateRawSync(compressed) : compressed;
    if (content.length !== bytes.readUInt32LE(local + 22)) throw new Error("A size is wrong.");
    if (crc32(content) >>> 0 !== bytes.readUInt32LE(local + 14)) throw new Error("A CRC is wrong.");
    if (method === 8 && !compressed.equals(deflateRawSync(content))) {
      throw new Error(`${name.toString("utf8")} isn't deflated as this Node deflates it.`);
    }

    const header = Buffer.from(bytes.subarray(local, local + localLength));
    header.writeUInt32LE(content.length, 18);
    hash.update(header).update(content);
    const record = Buffer.from(bytes.subarray(central, central + centralLength));
    record.writeUInt32LE(content.length, 20);
    record.writeUInt32LE(canonical, 42);
    centrals.push(record);
    canonical += localLength + content.length;
    local += localLength + compressed.length;
    central += centralLength;
  }
  if (local !== directoryStart)
    throw new Error("Something lies between the entries and the directory.");
  if (central !== end || bytes.readUInt32LE(end + 12) !== end - directoryStart) {
    throw new Error("The end record doesn't match the directory.");
  }
  const record = Buffer.from(bytes.subarray(end));
  record.writeUInt32LE(canonical, 16);
  for (const each of centrals) hash.update(each);
  return hash.update(record).digest("hex");
}
