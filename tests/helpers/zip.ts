import { crc32, deflateRawSync } from "node:zlib";

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
