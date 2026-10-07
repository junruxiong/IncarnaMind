/**
 * Writes a ZIP archive, the container of a .docx, with Node's own zlib: each
 * entry deflated, no ZIP64 (an export is far below 4 GB), and a fixed
 * timestamp, so the same content always gives the same bytes.
 */
import { crc32, deflateRawSync } from "node:zlib";

export interface ZipEntry {
  /** The path inside the archive, with "/" separators. */
  name: string;
  /** Text is stored as UTF-8. */
  data: string | Uint8Array;
}

/** 1980-01-01 00:00, the earliest MS-DOS date ZIP can hold. */
const DOS_TIME = 0;
const DOS_DATE = (1 << 5) | 1;
/** Version 2.0: deflate. */
const VERSION = 20;
const DEFLATE = 8;
/** File names are UTF-8. */
const UTF8_NAMES = 1 << 11;

export function zip(entries: readonly ZipEntry[]): Uint8Array {
  const locals: Buffer[] = [];
  const centrals: Buffer[] = [];
  let offset = 0;

  for (const entry of entries) {
    const name = Buffer.from(entry.name, "utf8");
    const data = typeof entry.data === "string" ? Buffer.from(entry.data, "utf8") : entry.data;
    const compressed = deflateRawSync(data);
    const checksum = crc32(data);

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(VERSION, 4);
    local.writeUInt16LE(UTF8_NAMES, 6);
    local.writeUInt16LE(DEFLATE, 8);
    local.writeUInt16LE(DOS_TIME, 10);
    local.writeUInt16LE(DOS_DATE, 12);
    local.writeUInt32LE(checksum, 14);
    local.writeUInt32LE(compressed.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);
    locals.push(local, name, compressed);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(VERSION, 4);
    central.writeUInt16LE(VERSION, 6);
    central.writeUInt16LE(UTF8_NAMES, 8);
    central.writeUInt16LE(DEFLATE, 10);
    central.writeUInt16LE(DOS_TIME, 12);
    central.writeUInt16LE(DOS_DATE, 14);
    central.writeUInt32LE(checksum, 16);
    central.writeUInt32LE(compressed.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(name.length, 28);
    // Extra field, comment, disk number, internal and external attributes: none.
    central.writeUInt32LE(offset, 42);
    centrals.push(central, name);

    offset += local.length + name.length + compressed.length;
  }

  const directorySize = centrals.reduce((size, part) => size + part.length, 0);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directorySize, 12);
  end.writeUInt32LE(offset, 16);

  return new Uint8Array(Buffer.concat([...locals, ...centrals, end]));
}
