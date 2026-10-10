/**
 * The one ZIP module (src/core/documents/formats/zip.ts, #77): the archives
 * IncarnaMind writes (the container of a .docx export), and entries copied
 * from one package into another as they are. Reading Office files through it
 * is tested with the formats (formats.test.ts).
 */
import { readFileSync } from "node:fs";
import { deflateRawSync } from "node:zlib";
import { describe, expect, test } from "vitest";
import { extractDocx } from "../../src/core/documents/formats/docx";
import {
  crc32,
  openPackage,
  type RawEntry,
  writeZip,
  type ZipWriteEntry,
} from "../../src/core/documents/formats/zip";
import { unzip } from "../helpers/docx";
import { zipFingerprint } from "../helpers/zip";

/** Text, a name outside ASCII, binary data and an empty entry. */
const ENTRIES = [
  { name: "[Content_Types].xml", data: '<?xml version="1.0"?><Types/>' },
  {
    name: "word/document.xml",
    data: `<w:document>${"<w:p>Spring tides.</w:p>".repeat(40)}</w:document>`,
  },
  { name: "word/media/潮汐.bin", data: new Uint8Array([0, 1, 2, 3, 255, 254, 253, 0, 0, 0]) },
  { name: "word/empty.xml", data: "" },
];

/** A Word file the docx package wrote, with JSZip. */
const DOCX = new Uint8Array(
  readFileSync(new URL("../fixtures/formats/Coastal Flood Risk Review.docx", import.meta.url)),
);

describe("Writing", () => {
  test("the bytes it writes don't change: every entry deflated, a fixed time, UTF-8 names", () => {
    const archive = writeZip(ENTRIES, deflateRawSync);

    expect(zipFingerprint(archive)).toBe(
      "2375796cf454cb68add224239233101b834e5b6f7c4885ff7c498df32d9d7014",
    );
    expect([...unzip(archive)]).toEqual(ENTRIES.map(({ name, data }) => [name, Buffer.from(data)]));
  });

  test("its checksum is ZIP's CRC-32", () => {
    expect(crc32(new Uint8Array())).toBe(0);
    expect(crc32(new TextEncoder().encode("123456789"))).toBe(0xcbf43926);
  });
});

describe("Copying", () => {
  test("an entry copied from another package keeps its compressed bytes, beside the ones written anew", async () => {
    // Deflated faster than the writer deflates, and stored: bytes the writer wouldn't make.
    const text = "Spring tides happen at new and full moon; neap tides at the quarters. ".repeat(
      200,
    );
    const fast = (data: Uint8Array) => deflateRawSync(data, { level: 1 });
    const kept = new TextEncoder().encode("Kept as it is.");
    const source = openPackage(
      writeZip(
        [
          { name: "word/document.xml", data: text },
          { name: "word/media/", raw: { method: 0, crc32: 0, size: 0, data: new Uint8Array() } },
          { name: "word/notes.txt", raw: { method: 0, crc32: crc32(kept), size: 14, data: kept } },
          { name: "docProps/core.xml", data: "<cp:coreProperties/>" },
        ],
        fast,
      ),
    );
    const document = source.raw("word/document.xml") as RawEntry;
    expect(Buffer.from(document.data).equals(deflateRawSync(text))).toBe(false);

    const entries = source
      .names()
      .map(
        (name): ZipWriteEntry =>
          name === "docProps/core.xml"
            ? { name, data: "<cp:coreProperties><dc:title>Tides</dc:title></cp:coreProperties>" }
            : { name, raw: source.raw(name) as RawEntry },
      );
    const bytes = writeZip(entries, deflateRawSync);
    const copy = openPackage(bytes);

    expect(copy.names()).toEqual(source.names());
    for (const name of ["word/document.xml", "word/media/", "word/notes.txt"]) {
      expect(copy.raw(name)).toEqual(source.raw(name));
    }
    expect(Buffer.from(copy.raw("word/document.xml")?.data ?? []).equals(document.data)).toBe(true);
    expect(await copy.readText("word/document.xml")).toBe(text);
    expect(await copy.readText("word/notes.txt")).toBe("Kept as it is.");
    expect(await copy.readText("docProps/core.xml")).toBe(
      "<cp:coreProperties><dc:title>Tides</dc:title></cp:coreProperties>",
    );
    // The checksums and sizes it records are right: unzip checks them.
    expect([...unzip(bytes).keys()]).toEqual(source.names());
  });

  test("a Word file whose entries are all copied reads as the original does", async () => {
    const source = openPackage(DOCX);

    const bytes = writeZip(
      source.names().map((name) => ({ name, raw: source.raw(name) as RawEntry })),
      deflateRawSync,
    );

    for (const name of source.names()) {
      expect(openPackage(bytes).raw(name)).toEqual(source.raw(name));
    }
    expect((await extractDocx(bytes)).units).toEqual((await extractDocx(DOCX)).units);
  });

  test("no such entry: nothing to copy", () => {
    expect(openPackage(DOCX).raw("word/comments.xml")).toBeUndefined();
  });
});
