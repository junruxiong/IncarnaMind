/**
 * The ZIP archives IncarnaMind writes: the container of a .docx export.
 */
import { describe, expect, test } from "vitest";
import { zip } from "../../src/core/exports/zip";
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

describe("Writing", () => {
  test("the bytes it writes don't change: every entry deflated, a fixed time, UTF-8 names", () => {
    const archive = zip(ENTRIES);

    expect(zipFingerprint(archive)).toBe(
      "2375796cf454cb68add224239233101b834e5b6f7c4885ff7c498df32d9d7014",
    );
    expect([...unzip(archive)]).toEqual(ENTRIES.map(({ name, data }) => [name, Buffer.from(data)]));
  });
});
