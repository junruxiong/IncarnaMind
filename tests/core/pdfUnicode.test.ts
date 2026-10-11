import { deflateSync } from "node:zlib";
import { describe, expect, test } from "vitest";
import { extractText } from "../../src/core/documents/extract";
import { unifyIdeographs } from "../../src/core/exports/pdfUnicode";

/**
 * A one-page PDF laid out as Chromium writes one: compressed streams, its font's text in a
 * ToUnicode map, one cross-reference table. Its page draws the codes in `drawn`.
 */
function chromiumLikePdf(cmapBody: string, drawn: string): Uint8Array {
  const cmap = `/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
${cmapBody}endcmap
CMapName currentdict /CMap defineresource pop
end
end`;
  const stream = (data: Buffer) => [
    Buffer.from(`<</Filter /FlateDecode\n/Length ${data.length}>> stream\n`, "latin1"),
    data,
    Buffer.from("\nendstream", "latin1"),
  ];
  const objects: Buffer[][] = [
    [Buffer.from("<</Type /Catalog /Pages 2 0 R>>")],
    [Buffer.from("<</Type /Pages /Kids [3 0 R] /Count 1>>")],
    [
      Buffer.from(
        "<</Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>",
      ),
    ],
    stream(deflateSync(Buffer.from(`BT /F1 24 Tf 20 100 Td (${drawn}) Tj ET`, "latin1"))),
    [Buffer.from("<</Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R>>")],
    stream(deflateSync(Buffer.from(cmap, "latin1"))),
  ];
  const parts: Buffer[] = [Buffer.from("%PDF-1.4\n")];
  const offsets: number[] = [];
  let length = parts[0]?.length ?? 0;
  objects.forEach((body, index) => {
    offsets.push(length);
    const object = Buffer.concat([
      Buffer.from(`${index + 1} 0 obj\n`),
      ...body,
      Buffer.from("\nendobj\n"),
    ]);
    parts.push(object);
    length += object.length;
  });
  const xref = [
    "xref",
    `0 ${objects.length + 1}`,
    "0000000000 65535 f\r",
    ...offsets.map((offset) => `${String(offset).padStart(10, "0")} 00000 n\r`),
    "trailer",
    `<</Size ${objects.length + 1}\n/Root 1 0 R>>`,
    "startxref",
    String(length),
    "%%EOF",
  ].join("\n");
  parts.push(Buffer.from(xref));
  return new Uint8Array(Buffer.concat(parts));
}

const textOf = async (pdf: Uint8Array) => (await extractText("pdf", pdf)).units[0]?.text;

describe("Chinese text in a printed PDF", () => {
  test("a glyph mapped to a Kangxi radical copies as its ideograph, in single mappings and in ranges", async () => {
    // 大 and 月 drawn with radicals' code points (⼤ U+2F24, ⽉ U+2F49), as Chromium writes them;
    // 潮 and the comma as they are; a range from ⼈ (U+2F08) over two codes.
    const pdf = chromiumLikePdf(
      "4 beginbfchar\n<41> <2F24>\n<42> <6F6E>\n<43> <2F49>\n<44> <FF0C>\nendbfchar\n1 beginbfrange\n<45> <46> <2F08>\nendbfrange\n",
      "ABCDEF",
    );
    expect(await textOf(pdf)).toBe("⼤潮⽉，⼈⼉");

    const fixed = unifyIdeographs(pdf);

    expect(await textOf(fixed)).toBe("大潮月，人儿");
  });

  test("a PDF with no radicals, or one not laid out as expected, comes back as it was", async () => {
    const plain = chromiumLikePdf("1 beginbfchar\n<41> <6F6E>\nendbfchar\n", "A");
    expect(unifyIdeographs(plain)).toBe(plain);
    const notAPdf = new TextEncoder().encode("%PDF-1.4\nnot really");
    expect(unifyIdeographs(notAPdf)).toBe(notAPdf);
  });
});
