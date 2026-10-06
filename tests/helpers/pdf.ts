/**
 * Builds tiny, valid PDFs for tests, so fixtures stay readable and small
 * instead of being committed as binaries.
 */

export interface PdfPage {
  /** Lines of text drawn top to bottom in Helvetica. Latin-1 characters only. */
  lines?: readonly string[];
  /**
   * Lines of Chinese text drawn with STSong-Light, a CJK font PDF readers
   * supply themselves, so extracting it needs pdf.js's CMap files.
   */
  chineseLines?: readonly string[];
  /** Draws a small greyscale image, like a scanned page. */
  image?: boolean;
}

const LINE_HEIGHT = 14;
const TOP = 760;

const escapeLatin = (text: string) => text.replace(/[\\()]/g, (char) => `\\${char}`);

const toUcs2Hex = (text: string) =>
  [...text]
    .map((char) => {
      const code = char.codePointAt(0) ?? 0;
      if (code > 0xffff) throw new Error("Only Basic Multilingual Plane characters are supported.");
      return code.toString(16).padStart(4, "0");
    })
    .join("");

function contentStream(page: PdfPage): string {
  const parts: string[] = [];
  let y = TOP;
  for (const line of page.lines ?? []) {
    if (/[^\x20-\xff]/.test(line)) throw new Error(`Not Latin-1: ${line}`);
    parts.push(`BT /F1 11 Tf 72 ${y} Td (${escapeLatin(line)}) Tj ET`);
    y -= LINE_HEIGHT;
  }
  for (const line of page.chineseLines ?? []) {
    parts.push(`BT /F2 11 Tf 72 ${y} Td <${toUcs2Hex(line)}> Tj ET`);
    y -= LINE_HEIGHT;
  }
  if (page.image) {
    // A 2×2 greyscale inline image, scaled up to 300pt square.
    parts.push(`q 300 0 0 300 150 300 cm BI /W 2 /H 2 /CS /G /BPC 8 ID \x00\xff\xff\x00 EI Q`);
  }
  return parts.join("\n");
}

/** Returns the bytes of a PDF with one page per entry. */
export function buildPdf(pages: readonly PdfPage[]): Uint8Array {
  // Objects 1–5 are fixed; each page then adds a page object and a content stream.
  const objects: string[] = [];
  const pageIds = pages.map((_, index) => 6 + index * 2);
  objects[1] = "<< /Type /Catalog /Pages 2 0 R >>";
  objects[2] = `<< /Type /Pages /Kids [${pageIds.map((id) => `${id} 0 R`).join(" ")}] /Count ${pages.length} >>`;
  objects[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";
  objects[4] =
    "<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H " +
    "/DescendantFonts [<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light " +
    "/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 4 >> /FontDescriptor 5 0 R >>] >>";
  objects[5] =
    "<< /Type /FontDescriptor /FontName /STSong-Light /Flags 6 /FontBBox [-25 -254 1000 880] " +
    "/ItalicAngle 0 /Ascent 880 /Descent -120 /CapHeight 880 /StemV 93 >>";
  pages.forEach((page, index) => {
    const pageId = 6 + index * 2;
    const content = contentStream(page);
    objects[pageId] =
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] ` +
      `/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents ${pageId + 1} 0 R >>`;
    objects[pageId + 1] = `<< /Length ${content.length} >>\nstream\n${content}\nendstream`;
  });

  // Latin-1 throughout, so string length equals byte length for the xref offsets.
  let pdf = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n";
  const offsets: number[] = [];
  for (let id = 1; id < objects.length; id++) {
    offsets[id] = pdf.length;
    pdf += `${id} 0 obj\n${objects[id]}\nendobj\n`;
  }
  const xrefOffset = pdf.length;
  pdf += `xref\n0 ${objects.length}\n0000000000 65535 f \n`;
  for (let id = 1; id < objects.length; id++) {
    pdf += `${String(offsets[id]).padStart(10, "0")} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Uint8Array.from(Buffer.from(pdf, "latin1"));
}
