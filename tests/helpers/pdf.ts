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

/** An entry of a PDF's outline (its bookmarks). */
export interface PdfOutlineEntry {
  title: string;
  /** The page it goes to, from 1. */
  page?: number;
  /**
   * How it says where it goes: an explicit destination (`[page /XYZ …]`, the
   * default), a named destination looked up in the catalog's `/Dests`, a
   * GoTo action, or a web link (`url`) instead of a page.
   */
  via?: "explicit" | "named" | "action";
  url?: string;
  /** Shown expanded at first (a positive `/Count`). */
  open?: boolean;
  items?: readonly PdfOutlineEntry[];
}

export interface PdfOptions {
  outline?: readonly PdfOutlineEntry[];
  /**
   * The document information dictionary (the trailer's /Info), each entry
   * written as a literal string: `{ CreationDate: "D:20190304103000Z" }`.
   * Latin-1 characters only.
   */
  info?: Readonly<Record<string, string>>;
  /** The /Info object as written, in place of `info`: for malformed files. */
  rawInfo?: string;
  /** An XMP metadata stream, as written, for the catalog's /Metadata. ASCII only. */
  xmp?: string;
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

/**
 * Adds an outline's objects, numbered from `firstId`, and returns the
 * catalog's entries for it: `/Outlines` and, for named destinations, `/Dests`.
 */
function addOutline(
  objects: string[],
  firstId: number,
  outline: readonly PdfOutlineEntry[],
  pageRef: (page: number) => string,
): string {
  let nextId = firstId;
  const named: string[] = [];

  const destination = (entry: PdfOutlineEntry): string => {
    if (entry.url) return `/A << /S /URI /URI (${escapeLatin(entry.url)}) >>`;
    const explicit = `[${pageRef(entry.page ?? 1)} /XYZ 0 792 0]`;
    if (entry.via === "named") {
      const name = `section${named.length + 1}`;
      named.push(`/${name} ${explicit}`);
      return `/Dest (${name})`;
    }
    if (entry.via === "action") return `/A << /S /GoTo /D ${explicit} >>`;
    return `/Dest ${explicit}`;
  };

  /** The entries shown under an open entry: its children, and theirs if open. */
  const shown = (entries: readonly PdfOutlineEntry[]): number =>
    entries.reduce((count, entry) => count + 1 + (entry.open ? shown(entry.items ?? []) : 0), 0);

  const add = (entries: readonly PdfOutlineEntry[], parentId: number) => {
    const ids = entries.map(() => nextId++);
    entries.forEach((entry, index) => {
      const id = ids[index] as number;
      const children = entry.items ?? [];
      let object = `<< /Title (${escapeLatin(entry.title)}) /Parent ${parentId} 0 R ${destination(entry)}`;
      if (index > 0) object += ` /Prev ${ids[index - 1]} 0 R`;
      if (index < ids.length - 1) object += ` /Next ${ids[index + 1]} 0 R`;
      if (children.length > 0) {
        const range = add(children, id);
        const count = entry.open ? shown(children) : -children.length;
        object += ` /First ${range.first} 0 R /Last ${range.last} 0 R /Count ${count}`;
      }
      objects[id] = `${object} >>`;
    });
    return { first: ids[0], last: ids.at(-1) };
  };

  const rootId = nextId++;
  const range = add(outline, rootId);
  objects[rootId] =
    `<< /Type /Outlines /First ${range.first} 0 R /Last ${range.last} 0 R /Count ${shown(outline)} >>`;
  const dests = named.length > 0 ? ` /Dests << ${named.join(" ")} >>` : "";
  return ` /Outlines ${rootId} 0 R /PageMode /UseOutlines${dests}`;
}

/** Returns the bytes of a PDF with one page per entry, and the outline and metadata given. */
export function buildPdf(pages: readonly PdfPage[], options: PdfOptions = {}): Uint8Array {
  // Objects 1–5 are fixed; each page then adds a page object and a content stream.
  const objects: string[] = [];
  const pageIds = pages.map((_, index) => 6 + index * 2);
  const outline =
    options.outline && options.outline.length > 0
      ? addOutline(objects, 6 + pages.length * 2, options.outline, (page) => {
          const id = pageIds[page - 1];
          if (id === undefined) throw new Error(`There is no page ${page}.`);
          return `${id} 0 R`;
        })
      : "";
  // The metadata, if any, after every other object.
  let nextId = Math.max(objects.length, 6 + pages.length * 2);
  let info = "";
  if (options.info || options.rawInfo !== undefined) {
    const id = nextId++;
    objects[id] =
      options.rawInfo ??
      `<< ${Object.entries(options.info ?? {})
        .map(([key, value]) => `/${key} (${escapeLatin(value)})`)
        .join(" ")} >>`;
    info = ` /Info ${id} 0 R`;
  }
  let metadata = "";
  if (options.xmp !== undefined) {
    const id = nextId++;
    objects[id] =
      `<< /Type /Metadata /Subtype /XML /Length ${options.xmp.length} >>\n` +
      `stream\n${options.xmp}\nendstream`;
    metadata = ` /Metadata ${id} 0 R`;
  }
  objects[1] = `<< /Type /Catalog /Pages 2 0 R${outline}${metadata} >>`;
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
  pdf += `trailer\n<< /Size ${objects.length} /Root 1 0 R${info} >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Uint8Array.from(Buffer.from(pdf, "latin1"));
}
