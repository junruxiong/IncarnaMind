/**
 * Extracts a Document's text as Units (ADR-0011): a PDF's pages with pdf.js,
 * every other kind with our own readers (./formats). Runs in the processing
 * worker: parsing a large PDF takes seconds. A PDF's scanned pages are read
 * from their pictures (#397), when reading is given.
 */
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { CJK } from "../../shared/text";
import type { TextUnit } from "../../shared/units";
import type { PagePicture, RecognizedLine } from "../adapters";
import type { DocumentKind } from "../api";
import { isRecord } from "../errors";
import { ExtractionError, extractUnits } from "./formats";
import { scannedUnit } from "./scan/layout";
import { type CanvasFactory, isScanned, pagePicture } from "./scan/pages";

export { ExtractionError };

/** Reading a PDF's scanned pages (#397), as the processing worker is given it. */
export interface PageReading {
  /** What reads them now (`UnitScan.reader`). */
  reader: string;
  /** Pages of this version read already, used as they are, by page number. */
  keep: ReadonlyMap<number, TextUnit>;
  /** The lines on a page's picture. Rejects with `ReadingUnavailable` when no page can be read now. */
  read(page: number, picture: PagePicture): Promise<RecognizedLine[]>;
  /** `read` of `of` scanned pages are read; `unit` is the one just read. */
  progress(read: number, of: number, unit: TextUnit | null): void;
  /** Aborted when the job is stopped: no further page is read. */
  signal: AbortSignal;
}

/** No scanned page can be read now: the text recognition model can't be downloaded or started. */
export class ReadingUnavailable extends Error {
  override name = "ReadingUnavailable";
}

export interface ExtractedText {
  /** PDFs only. */
  pageCount: number | null;
  units: TextUnit[];
  /** Scanned pages that couldn't be read, and why; null when every one was. */
  unread: { pages: number[]; message: string } | null;
}

export async function extractText(
  kind: DocumentKind,
  bytes: Uint8Array,
  reading: PageReading | null = null,
): Promise<ExtractedText> {
  if (kind === "pdf") return extractPdf(bytes, reading);
  return { pageCount: null, units: await extractUnits(kind, bytes), unread: null };
}

type PdfJs = typeof import("pdfjs-dist/legacy/build/pdf.mjs");

let pdfjs: Promise<PdfJs> | undefined;

/** pdf.js's Node-compatible ("legacy") build, loaded on first use. */
const loadPdfJs = () => {
  pdfjs ??= import("pdfjs-dist/legacy/build/pdf.mjs");
  return pdfjs;
};

/**
 * pdf.js reads character maps (for CJK fonts), standard font data, and the
 * decoders a scan's pictures may need (JPEG 2000, JBIG2) from its package folder.
 */
function pdfjsDataUrls() {
  const packageDir = dirname(createRequire(import.meta.url).resolve("pdfjs-dist/package.json"));
  return {
    cMapUrl: `${join(packageDir, "cmaps")}/`,
    standardFontDataUrl: `${join(packageDir, "standard_fonts")}/`,
    wasmUrl: `${join(packageDir, "wasm")}/`,
  };
}

/** A scan's picture larger than this many pixels isn't drawn: reading it would take too much memory. */
const MAX_PICTURE_PIXELS = 50_000_000;

// Lines inside a paragraph of CJK text: a PDF breaks them anywhere, and the break isn't a space.
const CJK_LINE_WRAP = new RegExp(`([${CJK}])[^\\S\\n]*\\n[^\\S\\n]*(?=[${CJK}])`, "gu");

function pageText(items: readonly object[]): string {
  let text = "";
  for (const item of items) {
    // Marked-content boundaries carry no text.
    if (!("str" in item) || typeof item.str !== "string") continue;
    text += item.str;
    if ("hasEOL" in item && item.hasEOL === true) text += "\n";
  }
  return text.replaceAll("\u0000", "").replace(/\r\n?/g, "\n").replace(CJK_LINE_WRAP, "$1").trim();
}

/** XMP's creation date, written as an element or as an attribute of its description. */
const XMP_CREATE_DATE = /xmp:CreateDate\s*(?:=\s*(["'])([^"'<>]*)\1|>([^<]*)<)/i;

/** What a PDF says of itself, as `pdfMetadata` reads it. */
export interface PdfMetadata {
  /** Its Info dictionary's CreationDate, a PDF date string. */
  info: string | null;
  /** Its XMP metadata's xmp:CreateDate, an ISO 8601 date. */
  xmp: string | null;
  /** Its Info dictionary's Title (#213). */
  title: string | null;
  /** Its Info dictionary's Author (#286). */
  author: string | null;
  /** Its XMP metadata packet, as written (#286). */
  xmpPacket: string | null;
  /** The text of its first `pages` pages, as extracted, running headers and all (#286). */
  firstPages: string[];
}

/**
 * A PDF's metadata as written: its document information dictionary's
 * CreationDate (#53), Title (#213) and Author, and its XMP metadata, whose
 * xmp:CreateDate is read here; never its ModDate. With `pages`, the text of
 * its first pages too (#286: a DOI or an arXiv id is printed there).
 * Throws if the PDF can't be opened.
 */
export async function pdfMetadata(bytes: Uint8Array, pages = 0): Promise<PdfMetadata> {
  const { getDocument } = await loadPdfJs();
  // pdf.js may take ownership of the buffer, so give it a copy.
  const task = getDocument({
    data: new Uint8Array(bytes),
    ...(pages > 0 && { ...pdfjsDataUrls(), cMapPacked: true }),
    verbosity: 0,
  });
  try {
    const document = await task.promise;
    const { info, metadata } = await document.getMetadata();
    const raw = metadata ? String(metadata.getRaw()) : null;
    const xmp = raw ? XMP_CREATE_DATE.exec(raw) : null;
    const entry = (name: string) => {
      const value = isRecord(info) ? info[name] : undefined;
      return typeof value === "string" ? value : null;
    };
    const firstPages: string[] = [];
    for (let number = 1; number <= Math.min(pages, document.numPages); number++) {
      const page = await document.getPage(number);
      firstPages.push(pageText((await page.getTextContent()).items));
      page.cleanup();
    }
    return {
      info: entry("CreationDate"),
      xmp: xmp ? (xmp[2] ?? xmp[3] ?? null) : null,
      title: entry("Title"),
      author: entry("Author"),
      xmpPacket: raw,
      firstPages,
    };
  } finally {
    await task.destroy();
  }
}

async function extractPdf(bytes: Uint8Array, reading: PageReading | null): Promise<ExtractedText> {
  const { getDocument, OPS } = await loadPdfJs();
  const task = getDocument({
    // pdf.js may take ownership of the buffer, so give it a copy.
    data: new Uint8Array(bytes),
    ...pdfjsDataUrls(),
    cMapPacked: true,
    disableFontFace: true,
    useSystemFonts: false,
    maxImageSize: MAX_PICTURE_PIXELS,
    verbosity: 0, // errors only
  });
  try {
    const document = await task.promise;
    const pages: TextUnit[] = [];
    const scanned: number[] = [];
    for (let number = 1; number <= document.numPages; number++) {
      const page = await document.getPage(number);
      const content = await page.getTextContent();
      const text = pageText(content.items);
      pages.push({ page: number, kind: "page", label: null, text, anchors: null });
      if (reading && (await isScanned(page, text, OPS))) scanned.push(number);
      page.cleanup();
    }
    const unread =
      reading && scanned.length > 0
        ? await readScannedPages(document, pages, scanned, reading)
        : null;
    return { pageCount: pages.length, units: pages, unread };
  } catch (error) {
    if (reading?.signal.aborted) throw reading.signal.reason;
    const name = error instanceof Error ? error.name : "";
    const message = error instanceof Error ? error.message : String(error);
    if (name === "PasswordException") {
      throw new ExtractionError("password-protected", `The PDF needs a password: ${message}`);
    }
    throw new ExtractionError("unreadable", `The PDF couldn't be read: ${message}`);
  } finally {
    await task.destroy();
  }
}

/**
 * Reads the scanned pages of a PDF into `pages`, in order: each kept page as
 * it was, the others from their pictures. A page that can't be read keeps
 * its text layer, marked unread; once the model can't be had, neither can
 * the pages after. Throws the signal's reason when the job is stopped.
 */
async function readScannedPages(
  document: Awaited<ReturnType<PdfJs["getDocument"]>["promise"]>,
  pages: TextUnit[],
  scanned: readonly number[],
  reading: PageReading,
): Promise<ExtractedText["unread"]> {
  const factory = document.canvasFactory as CanvasFactory;
  const unread: number[] = [];
  let message: string | null = null;
  let unavailable = false;
  reading.progress(0, scanned.length, null);
  for (const [index, number] of scanned.entries()) {
    reading.signal.throwIfAborted();
    const at = number - 1;
    const layer = pages[at] as TextUnit;
    let unit = reading.keep.get(number) ?? null;
    if (!unit && !unavailable) {
      const page = await document.getPage(number);
      try {
        const picture = await pagePicture(page, factory);
        unit = scannedUnit(number, await reading.read(number, picture), picture, reading.reader);
      } catch (error) {
        reading.signal.throwIfAborted();
        message ??= error instanceof Error ? error.message : String(error);
        if (error instanceof ReadingUnavailable) unavailable = true;
      } finally {
        page.cleanup();
      }
    }
    if (!unit) {
      unread.push(number);
      unit = { ...layer, label: { scan: { reader: reading.reader, unread: true } } };
    }
    pages[at] = unit;
    reading.progress(index + 1, scanned.length, unit);
  }
  return unread.length > 0 ? { pages: unread, message: message ?? "" } : null;
}
