/**
 * Extracts a Document's text as Units (ADR-0011): a PDF's pages with pdf.js,
 * every other kind with our own readers (./formats). Runs in the processing
 * worker: parsing a large PDF takes seconds.
 */
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { CJK } from "../../shared/text";
import type { TextUnit } from "../../shared/units";
import type { DocumentKind } from "../api";
import { isRecord } from "../errors";
import { ExtractionError, extractUnits } from "./formats";

export { ExtractionError };

export interface ExtractedText {
  /** PDFs only. */
  pageCount: number | null;
  units: TextUnit[];
}

export async function extractText(kind: DocumentKind, bytes: Uint8Array): Promise<ExtractedText> {
  if (kind === "pdf") {
    const units = await extractPdf(bytes);
    return { pageCount: units.length, units };
  }
  return { pageCount: null, units: await extractUnits(kind, bytes) };
}

type PdfJs = typeof import("pdfjs-dist/legacy/build/pdf.mjs");

let pdfjs: Promise<PdfJs> | undefined;

/** pdf.js's Node-compatible ("legacy") build, loaded on first use. */
const loadPdfJs = () => {
  pdfjs ??= import("pdfjs-dist/legacy/build/pdf.mjs");
  return pdfjs;
};

/** pdf.js reads character maps (for CJK fonts) and standard font data from its package folder. */
function pdfjsDataUrls() {
  const packageDir = dirname(createRequire(import.meta.url).resolve("pdfjs-dist/package.json"));
  return {
    cMapUrl: `${join(packageDir, "cmaps")}/`,
    standardFontDataUrl: `${join(packageDir, "standard_fonts")}/`,
  };
}

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

async function extractPdf(bytes: Uint8Array): Promise<TextUnit[]> {
  const { getDocument } = await loadPdfJs();
  const task = getDocument({
    // pdf.js may take ownership of the buffer, so give it a copy.
    data: new Uint8Array(bytes),
    ...pdfjsDataUrls(),
    cMapPacked: true,
    disableFontFace: true,
    useSystemFonts: false,
    verbosity: 0, // errors only
  });
  try {
    const document = await task.promise;
    const pages: TextUnit[] = [];
    for (let number = 1; number <= document.numPages; number++) {
      const page = await document.getPage(number);
      const content = await page.getTextContent();
      pages.push({
        page: number,
        kind: "page",
        label: null,
        text: pageText(content.items),
        anchors: null,
      });
      page.cleanup();
    }
    return pages;
  } catch (error) {
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
