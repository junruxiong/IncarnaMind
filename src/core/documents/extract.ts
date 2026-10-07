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

/**
 * A PDF's creation dates as written (#53): its document information
 * dictionary's CreationDate, a PDF date string, and its XMP metadata's
 * xmp:CreateDate, an ISO 8601 date. Null where there is none. Never its
 * ModDate. Only the metadata is read, not the pages. Throws if the PDF
 * can't be opened.
 */
export async function pdfCreationDates(
  bytes: Uint8Array,
): Promise<{ info: string | null; xmp: string | null }> {
  const { getDocument } = await loadPdfJs();
  // pdf.js may take ownership of the buffer, so give it a copy.
  const task = getDocument({ data: new Uint8Array(bytes), verbosity: 0 });
  try {
    const document = await task.promise;
    const { info, metadata } = await document.getMetadata();
    const created = isRecord(info) ? info.CreationDate : undefined;
    const xmp = metadata ? XMP_CREATE_DATE.exec(String(metadata.getRaw())) : null;
    return {
      info: typeof created === "string" ? created : null,
      xmp: xmp ? (xmp[2] ?? xmp[3] ?? null) : null,
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
