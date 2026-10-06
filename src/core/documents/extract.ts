/**
 * Extracts a Document's text, page by page. Runs in the processing worker:
 * parsing a large PDF takes seconds.
 */
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import type { DocumentFailureReason, DocumentKind } from "../api";
import { decodeText } from "./decode";
import type { PageText } from "./passages";
import { CJK } from "./text";

/** Why a file's text couldn't be extracted. */
export class ExtractionError extends Error {
  override name = "ExtractionError";
  constructor(
    readonly reason: DocumentFailureReason,
    message: string,
  ) {
    super(message);
  }
}

export interface ExtractedText {
  /** PDFs only. */
  pageCount: number | null;
  pages: PageText[];
}

export async function extractText(kind: DocumentKind, bytes: Uint8Array): Promise<ExtractedText> {
  if (kind === "pdf") {
    const pages = await extractPdf(bytes);
    return { pageCount: pages.length, pages };
  }
  const text = decodeText(bytes);
  if (text === null)
    throw new ExtractionError("unreadable", "The file holds binary data, not text.");
  return { pageCount: null, pages: [{ page: null, text }] };
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

async function extractPdf(bytes: Uint8Array): Promise<PageText[]> {
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
    const pages: PageText[] = [];
    for (let number = 1; number <= document.numPages; number++) {
      const page = await document.getPage(number);
      const content = await page.getTextContent();
      pages.push({ page: number, text: pageText(content.items) });
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
