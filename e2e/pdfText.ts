import { dirname, join } from "node:path";
import { CJK } from "../src/shared/text";

/** pdf.js's data files, which reading CJK text needs. */
function dataUrls() {
  const packageDir = dirname(require.resolve("pdfjs-dist/package.json"));
  return {
    cMapUrl: `${join(packageDir, "cmaps")}/`,
    standardFontDataUrl: `${join(packageDir, "standard_fonts")}/`,
  };
}

/** A line broken between two CJK characters isn't a space. */
const CJK_LINE_WRAP = new RegExp(`([${CJK}])\\s*\\n\\s*(?=[${CJK}])`, "gu");

/**
 * What a PDF says, page by page, as pdf.js reads it (the library the app reads PDFs with):
 * each page's text in drawing order, a line to each line, without opening the file in an app.
 */
export async function pdfText(bytes: Uint8Array): Promise<string[]> {
  const { getDocument } = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const task = getDocument({
    data: new Uint8Array(bytes),
    ...dataUrls(),
    cMapPacked: true,
    disableFontFace: true,
    useSystemFonts: false,
    verbosity: 0,
  });
  try {
    const pdf = await task.promise;
    const pages: string[] = [];
    for (let number = 1; number <= pdf.numPages; number++) {
      const page = await pdf.getPage(number);
      let text = "";
      for (const item of (await page.getTextContent()).items) {
        if (!("str" in item)) continue;
        text += item.str;
        if (item.hasEOL) text += "\n";
      }
      pages.push(text.replaceAll("\u0000", "").replace(CJK_LINE_WRAP, "$1").trim());
      page.cleanup();
    }
    return pages;
  } finally {
    await task.destroy();
  }
}
