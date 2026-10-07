/**
 * Reading every format but PDF into Units (ADR-0011), with our own small ZIP
 * and XML reader and no dependency. Pure, with no Node imports: the
 * processing worker extracts with it, and the Document viewer reads files
 * with the very same code, so what it shows and highlights is what was
 * indexed. PDFs are read with pdf.js (../extract).
 */

import type { TextUnit } from "../../../shared/units";
import type { DocumentKind } from "../../api";
import { decodeText } from "../decode";
import { extractCsv } from "./csv";
import { extractDocx } from "./docx";
import { ExtractionError } from "./errors";
import { extractPptx } from "./pptx";
import { lineUnits, markdownUnits } from "./text";
import { extractXlsx } from "./xlsx";

export { ExtractionError } from "./errors";

/** The Units of a file of any kind but PDF. Throws `ExtractionError` when it can't be read. */
export async function extractUnits(
  kind: Exclude<DocumentKind, "pdf">,
  bytes: Uint8Array,
): Promise<TextUnit[]> {
  switch (kind) {
    case "text":
    case "markdown": {
      const text = decodeText(bytes);
      if (text === null) {
        throw new ExtractionError("unreadable", "The file holds binary data, not text.");
      }
      // The Units' offsets in the file are for the viewer; what is stored is the Unit itself.
      return (kind === "markdown" ? markdownUnits(text) : lineUnits(text)).map(
        ({ start: _start, end: _end, ...unit }) => unit,
      );
    }
    case "docx":
      return (await extractDocx(bytes)).units;
    case "pptx":
      return (await extractPptx(bytes)).units;
    case "xlsx":
      return (await extractXlsx(bytes)).units;
    case "csv":
      return extractCsv(bytes).units;
  }
}
