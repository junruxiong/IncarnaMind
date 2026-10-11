/**
 * A document to print as a PDF (#100): what the core makes of a Mind, which the desktop app's
 * hidden print page lays out in pages, each footnote at the foot of the page its marker is on,
 * and prints.
 */
export interface PrintDocument {
  /** The `lang` of the page: line breaking, and which faces Chinese text takes. */
  language: "en" | "zh-CN";
  /** Letter in English, A4 otherwise, as the .docx export chooses. */
  paper: PrintPaper;
  /** The PDF's title. */
  title: string;
  /**
   * The body, as HTML the core escaped. Each footnote's marker is
   * `<sup class="note-ref" data-note="N">N</sup>`, N counting from 1 in reading order.
   */
  body: string;
  /** Each footnote's text, as HTML: footnote N is `notes[N - 1]`. */
  notes: string[];
}

export type PrintPaper = "a4" | "letter";

/** A page's size and margins in CSS pixels (96 to the inch), the same in the core and the print page. */
export const PAPER: Readonly<
  Record<PrintPaper, { width: number; height: number; margin: number; css: string }>
> = {
  // 210 × 297 mm, with 20 mm margins.
  a4: { width: 793.7, height: 1122.5, margin: 75.6, css: "A4" },
  // 8.5 × 11 in, with 0.8 in margins.
  letter: { width: 816, height: 1056, margin: 76.8, css: "letter" },
};

/** What the print page names its entry point: the main process calls it with the document. */
export const PRINT_ENTRY = "incarnamindPrint";
