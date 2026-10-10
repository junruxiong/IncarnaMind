/**
 * Units (ADR-0011): the spans a Document's text is stored in, in reading
 * order, and what a Citation's Location points at. A Unit is a PDF's page, a
 * deck's slide (its speaker notes included), a heading section of a Word or
 * Markdown file, a block of rows of one sheet of a spreadsheet or CSV, or a
 * block of lines of a plain-text file. Each is one `document_pages` row of a
 * version: `page` is the Unit's number, from 1, which for a PDF is its page
 * and for a deck its slide.
 *
 * A Citation names one Unit, or two consecutive ones in the same sheet or
 * Document, and its quote is checked against their stored text. Anchors map
 * spans of a Unit's text back to what a preview can highlight (a cell, a
 * paragraph, a slide's notes, a Word comment); only the viewer, a Citation's
 * label and the marks in the Passages a model reads need them.
 *
 * Pure, with no Node or DOM imports: the processing worker, the core and the
 * renderer all use it.
 */

/**
 * - "page": a PDF's page.
 * - "slide": a deck's slide, with its speaker notes.
 * - "section": the text under one heading of a Word or Markdown file (or the
 *   part of a long one), or what comes before the first heading.
 * - "rows": a block of rows of one sheet (a CSV has one sheet).
 * - "lines": a block of lines of a plain-text file.
 * - "text": a whole TXT or Markdown file, as text was stored before Units
 *   (migration 22); kept only for the old versions Citations quote.
 */
export type UnitKind = "page" | "slide" | "section" | "rows" | "lines" | "text";

/**
 * What a Unit's Location label is made of, by kind; stored as JSON beside
 * the Unit's text. Language-neutral: the label is worded when shown.
 */
export interface UnitLabel {
  /** section: the heading path, outermost first; empty before the first heading. */
  path?: string[];
  /** section: which part of a long section this is, from 2 (the first part has none). */
  part?: number;
  /** section: the Word file's footnotes and endnotes, gathered at the end. */
  notes?: boolean;
  /**
   * section: the Word comments anchored in it, in order, read after its own
   * text: each one's anchor and its author.
   */
  comments?: UnitComment[];
  /** slide: its title, if it has one. */
  title?: string;
  /** slide: hidden in a slide show. */
  hidden?: boolean;
  /** rows: the sheet's name; none for a CSV. */
  sheet?: string;
  /** rows and lines: the first and last row (or line) number, from 1, as the file numbers them. */
  from?: number;
  to?: number;
  /** rows: the row number of each line of the Unit's text, in order. */
  rows?: number[];
  /** rows: the sheet's first column (from 0): each line lists the cells from it, tab-separated. */
  column?: number;
  /** rows: the Unit's first line repeats the sheet's header row, for context. */
  header?: boolean;
}

/** A Word comment in a section's text: the target of its anchor ("comment3") and who wrote it. */
export interface UnitComment {
  target: string;
  /** Null when the file names no author. */
  author: string | null;
}

/** A span of a Unit's text and what it stands for in the file, e.g. the cell "B12". */
export interface Anchor {
  /** UTF-16 offsets into the Unit's text; `end` is exclusive. */
  start: number;
  end: number;
  /**
   * slide: "title", "shape2", "table1.row3", "chart1", "image1" or "notes";
   * section: "p12" (the 12th paragraph read), or "comment3" (a Word comment,
   * by its id); rows: a cell, "B12".
   */
  target: string;
}

/** A Unit: its number, kind, label, text and anchors. */
export interface TextUnit {
  /**
   * The Unit's number, from 1, in reading order: a PDF's page number, a
   * deck's slide number. Called `page`, as the `document_pages` column is.
   */
  page: number;
  kind: UnitKind;
  label: UnitLabel | null;
  text: string;
  /** Stored for slides and sections; worked out from the text for rows (see `anchorsOf`). */
  anchors: Anchor[] | null;
}

/**
 * A Unit's text as the check reads it from storage: `page` is the Unit's
 * number (null only in data from before Units); `kind` defaults to "page".
 */
export interface UnitText {
  page: number | null;
  text: string;
  kind?: UnitKind;
  label?: UnitLabel | null;
  anchors?: Anchor[] | null;
}

/** Builds a Unit's text while recording anchors. */
export class UnitBuilder {
  text = "";
  anchors: Anchor[] = [];

  /** Appends `content` as one anchored span, after `separator` unless it is the first. */
  add(content: string, target: string | null, separator = "\n"): void {
    if (!content) return;
    if (this.text) this.text += separator;
    const start = this.text.length;
    this.text += content;
    if (target) this.anchors.push({ start, end: this.text.length, target });
  }

  get empty(): boolean {
    return this.text.trim() === "";
  }
}

/** "A", "B", … "Z", "AA", … for a column from 0. */
export function columnName(index: number): string {
  let name = "";
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) {
    name = String.fromCharCode(65 + ((n - 1) % 26)) + name;
  }
  return name;
}

/** "C12" → { column: 2, row: 12 }; null if it isn't a cell reference. */
export function parseCellRef(ref: string): { column: number; row: number } | null {
  const match = /^\$?([A-Za-z]{1,3})\$?(\d{1,7})$/.exec(ref);
  if (!match) return null;
  let column = 0;
  for (const letter of (match[1] as string).toUpperCase()) {
    column = column * 26 + (letter.charCodeAt(0) - 64);
  }
  return { column: column - 1, row: Number(match[2]) };
}

/**
 * The anchors of a block of rows: one per non-empty cell, from its text. Each
 * line is a row (`label.rows` gives its number) and lists its cells from the
 * sheet's first column, separated by tabs.
 */
export function rowAnchors(text: string, label: UnitLabel | null): Anchor[] {
  const rows = label?.rows ?? [];
  const first = label?.column ?? 0;
  const anchors: Anchor[] = [];
  let lineStart = 0;
  let line = 0;
  while (lineStart <= text.length) {
    const lineEnd = text.indexOf("\n", lineStart);
    const end = lineEnd < 0 ? text.length : lineEnd;
    const row = rows[line];
    if (row !== undefined) {
      let cellStart = lineStart;
      let column = first;
      while (cellStart <= end) {
        const tab = text.indexOf("\t", cellStart);
        const cellEnd = tab < 0 || tab > end ? end : tab;
        if (cellEnd > cellStart) {
          anchors.push({ start: cellStart, end: cellEnd, target: `${columnName(column)}${row}` });
        }
        column++;
        cellStart = cellEnd + 1;
      }
    }
    if (lineEnd < 0) break;
    lineStart = lineEnd + 1;
    line++;
  }
  return anchors;
}

/** A Unit's anchors: those stored, or for rows, those its text gives. */
export function anchorsOf(unit: Pick<UnitText, "kind" | "label" | "text" | "anchors">): Anchor[] {
  if (unit.kind === "rows") return rowAnchors(unit.text, unit.label ?? null);
  return unit.anchors ?? [];
}

/** The line of a block of lines that `offset` (into the Unit's text) is on, numbered as the file is. */
export function lineAt(text: string, label: UnitLabel | null, offset: number): number {
  let line = label?.from ?? 1;
  for (let at = text.indexOf("\n"); at >= 0 && at < offset; at = text.indexOf("\n", at + 1)) line++;
  return line;
}

/** Whether two Units can be cited together: the same kind, and for rows, the same sheet. */
export function sameRun(
  a: Pick<UnitText, "kind" | "label">,
  b: Pick<UnitText, "kind" | "label">,
): boolean {
  return (
    (a.kind ?? "page") === (b.kind ?? "page") &&
    (a.kind !== "rows" || a.label?.sheet === b.label?.sheet)
  );
}
