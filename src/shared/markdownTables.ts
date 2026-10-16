/**
 * GitHub's pipe tables, read the same way wherever IncarnaMind reads Markdown:
 * the Document viewer, and the core reading what a chat model writes. A row
 * is split at each "|" that isn't escaped or inside code; the row under the
 * heading row ("| --- | :-: |") gives each column's alignment.
 */

import { linkSources, type NamedSource } from "./namedSources";

/** How a column's text is aligned; null: as the text flows. */
export type TableAlign = "left" | "center" | "right" | null;

/** A cell's place in a row's text: from `start` to `end`, its surrounding spaces left out. */
export interface CellRange {
  start: number;
  end: number;
}

const DELIMITER_CELL = /^:?-+:?$/;

/** The cells of a table row, as ranges of `text`, split at "|" outside code and escapes. */
export function tableCellRanges(text: string): CellRange[] {
  const cells: CellRange[] = [];
  let at = 0;
  let cellStart = 0;
  let inCode = false;
  // A leading pipe opens the first cell.
  const lead = text.search(/\S/);
  if (lead >= 0 && text[lead] === "|") {
    at = lead + 1;
    cellStart = at;
  }
  for (; at < text.length; at++) {
    const char = text[at];
    if (char === "\\") at++;
    else if (char === "`") inCode = !inCode;
    else if (char === "|" && !inCode) {
      cells.push({ start: cellStart, end: at });
      cellStart = at + 1;
    }
  }
  if (text.slice(cellStart).trim() !== "") cells.push({ start: cellStart, end: text.length });
  return cells.map(({ start, end }) => {
    let from = start;
    let to = end;
    while (from < to && /\s/.test(text[from] as string)) from++;
    while (to > from && /\s/.test(text[to - 1] as string)) to--;
    return { start: from, end: to };
  });
}

/**
 * Each column's alignment, when a table starts at `lines[i]`: a row with a
 * pipe, then a delimiter row with as many cells.
 */
export function tableStart(lines: readonly string[], i: number): TableAlign[] | null {
  const head = lines[i];
  const below = lines[i + 1];
  if (head === undefined || below === undefined || !head.includes("|")) return null;
  const align = delimiterRow(below);
  return align && tableCellRanges(head).length === align.length ? align : null;
}

/** A line that reads as a table's row, its delimiter row or HTML table markup: what is left of a table that couldn't be made. */
export const looksLikeTable = (line: string): boolean =>
  /^\s*\|.*\|\s*$/.test(line) ||
  /\|\s*:?-{3,}:?\s*\|/.test(line) ||
  /<\/?(?:table|thead|tbody|tr|td|th)\b[^>]*>/i.test(line);

/**
 * A row's cells' text, as many as its table has columns: a missing one is
 * empty, and the text of extra ones (`extra`) is kept in the last, after " | ".
 */
export function rowCells(line: string, columns: number): { cells: string[]; extra: number } {
  const texts = tableCellRanges(line).map(({ start, end }) => line.slice(start, end));
  const extras = texts.slice(columns).filter((text) => text !== "");
  const cells = Array.from({ length: columns }, (_, column) => texts[column] ?? "");
  if (extras.length > 0) cells[columns - 1] = [cells[columns - 1], ...extras].join(" | ");
  return { cells, extra: extras.length };
}

/**
 * The pipe tables of Markdown, with each row's bracketed sources linked
 * (`linkSources`), and how many of them were left as text and how many
 * extra cells rows have (`rowCells`).
 */
export function linkTableSources(
  markdown: string,
  link: (source: NamedSource) => string | null,
): { markdown: string; unlinked: number; extraCells: number } {
  const lines = markdown.split("\n");
  let unlinked = 0;
  let extraCells = 0;
  for (let at = 0; at < lines.length; at++) {
    const align = tableStart(lines, at);
    if (!align) continue;
    let row = at;
    for (; row < lines.length; row++) {
      const line = lines[row] ?? "";
      if (row === at + 1) continue;
      if (row > at + 1 && (line.trim() === "" || !line.includes("|"))) break;
      extraCells += rowCells(line, align.length).extra;
      const linked = linkSources(line, link);
      lines[row] = linked.text;
      unlinked += linked.unlinked;
    }
    at = row - 1;
  }
  return { markdown: lines.join("\n"), unlinked, extraCells };
}

/** Each column's alignment, when `text` is a table's delimiter row ("|:--|--:|"); otherwise null. */
export function delimiterRow(text: string): TableAlign[] | null {
  if (!text.includes("-") || text.trim() === "") return null;
  const cells = tableCellRanges(text).map(({ start, end }) => text.slice(start, end));
  if (cells.length === 0 || !cells.every((cell) => DELIMITER_CELL.test(cell))) return null;
  return cells.map((cell) =>
    cell.startsWith(":") && cell.endsWith(":")
      ? "center"
      : cell.endsWith(":")
        ? "right"
        : cell.startsWith(":")
          ? "left"
          : null,
  );
}
