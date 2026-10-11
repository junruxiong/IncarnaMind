/**
 * GitHub's pipe tables, read the same way wherever IncarnaMind reads Markdown:
 * the Document viewer, and the core reading what a chat model writes. A row
 * is split at each "|" that isn't escaped or inside code; the row under the
 * heading row ("| --- | :-: |") gives each column's alignment.
 */

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
