/**
 * The tables on a PDF page (#1368), from where pdf.js places its text: lines whose words fall
 * into the same columns, three lines or more, with a column of figures or three columns at least.
 * Two columns of prose, side by side, are no table.
 */

/** A page's text in reading order: the items between its tables, and each table's cells. */
export type PagePart =
  | { kind: "text"; items: object[] }
  | { kind: "table"; rows: string[][]; numeric: boolean[] };

interface Placed {
  item: object;
  str: string;
  x: number;
  y: number;
  width: number;
  size: number;
  upright: boolean;
  eol: boolean;
}

interface Cell {
  text: string;
  left: number;
  right: number;
}

interface Line {
  items: object[];
  cells: Cell[];
}

const FIGURE = /^[-+−–(（]?[$€£¥]?\s?\d[\d,.\s]*%?[)）]?$/u;

/** Table rows at least, a header among them. */
const MIN_ROWS = 3;

export const isFigure = (text: string) => FIGURE.test(text.trim());

function placed(item: object): Placed | null {
  if (!("str" in item) || typeof item.str !== "string") return null;
  const transform =
    "transform" in item && Array.isArray(item.transform) ? (item.transform as unknown[]) : [];
  const [a = 0, b = 0, c = 0, d = 0, e = 0, f = 0] = transform.map((value) =>
    typeof value === "number" ? value : 0,
  );
  const width = "width" in item && typeof item.width === "number" ? item.width : 0;
  const height = "height" in item && typeof item.height === "number" ? item.height : 0;
  return {
    item,
    str: item.str,
    x: e,
    y: f,
    width,
    size: Math.max(Math.abs(d), height),
    upright: Math.abs(b) < 0.01 && Math.abs(c) < 0.01 && a > 0,
    eol: "hasEOL" in item && item.hasEOL === true,
  };
}

/** A line's words as cells: a gap wider than one and a half of its font's size parts two. */
function cellsOf(line: readonly Placed[]): Cell[] {
  const cells: Cell[] = [];
  let space = false;
  for (const each of line) {
    if (!each.upright) continue;
    if (each.str.trim() === "") {
      space = each.str !== "";
      continue;
    }
    const last = cells.at(-1);
    if (last && each.x - last.right <= each.size * 1.5) {
      last.text += (space ? " " : "") + each.str;
      last.right = Math.max(last.right, each.x + each.width);
    } else {
      cells.push({ text: each.str, left: each.x, right: each.x + each.width });
    }
    space = /\s$/.test(each.str);
  }
  return cells.map((cell) => ({ ...cell, text: cell.text.replace(/\s+/g, " ").trim() }));
}

function linesOf(items: readonly object[]): Line[] {
  const lines: Line[] = [];
  let current: Placed[] = [];
  let all: object[] = [];
  const end = () => {
    if (all.length > 0) lines.push({ items: all, cells: cellsOf(current) });
    current = [];
    all = [];
  };
  for (const item of items) {
    const each = placed(item);
    if (!each) {
      all.push(item);
      continue;
    }
    const baseline = current.find((one) => one.str.trim() !== "");
    if (baseline && each.str.trim() !== "" && Math.abs(each.y - baseline.y) > each.size / 2) end();
    current.push(each);
    all.push(item);
    if (each.eol) end();
  }
  end();
  return lines;
}

/** Which column each cell is in, left to right, each cell over its own column; null when one isn't. */
function columnsOf(cells: readonly Cell[], columns: readonly Cell[]): number[] | null {
  const found: number[] = [];
  let next = 0;
  for (const cell of cells) {
    let column = next;
    while (
      column < columns.length &&
      !((columns[column]?.left ?? 0) < cell.right && cell.left < (columns[column]?.right ?? 0))
    ) {
      column++;
    }
    if (column >= columns.length) return null;
    found.push(column);
    next = column + 1;
  }
  return found;
}

interface Run {
  start: number;
  columns: Cell[];
  rows: string[][];
}

function asTable(run: Run): PagePart | null {
  if (run.rows.length < MIN_ROWS) return null;
  const width = run.columns.length;
  const numeric = run.columns.map((_, column) => {
    const filled = run.rows.map((row) => row[column] ?? "").filter((text) => text !== "");
    const figures = filled.filter(isFigure).length;
    return figures > 0 && figures >= filled.length / 2;
  });
  if (width < 3 && !numeric.slice(1).some(Boolean)) return null;
  return { kind: "table", rows: run.rows, numeric };
}

/** The page's parts, its tables among them; null when it has none. */
export function tablesIn(items: readonly object[]): PagePart[] | null {
  const lines = linesOf(items);
  const tables: { start: number; end: number; table: PagePart }[] = [];
  let run: Run | null = null;
  const close = (end: number) => {
    const table = run && asTable(run);
    if (run && table) tables.push({ start: run.start, end, table });
    run = null;
  };
  /** Puts a line's cells in the run's columns, widening them; false when the line doesn't fit. */
  const add = (into: Run, cells: readonly Cell[], first = false) => {
    const fitted = columnsOf(cells, into.columns);
    if (!fitted) return false;
    const row = into.columns.map(() => "");
    fitted.forEach((column, at) => {
      const cell = cells[at];
      const span = into.columns[column];
      if (!cell || !span) return;
      row[column] = cell.text;
      span.left = Math.min(span.left, cell.left);
      span.right = Math.max(span.right, cell.right);
    });
    if (first) into.rows.unshift(row);
    else into.rows.push(row);
    return true;
  };
  lines.forEach((line, index) => {
    const { cells } = line;
    if (cells.length < 2) {
      close(index);
      return;
    }
    if (run && add(run, cells)) return;
    const next: Run = {
      start: index,
      columns: cells.map(({ left, right }) => ({ text: "", left, right })),
      rows: [cells.map((cell) => cell.text)],
    };
    // A heading row with fewer cells ("2025  2024" over each row's label and figures) heads the run that starts under it.
    const heading = run?.rows.length === 1 ? lines[index - 1] : undefined;
    if (heading && add(next, heading.cells, true)) {
      next.start = index - 1;
      run = null;
    } else {
      close(index);
    }
    run = next;
  });
  close(lines.length);
  if (tables.length === 0) return null;

  const parts: PagePart[] = [];
  let at = 0;
  for (const { start, end, table } of tables) {
    const text = lines.slice(at, start).flatMap((line) => line.items);
    if (text.length > 0) parts.push({ kind: "text", items: text });
    parts.push(table);
    at = end;
  }
  const rest = lines.slice(at).flatMap((line) => line.items);
  if (rest.length > 0) parts.push({ kind: "text", items: rest });
  return parts;
}
