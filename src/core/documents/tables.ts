/**
 * The tables on a page (#1368), from where its text sits: a PDF's text layer and a scanned page's
 * recognized lines are read the same way. A page's words are put in rows by their baselines; a
 * table is three rows or more whose words line up in columns, with a column of figures or three
 * columns at least, so two columns of prose side by side are none. The columns are where the rows'
 * words lie, across the rows, not a gap rule. Rows with no figure above the first that has one are
 * the headings, wrapped or in two levels; a lone word under a row goes on its cell, and a lone
 * label ending in a colon names the section the rows under it are in. Pure.
 */

/** Text that sits together on a row: a word, or words set close enough to read as one. */
export interface Fragment {
  text: string;
  left: number;
  right: number;
  /** Its font size, or its line's height: the unit the gaps are measured in. */
  size: number;
}

/** A row of a page: its words left to right, and how far down the page it is, in their sizes' unit. */
export interface Row {
  fragments: Fragment[];
  y: number;
}

/** A table's text, as its columns read it. */
export interface TableGrid {
  /** Each column's heading, its rows of headings joined; empty when the table has none. */
  headings: string[];
  rows: { cells: string[]; section: boolean }[];
  /** Which columns hold figures. */
  numeric: boolean[];
}

/** A table found among a page's rows: the rows it takes, first to last. */
export interface FoundTable {
  first: number;
  last: number;
  grid: TableGrid;
}

/** Wider than a space, narrower than a column's gap: in font sizes. */
const WORD_GAP = 0.4;
/** Two baselines this close, in font sizes, are one row. */
const BASELINE = 0.5;
const MIN_ROWS = 3;
const MAX_HEADING = 80;
/** Most of a column's cells are this wide at most, in Latin letters, a CJK character two: prose runs wider. */
const MAX_CELL = 56;

const widthOf = (text: string) =>
  [...text].reduce((width, character) => width + (CJK_EDGE.test(character) ? 2 : 1), 0);
/** A table's headings take this many rows at most: rows above them aren't its. */
const MAX_HEADING_ROWS = 3;

const FIGURE =
  /^[<>≈~≤≥]?\s?[-+−–(（]?[$€£¥]?\s?\d[\d,.\s]*(?:%|\s?(?:[kmb]n?|bn|pp|x|×|万|亿|元))?[)）]?$/iu;
const CJK_EDGE = /[\u2e80-\u2fdf\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]/u;

export const isFigure = (text: string) => FIGURE.test(text.trim());
const YEAR = /^(?:FY\s?)?(?:19|20)\d{2}$/i;

const join = (a: string, b: string) =>
  !a
    ? b
    : !b
      ? a
      : CJK_EDGE.test(a.at(-1) ?? "") && CJK_EDGE.test(b[0] ?? "")
        ? a + b
        : `${a} ${b}`;

/** A pdf.js text item's place: where it starts, how wide it is, its size, and whether it reads level. */
export function textItemOf(item: object) {
  if (!("str" in item) || typeof item.str !== "string") return null;
  const transform = "transform" in item && Array.isArray(item.transform) ? item.transform : [];
  const [a = 0, b = 0, c = 0, d = 0, e = 0, f = 0] = transform.map((value: unknown) =>
    typeof value === "number" ? value : 0,
  );
  const width = "width" in item && typeof item.width === "number" ? item.width : 0;
  const height = "height" in item && typeof item.height === "number" ? item.height : 0;
  return {
    str: item.str,
    x: e,
    baseline: f,
    width,
    size: Math.max(Math.abs(d), height),
    fontSize: Math.abs(d),
    level: Math.abs(b) < 0.01 && Math.abs(c) < 0.01 && a > 0,
    eol: "hasEOL" in item && item.hasEOL === true,
  };
}

/**
 * A PDF page's rows, top to bottom, from its text layer: each with its words, and every item on
 * it, spaces and line ends too, for the text around a table. pdf.js gives items in the order they
 * were drawn, which isn't always a row's; text set at an angle is no row's.
 */
export function textLayerRows(items: readonly object[]): (Row & { items: object[] })[] {
  const rows: {
    baseline: number;
    size: number;
    placed: { x: number; right: number; str: string; size: number }[];
    items: object[];
  }[] = [];
  const rowOf = (baseline: number, size: number) =>
    rows.find((row) => Math.abs(row.baseline - baseline) <= Math.max(row.size, size) * BASELINE);
  let last: (typeof rows)[number] | undefined;
  for (const item of items) {
    const placed = textItemOf(item);
    if (!placed?.level || placed.str.trim() === "") {
      // A space or a line end stays with the row of the text before it.
      last?.items.push(item);
      continue;
    }
    let row = rowOf(placed.baseline, placed.size);
    if (!row) {
      row = { baseline: placed.baseline, size: placed.size, placed: [], items: [] };
      rows.push(row);
    }
    row.size = Math.max(row.size, placed.size);
    row.placed.push({
      x: placed.x,
      right: placed.x + placed.width,
      str: placed.str,
      size: placed.size,
    });
    row.items.push(item);
    last = row;
  }
  return rows
    .sort((a, b) => b.baseline - a.baseline)
    .map((row) => ({ fragments: fragmentsOf(row.placed), y: -row.baseline, items: row.items }));
}

function fragmentsOf(placed: readonly { x: number; right: number; str: string; size: number }[]) {
  const fragments: Fragment[] = [];
  for (const each of [...placed].sort((a, b) => a.x - b.x)) {
    const last = fragments.at(-1);
    if (last && each.x - last.right <= each.size * WORD_GAP) {
      last.text =
        /\s$/.test(last.text) || /^\s/.test(each.str) || each.x - last.right < each.size * 0.15
          ? last.text + each.str
          : join(last.text, each.str);
      last.right = Math.max(last.right, each.right);
      last.size = Math.max(last.size, each.size);
    } else {
      fragments.push({ text: each.str, left: each.x, right: each.right, size: each.size });
    }
  }
  return fragments.map((fragment) => ({
    ...fragment,
    text: fragment.text.replace(/\s+/g, " ").trim(),
  }));
}

/**
 * A scanned page's rows, from its Unit: its text a row a line, each recognized line a fragment
 * with its box (`Anchor.quad`, in fractions of the page), its height the unit of its gaps.
 */
export function scannedRows(
  text: string,
  anchors: readonly { start: number; end: number; quad?: number[] }[],
): Row[] {
  const rows: Row[] = [];
  let at = 0;
  for (const line of text.split("\n")) {
    const end = at + line.length;
    const fragments = anchors
      .filter((anchor) => anchor.quad && anchor.start >= at && anchor.end <= end)
      .map((anchor) => {
        const quad = anchor.quad ?? [];
        const xs = quad.filter((_, index) => index % 2 === 0);
        const ys = quad.filter((_, index) => index % 2 === 1);
        return {
          text: text.slice(anchor.start, anchor.end).trim(),
          left: Math.min(...xs),
          right: Math.max(...xs),
          size: Math.max(...ys) - Math.min(...ys),
        };
      })
      .filter((fragment) => fragment.text !== "")
      .sort((a, b) => a.left - b.left);
    const tops = anchors
      .filter((anchor) => anchor.quad && anchor.start >= at && anchor.end <= end)
      .flatMap((anchor) => (anchor.quad ?? []).filter((_, index) => index % 2 === 1));
    rows.push({ fragments, y: tops.length > 0 ? Math.min(...tops) : (rows.at(-1)?.y ?? 0) });
    at = end + 1;
  }
  return rows;
}

interface Span {
  left: number;
  right: number;
}

function spansOf(rows: readonly (readonly Fragment[])[]): Span[] {
  const all = rows.flat().sort((a, b) => a.left - b.left);
  const spans: Span[] = [];
  for (const each of all) {
    const last = spans.at(-1);
    if (last && each.left <= last.right) last.right = Math.max(last.right, each.right);
    else spans.push({ left: each.left, right: each.right });
  }
  return spans;
}

const overlapping = (fragment: Fragment, spans: readonly Span[]) =>
  spans.flatMap((span, index) =>
    fragment.left < span.right && span.left < fragment.right ? [index] : [],
  );

// Headings set left in a wide column lie over no figure: they name the column to their right.
function headedColumn(fragment: Fragment, spans: readonly Span[]): number {
  const over = overlapping(fragment, spans);
  if (over.length > 0) return over[0] as number;
  const right = spans.findIndex((span) => span.left >= fragment.right);
  return right >= 0 ? right : spans.length - 1;
}

// A heading centred over two columns lies between them: it starts over the left one.
function startColumn(fragment: Fragment, spans: readonly Span[]): number {
  const over = overlapping(fragment, spans);
  if (over.length > 0) return over[0] as number;
  let best = 0;
  let distance = Number.POSITIVE_INFINITY;
  spans.forEach((span, index) => {
    const away = Math.max(span.left - fragment.right, fragment.left - span.right);
    if (away < distance) {
      best = index;
      distance = away;
    }
  });
  return best;
}

function cellsOf(row: readonly Fragment[], spans: readonly Span[]): string[] | null {
  const cells = spans.map(() => "");
  for (const fragment of row) {
    const over = overlapping(fragment, spans);
    if (over.length !== 1) return null;
    const column = over[0] as number;
    cells[column] = join(cells[column] ?? "", fragment.text);
  }
  return cells;
}

function gridOf(
  rows: readonly (readonly Fragment[])[],
): { grid: TableGrid; dropped: number } | null {
  const hasFigure = (row: readonly Fragment[]) => row.some((fragment) => isFigure(fragment.text));
  // A year names a column more often than it fills one: "2025" over a column of figures.
  const hasData = (row: readonly Fragment[]) =>
    row.some((fragment) => isFigure(fragment.text) && !YEAR.test(fragment.text.trim()));
  const firstData = rows.findIndex(hasData);
  const firstBody = firstData >= 0 ? firstData : rows.findIndex(hasFigure);
  const isSection = (row: readonly Fragment[]) =>
    row.length === 1 && /[:：]$/.test(row[0]?.text ?? "");
  const firstSection = rows.findIndex(isSection);
  // Words only: the first row names the columns. A section's name starts the body.
  const headingCount =
    firstBody < 0 ? 1 : firstSection >= 0 && firstSection < firstBody ? firstSection : firstBody;
  const bodyRows = rows.slice(headingCount);
  const spans = spansOf(bodyRows.filter((row) => row.length >= 2));
  if (spans.length < 2) return null;
  // Over the row of headings next to the body, a row heads the columns only if it fills them, or
  // groups the ones right of the labels (2025 over Q1 and Q2): a form's fields above aren't headings.
  let dropped = headingCount;
  while (dropped > 0) {
    const row = rows[dropped - 1] ?? [];
    const adjacent = dropped === headingCount;
    const grouping = row.every((fragment) => fragment.left > (spans[0]?.right ?? 0));
    if (!adjacent && row.length < spans.length && !grouping) break;
    dropped--;
  }
  const headingRows = rows.slice(dropped, headingCount);

  const body: TableGrid["rows"] = [];
  for (const row of bodyRows) {
    const cells = cellsOf(row, spans);
    if (!cells) return null;
    const filled = cells.filter((cell) => cell !== "").length;
    const previous = body.at(-1);
    if (filled === 1 && previous) {
      const column = cells.findIndex((cell) => cell !== "");
      const text = cells[column] ?? "";
      if (column === 0 && /[:：]$/.test(text)) {
        body.push({ cells, section: true });
      } else {
        // A label or a cell that wrapped onto a line of its own.
        previous.cells[column] = join(previous.cells[column] ?? "", text);
      }
      continue;
    }
    body.push({ cells, section: filled === 1 && cells[0] !== "" && /[:：]$/.test(cells[0] ?? "") });
  }
  const data = body.filter((row) => !row.section);
  if (data.length < 2 || headingRows.length + data.length < MIN_ROWS) return null;

  const numeric = spans.map((_, column) => {
    const filled = data.map((row) => row.cells[column] ?? "").filter((text) => text !== "");
    const figures = filled.filter(isFigure).length;
    return figures >= 2 && figures >= filled.length / 2 && filled.length >= data.length / 2;
  });
  if (spans.length < 3 && !numeric.slice(1).some(Boolean)) return null;
  // Words only, most rows have every column: a page of references in columns has gaps.
  const full = data.filter((row) => row.cells.every((text) => text !== "")).length;
  if (!numeric.some(Boolean) && full < data.length * 0.7) return null;

  const headings = spans.map(() => "");
  headingRows.forEach((row, index) => {
    const adjacent = index === headingRows.length - 1;
    if (adjacent || row.length >= spans.length) {
      for (const fragment of row) {
        const column = headedColumn(fragment, spans);
        headings[column] = join(headings[column] ?? "", fragment.text);
      }
      return;
    }
    // An upper heading names the columns from the one it starts over to the next one's.
    const starts = row.map((fragment) => startColumn(fragment, spans));
    row.forEach((fragment, at) => {
      const from = starts[at] ?? 0;
      const to = starts[at + 1] ?? spans.length;
      for (let column = from; column < Math.max(to, from + 1); column++) {
        headings[column] = join(headings[column] ?? "", fragment.text);
      }
    });
  });
  // A table names every column it heads but its labels', and labels its rows: references set in columns do neither.
  const headed = headings.some((heading) => heading !== "");
  if (headed && headings.slice(1).some((heading) => heading === "")) return null;
  if (data.filter((row) => row.cells[0] === "").length > data.length * 0.3) return null;
  // A table's headings are names and its cells short: columns of prose or references are no table.
  if (headings.some((heading) => heading.length > MAX_HEADING)) return null;
  const widths = spans.map((_, column) => {
    const sorted = data
      .map((row) => widthOf(row.cells[column] ?? ""))
      .filter((width) => width > 0)
      .sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length * 0.75)];
  });
  if (widths.some((width) => (width ?? 0) > MAX_CELL)) return null;
  return { grid: { headings, rows: body, numeric }, dropped };
}

const median = (values: readonly number[]) =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function fits(row: Row, rows: readonly Row[]): boolean {
  const previous = rows.at(-1);
  if (!previous) return true;
  const size = Math.max(...row.fragments.map((each) => each.size), 0);
  const gap = row.y - previous.y;
  const pitch = median(rows.slice(1).map((each, index) => each.y - (rows[index]?.y ?? each.y)));
  // A gap wider than the rows' own is the table's end: a caption or a heading follows.
  if (pitch !== undefined && gap > pitch * 1.8 + size * 0.2) return false;
  const before = spansOf(rows.map((each) => each.fragments));
  const [lone] = row.fragments;
  if (row.fragments.length === 1 && lone) {
    if (overlapping(lone, before).length !== 1) return false;
    // A section's name, or a cell that wrapped onto a line close under its row.
    return /[:：]$/.test(lone.text) || gap <= size * 1.5;
  }
  // A row that joins two columns into one is no row of theirs.
  return spansOf([...rows.map((each) => each.fragments), row.fragments]).length >= before.length;
}

/** The tables among a page's rows, top to bottom, each with the rows it takes. */
export function findTables(rows: readonly Row[]): FoundTable[] {
  const found: FoundTable[] = [];
  let at = 0;
  while (at < rows.length) {
    if ((rows[at]?.fragments.length ?? 0) < 2) {
      at++;
      continue;
    }
    let end = at + 1;
    while (end < rows.length && fits(rows[end] as Row, rows.slice(at, end))) end++;
    const firstBody = rows
      .slice(at, end)
      .findIndex((row) =>
        row.fragments.some((each) => isFigure(each.text) && !YEAR.test(each.text.trim())),
      );
    const first = at + Math.max(0, firstBody - MAX_HEADING_ROWS);
    const made =
      end - first >= MIN_ROWS ? gridOf(rows.slice(first, end).map((row) => row.fragments)) : null;
    if (made) {
      found.push({ first: first + made.dropped, last: end - 1, grid: made.grid });
      at = end;
    } else {
      at++;
    }
  }
  return found;
}
