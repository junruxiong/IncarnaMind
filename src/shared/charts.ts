/**
 * Charts in a Mind (#436): what a chart Block holds, and how one is made from
 * a table or from cited figures. A chart keeps the values it draws, each with
 * where it came from (its table cell, or its line of text) and the Citation
 * that was there, so every point can be traced to its source after the table
 * changes or goes. Charts from calculated figures (#439, #440) and from an
 * Analysis (#449) add their own kinds of origin beside these.
 *
 * Pure, with no Node or DOM imports: the editor, the core's one reader
 * (exports, Question context) and the tests share it.
 */
import type { CitationAttributes, CitationCheck } from "../core/api";
import { type Figure, type FigureUnit, figureContextOf, readFigures } from "./figures";
import type { MessageKey, MessageParams } from "./i18n";

export const CHART_KINDS = ["bar", "line", "pie", "scatter"] as const;
export type ChartKind = (typeof CHART_KINDS)[number];

/** At most this many values in a chart: a longer table is cut, and the chart says so. */
export const MAX_CHART_VALUES = 500;

export type ChartValueOrigin =
  /** A cell of a table in the Mind: its row and column on the table's grid, from 0, the heading row included. */
  | { kind: "cell"; row: number; column: number }
  /** A cited figure in a line of text: the line's place among those the chart was made from, from 0. */
  | { kind: "text"; line: number };

export interface ChartValue {
  /** As written where it came from: "1.5亿", "$377.9 million", "41.2%". */
  shown: string;
  /** The number read from `shown` (./figures); null when it holds none, or none that reads one way: not drawn. */
  value: number | null;
  unit: FigureUnit | null;
  /** The Citation that was with it, as its node held it; null: it had none. */
  citation: CitationAttributes | null;
  from: ChartValueOrigin;
}

export interface ChartRow {
  /** What the row is: the first column's text ("Q3"), or a line's words before its figure. */
  label: string;
  /** One per series, in order; null where the row has nothing in that column. */
  values: (ChartValue | null)[];
}

export interface ChartData {
  /** The first column's heading ("Quarter"); "" when the source has none. */
  labelHeading: string;
  /** Each column that holds a number, by its heading ("2025"); "" when it has none. */
  series: string[];
  rows: ChartRow[];
  /** How many rows the source had when only the first ones fit (`MAX_CHART_VALUES`); null: all are here. */
  cutFrom: number | null;
}

export type ChartSource = { kind: "table"; blockId: string | null } | { kind: "figures" };

export interface ChartAttributes {
  kind: ChartKind;
  title: string;
  /** The series drawn, by their place in `data.series`; null: all of them. */
  drawn: number[] | null;
  data: ChartData;
  source: ChartSource;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const text = (value: unknown): string => (typeof value === "string" ? value : "");

const isKind = (value: unknown): value is ChartKind =>
  typeof value === "string" && (CHART_KINDS as readonly string[]).includes(value);

function readOrigin(value: unknown): ChartValueOrigin | null {
  if (!isRecord(value)) return null;
  const index = (n: unknown) => (typeof n === "number" && Number.isInteger(n) && n >= 0 ? n : null);
  if (value.kind === "cell") {
    const row = index(value.row);
    const column = index(value.column);
    return row === null || column === null ? null : { kind: "cell", row, column };
  }
  if (value.kind === "text") {
    const line = index(value.line);
    return line === null ? null : { kind: "text", line };
  }
  return null;
}

function readValue(value: unknown): ChartValue | null {
  if (!isRecord(value)) return null;
  const from = readOrigin(value.from);
  if (!from) return null;
  const number =
    typeof value.value === "number" && Number.isFinite(value.value) ? value.value : null;
  return {
    shown: text(value.shown),
    value: number,
    unit:
      isRecord(value.unit) && typeof value.unit.kind === "string"
        ? (value.unit as FigureUnit)
        : null,
    citation: isRecord(value.citation) ? (value.citation as unknown as CitationAttributes) : null,
    from,
  };
}

/** Anything malformed (from another version, or a sync) is left out rather than guessed. */
export function readChartData(value: unknown): ChartData {
  if (!isRecord(value)) return { labelHeading: "", series: [], rows: [], cutFrom: null };
  const series = Array.isArray(value.series) ? value.series.map(text) : [];
  const rows = (Array.isArray(value.rows) ? value.rows : []).flatMap((row): ChartRow[] => {
    if (!isRecord(row)) return [];
    const values = Array.isArray(row.values) ? row.values : [];
    return [{ label: text(row.label), values: series.map((_, index) => readValue(values[index])) }];
  });
  const cutFrom =
    typeof value.cutFrom === "number" &&
    Number.isInteger(value.cutFrom) &&
    value.cutFrom > rows.length
      ? value.cutFrom
      : null;
  return { labelHeading: text(value.labelHeading), series, rows, cutFrom };
}

function readSource(value: unknown): ChartSource {
  if (isRecord(value) && value.kind === "figures") return { kind: "figures" };
  const blockId = isRecord(value) && typeof value.blockId === "string" ? value.blockId : null;
  return { kind: "table", blockId };
}

export function readChart(attributes: Readonly<Record<string, unknown>>): ChartAttributes {
  const data = readChartData(attributes.data);
  const drawn = Array.isArray(attributes.drawn)
    ? [
        ...new Set(
          attributes.drawn.filter(
            (index): index is number =>
              typeof index === "number" &&
              Number.isInteger(index) &&
              index >= 0 &&
              index < data.series.length,
          ),
        ),
      ]
    : null;
  return {
    kind: isKind(attributes.kind) ? attributes.kind : "bar",
    title: text(attributes.title),
    drawn,
    data,
    source: readSource(attributes.source),
  };
}

/**
 * The series drawn, by their place in the data: those chosen (all, by default),
 * in order. A pie draws one, the last chosen ("this year"); a scatter chart
 * the first two, one across and one up.
 */
export function drawnSeries(chart: Pick<ChartAttributes, "kind" | "drawn" | "data">): number[] {
  const chosen = chart.drawn ?? chart.data.series.map((_, index) => index);
  if (chart.kind === "pie") return chosen.slice(-1);
  if (chart.kind === "scatter") return chosen.slice(0, 2);
  return chosen;
}

/**
 * A series' name: its heading, or, with none, "Column 3" (the column of the
 * table it came from) or "Value" for figures from text, in the words of `say`.
 */
export function seriesName(
  chart: Pick<ChartAttributes, "data" | "source">,
  index: number,
  say: (key: MessageKey, params?: MessageParams) => string,
): string {
  const name = chart.data.series[index];
  if (name) return name;
  if (chart.source.kind === "figures") return say("chart.series.value");
  const from = chart.data.rows
    .map((row) => row.values[index]?.from)
    .find((each) => each?.kind === "cell");
  return say("chart.series.unnamed", {
    number: from?.kind === "cell" ? from.column + 1 : index + 2,
  });
}

/** Whether a kind can be drawn from this data: a scatter chart needs two series. */
export const kindFits = (kind: ChartKind, data: ChartData) =>
  kind !== "scatter" || data.series.length >= 2;

export interface ChartPoint {
  row: number;
  series: number;
  value: ChartValue & { value: number };
}

/**
 * The points drawn, series by series, each in row order; a scatter chart's
 * are its rows with a number in both series, as the one series up.
 */
export function chartPoints(chart: Pick<ChartAttributes, "kind" | "drawn" | "data">): ChartPoint[] {
  const drawn = drawnSeries(chart);
  const { rows } = chart.data;
  const points: ChartPoint[] = [];
  if (chart.kind === "scatter") {
    const [across, up] = drawn;
    if (across === undefined || up === undefined) return [];
    rows.forEach((row, index) => {
      const x = row.values[across];
      const y = row.values[up];
      if (x?.value != null && y?.value != null) {
        points.push({ row: index, series: up, value: y as ChartPoint["value"] });
      }
    });
    return points;
  }
  for (const series of drawn) {
    rows.forEach((row, index) => {
      const value = row.values[series];
      // A pie has no room for a share below nothing.
      if (value?.value == null || (chart.kind === "pie" && value.value <= 0)) return;
      points.push({ row: index, series, value: value as ChartPoint["value"] });
    });
  }
  return points;
}

export interface ChartSourcePlace {
  /**
   * The first of its Citations whose quote wasn't found, if any, else the
   * first: what an export's footnote for the place says.
   */
  citation: CitationAttributes;
  count: number;
}

export interface ChartSources {
  values: number;
  uncited: number;
  /** Cells of the series drawn that hold words but no number, so aren't drawn. */
  notDrawn: number;
  places: ChartSourcePlace[];
  documents: number;
}

function placeKey(citation: CitationAttributes): string {
  const document = citation.documentId ?? citation.documentName ?? "";
  const where = citation.location
    ? JSON.stringify(citation.location)
    : `${citation.pageFrom ?? ""}-${citation.pageTo ?? ""}`;
  return `${document}\u0000${where}`;
}

const isFound = (check: CitationCheck | undefined) => check === "found";

export function chartSources(
  chart: Pick<ChartAttributes, "kind" | "drawn" | "data">,
): ChartSources {
  const points = chartPoints(chart);
  const places = new Map<string, ChartSourcePlace>();
  const documents = new Set<string>();
  let uncited = 0;
  for (const { value } of points) {
    const { citation } = value;
    if (!citation) {
      uncited++;
      continue;
    }
    documents.add(citation.documentId ?? citation.documentName ?? "");
    const key = placeKey(citation);
    const place = places.get(key);
    if (!place) places.set(key, { citation, count: 1 });
    else {
      place.count++;
      if (isFound(place.citation.check) && !isFound(citation.check)) place.citation = citation;
    }
  }
  let notDrawn = 0;
  const drawn = drawnSeries(chart);
  for (const row of chart.data.rows) {
    for (const series of drawn) {
      const value = row.values[series];
      if (value && value.value === null && value.shown.trim() !== "") notDrawn++;
    }
  }
  return {
    values: points.length,
    uncited,
    notDrawn,
    places: [...places.values()],
    documents: documents.size,
  };
}

/** What a figure doesn't settle that changes its value: drawn, it would be a guess. */
const UNSETTLED = new Set(["separator", "sign"]);

/** The first figure in a text, with the value to draw, or null when it can't be read one way. */
function figureIn(
  shown: string,
  context: ReturnType<typeof figureContextOf>,
): { figure: Figure; value: number | null } | null {
  const figure = readFigures(shown, context)[0];
  if (!figure) return null;
  const unsettled = figure.ambiguous.some((each) => UNSETTLED.has(each.kind));
  return { figure, value: unsettled || !Number.isFinite(figure.value) ? null : figure.value };
}

export interface SourceCell {
  /** Its text, without its Citations' markers or its footnotes' text. */
  text: string;
  citation: CitationAttributes | null;
  header: boolean;
  /** It spans down or across from another slot: its value is that slot's, its words read again. */
  continued?: boolean;
}

/** A table laid out on its grid, row by row: a cell in each slot, or null where none is. */
export type SourceGrid = readonly (readonly (SourceCell | null)[])[];

export interface FromTableOptions {
  /** The table Block's id, so the chart can tell when it changes. */
  blockId: string | null;
  title: string;
  /** The Mind's language, a BCP 47 tag: how its numbers are written when its text doesn't say. */
  language: string;
}

/**
 * A bar chart of a table (#436): the first column is the labels, and each
 * other column that holds a number is a series. The heading row names them:
 * the first row when it is one, or when its other cells hold no number. Each
 * value keeps its cell and the Citation in it. Only the first rows that fit
 * `MAX_CHART_VALUES` are kept, and the chart says how many there were. Null
 * when no column holds a number.
 */
export function chartFromTable(
  grid: SourceGrid,
  options: FromTableOptions,
): ChartAttributes | null {
  const first = grid[0] ?? [];
  const columns = Math.max(0, ...grid.map((row) => row.length));
  const context = figureContextOf(
    grid.flatMap((row) => row.map((cell) => cell?.text ?? "")).join("\n"),
    options.language,
  );
  const headingRow =
    first.length > 0 &&
    (first.every((cell) => cell === null || cell.header) ||
      first.slice(1).every((cell) => !cell || figureIn(cell.text, context) === null));
  const dataRows = headingRow ? grid.slice(1) : grid;
  const firstRow = headingRow ? 1 : 0;

  const valueAt = (cell: SourceCell | null, row: number, column: number): ChartValue | null => {
    if (!cell || cell.continued || cell.text.trim() === "") return null;
    const read = figureIn(cell.text, context);
    return {
      shown: cell.text.trim(),
      value: read?.value ?? null,
      unit: read ? read.figure.unit : null,
      citation: cell.citation,
      from: { kind: "cell", row, column },
    };
  };

  const seriesColumns: number[] = [];
  for (let column = 1; column < columns; column++) {
    const holdsNumber = dataRows.some(
      (row, index) => valueAt(row[column] ?? null, firstRow + index, column)?.value != null,
    );
    if (holdsNumber) seriesColumns.push(column);
  }
  if (seriesColumns.length === 0) return null;

  const fit = Math.max(1, Math.floor(MAX_CHART_VALUES / seriesColumns.length));
  const kept = dataRows.slice(0, fit);
  return {
    kind: "bar",
    title: options.title.trim(),
    drawn: null,
    data: {
      labelHeading: headingRow ? (first[0]?.text.trim() ?? "") : "",
      series: seriesColumns.map((column) => (headingRow ? (first[column]?.text.trim() ?? "") : "")),
      rows: kept.map((row, index) => ({
        label: row[0]?.text.trim() ?? "",
        values: seriesColumns.map((column) =>
          valueAt(row[column] ?? null, firstRow + index, column),
        ),
      })),
      cutFrom: dataRows.length > kept.length ? dataRows.length : null,
    },
    source: { kind: "table", blockId: options.blockId },
  };
}

/** A line of text (a paragraph, a list item, a table row's text), as a chart reads it. */
export interface SourceLine {
  text: string;
  citation: CitationAttributes | null;
}

/** What separates a label from the figure after it: "Q1: 289.4", "Revenue – $1.2bn". */
const LABEL_END = /[\s:：,，;；=\-–—]+$/u;
const LABEL_START = /^[\s:：,，;；=\-–—•·*]+/u;

/**
 * A bar chart of the figures in lines of text, one row each: the first
 * figure in the line, labelled with the words before it (or after it, when
 * it comes first), with the line's Citation. Lines with no figure are left
 * out. Null when none has one.
 */
export function chartFromFigures(
  lines: readonly SourceLine[],
  options: Omit<FromTableOptions, "blockId">,
): ChartAttributes | null {
  const context = figureContextOf(lines.map((line) => line.text).join("\n"), options.language);
  const rows: ChartRow[] = [];
  lines.forEach((line, index) => {
    const read = figureIn(line.text, context);
    if (!read) return;
    const { figure } = read;
    const before = line.text.slice(0, figure.start).replace(LABEL_END, "").replace(LABEL_START, "");
    const after = line.text.slice(figure.end).replace(LABEL_START, "").replace(LABEL_END, "");
    rows.push({
      label: (before || after).trim(),
      values: [
        {
          shown: figure.chars,
          value: read.value,
          unit: figure.unit,
          citation: line.citation,
          from: { kind: "text", line: index },
        },
      ],
    });
  });
  if (rows.length === 0) return null;
  const kept = rows.slice(0, MAX_CHART_VALUES);
  return {
    kind: "bar",
    title: options.title.trim(),
    drawn: null,
    data: {
      labelHeading: "",
      series: [""],
      rows: kept,
      cutFrom: rows.length > kept.length ? rows.length : null,
    },
    source: { kind: "figures" },
  };
}

/** Compares what is drawn and where it came from, not how it was stored. */
export function sameChartData(a: ChartData, b: ChartData): boolean {
  return JSON.stringify(readChartData(a)) === JSON.stringify(readChartData(b));
}
