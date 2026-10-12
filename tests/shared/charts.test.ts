import { describe, expect, test } from "vitest";
import type { CitationAttributes } from "../../src/core";
import {
  type ChartAttributes,
  chartCells,
  chartFromFigures,
  chartFromTable,
  chartPoints,
  chartSources,
  drawnSeries,
  MAX_CHART_VALUES,
  readChart,
  type SourceCell,
  type SourceGrid,
  unitlessRows,
} from "../../src/shared/charts";

const cited = (page: number, quote = `Revenue on page ${page}`): CitationAttributes => ({
  passageId: "passage",
  documentId: "acme",
  documentName: "Acme Annual Report 2025",
  contentHash: "hash",
  pageFrom: page,
  pageTo: page,
  location: { kind: "page", from: page, to: page },
  quote,
  check: "found",
  checkReason: null,
});

const head = (text: string): SourceCell => ({ text, citation: null, header: true });
const cell = (text: string, citation: CitationAttributes | null = null): SourceCell => ({
  text,
  citation,
  header: false,
});

/** Revenue by quarter, two years, each figure cited but one. */
const QUARTERS: SourceGrid = [
  [head("Quarter"), head("2024"), head("2025")],
  [cell("Q1"), cell("289.4", cited(22)), cell("341.5", cited(22))],
  [cell("Q2"), cell("301.2", cited(22)), cell("362.0", cited(22))],
  [cell("Q3"), cell("309.8", cited(23)), cell("377.9", cited(23))],
  [cell("Q4"), cell("330.6", cited(23)), cell("401.2")],
];

const fromTable = (grid: SourceGrid, language = "en") =>
  chartFromTable(grid, { blockId: "table-1", title: "Revenue by quarter", language });

describe("a chart from a table", () => {
  test("takes the first column as its labels and each other column of numbers as a series", () => {
    const chart = fromTable(QUARTERS) as ChartAttributes;
    expect(chart.kind).toBe("bar");
    expect(chart.title).toBe("Revenue by quarter");
    expect(chart.source).toEqual({ kind: "table", blockId: "table-1" });
    expect(chart.data.labelHeading).toBe("Quarter");
    expect(chart.data.series).toEqual(["2024", "2025"]);
    expect(chart.data.rows.map((row) => row.label)).toEqual(["Q1", "Q2", "Q3", "Q4"]);
    expect(chart.data.rows.map((row) => row.values.map((value) => value?.value))).toEqual([
      [289.4, 341.5],
      [301.2, 362],
      [309.8, 377.9],
      [330.6, 401.2],
    ]);
    expect(chart.data.cutFrom).toBeNull();
  });

  test("keeps each value's cell and the Citation in it, and marks a cell with none", () => {
    const chart = fromTable(QUARTERS) as ChartAttributes;
    const q3 = chart.data.rows[2]?.values[1];
    expect(q3?.shown).toBe("377.9");
    expect(q3?.from).toEqual({ kind: "cell", row: 3, column: 2 });
    expect(q3?.citation?.pageFrom).toBe(23);
    expect(q3?.citation?.quote).toBe("Revenue on page 23");
    const q4 = chart.data.rows[3]?.values[1];
    expect(q4?.from).toEqual({ kind: "cell", row: 4, column: 2 });
    expect(q4?.citation).toBeNull();
  });

  test("reads figures as written, in English and Chinese: scales, currencies, percents and brackets", () => {
    const chart = fromTable([
      [head("指标"), head("金额"), head("占比")],
      [cell("营业收入"), cell("1.5亿元", cited(4)), cell("41.2%")],
      [cell("净利润"), cell("(1,250)"), cell("约 3.5%")],
      [cell("Revenue"), cell("$1,482.6 million"), cell("n/a")],
    ]) as ChartAttributes;
    const values = chart.data.rows.map((row) => row.values.map((value) => value?.value ?? null));
    expect(values[0]?.[0]).toBe(150_000_000);
    expect(values[0]?.[1]).toBeCloseTo(0.412);
    expect(values[1]?.[0]).toBe(-1250);
    expect(values[2]?.[0]).toBe(1_482_600_000);
    // Words, not a number: kept as written, not drawn.
    expect(chart.data.rows[2]?.values[1]?.shown).toBe("n/a");
    expect(values[2]?.[1]).toBeNull();
    expect(chart.data.rows[0]?.values[1]?.unit).toEqual({ kind: "percent" });
  });

  test("leaves out a column with no number, and treats a first row of words as its headings", () => {
    const chart = fromTable([
      [cell("Site"), cell("Rise (mm/yr)"), cell("Notes")],
      [cell("Harbour"), cell("4.1"), cell("Gauge on the pier")],
      [cell("Estuary"), cell("3.2"), cell("Since 1990")],
    ]) as ChartAttributes;
    expect(chart.data.labelHeading).toBe("Site");
    expect(chart.data.series).toEqual(["Rise (mm/yr)"]);
    expect(chart.data.rows[1]?.values[0]?.from).toEqual({ kind: "cell", row: 2, column: 1 });
  });

  test("with no heading row, every row is drawn and the series have no name", () => {
    const chart = fromTable([
      [cell("Harbour"), cell("4.1")],
      [cell("Estuary"), cell("3.2")],
    ]) as ChartAttributes;
    expect(chart.data.series).toEqual([""]);
    expect(chart.data.rows.map((row) => row.label)).toEqual(["Harbour", "Estuary"]);
  });

  test("a cell spanning down gives its label to each row, and its value only to the first", () => {
    const merged: SourceCell = { ...cell("North"), continued: true };
    const span: SourceCell = { ...cell("12"), continued: true };
    const chart = fromTable([
      [head("Region"), head("Sales")],
      [cell("North"), cell("12")],
      [merged, span],
    ]) as ChartAttributes;
    expect(chart.data.rows.map((row) => row.label)).toEqual(["North", "North"]);
    expect(chart.data.rows[1]?.values[0]).toBeNull();
  });

  test("is null when no column holds a number", () => {
    expect(
      fromTable([
        [head("Claim"), head("Where")],
        [cell("Spring tides"), cell("Documents")],
      ]),
    ).toBeNull();
  });

  test(`is cut at ${MAX_CHART_VALUES} values, and says how many rows the table had`, () => {
    const rows = Array.from({ length: 300 }, (_, index) => [
      cell(`Day ${index + 1}`),
      cell(String(index), cited(1)),
      cell(String(index * 2)),
    ]);
    const chart = fromTable([[head("Day"), head("A"), head("B")], ...rows]) as ChartAttributes;
    expect(chart.data.rows).toHaveLength(MAX_CHART_VALUES / 2);
    expect(chart.data.cutFrom).toBe(300);
    expect(chartPoints(chart).length).toBeLessThanOrEqual(MAX_CHART_VALUES);
  });
});

describe("a chart from cited figures", () => {
  test("charts each line's first figure, labelled with its words, with the line's Citation", () => {
    const chart = chartFromFigures(
      [
        { text: "Q1 2025: $341.5 million", citation: cited(22) },
        { text: "Q2 2025 – $362.0 million", citation: cited(22) },
        { text: "Nothing to count here.", citation: null },
        { text: "三季度营业收入 3.88亿元", citation: null },
      ],
      { title: "Revenue", language: "en" },
    ) as ChartAttributes;
    expect(chart.source).toEqual({ kind: "figures" });
    expect(chart.data.rows.map((row) => row.label)).toEqual([
      "Q1 2025",
      "Q2 2025",
      "三季度营业收入",
    ]);
    expect(chart.data.rows.map((row) => row.values[0]?.shown)).toEqual([
      "$341.5 million",
      "$362.0 million",
      "3.88亿元",
    ]);
    expect(chart.data.rows[0]?.values[0]?.citation?.pageFrom).toBe(22);
    expect(chart.data.rows[2]?.values[0]?.from).toEqual({ kind: "text", line: 3 });
    expect(chart.data.rows[2]?.values[0]?.citation).toBeNull();
  });

  test("is null when no line has a figure", () => {
    expect(
      chartFromFigures([{ text: "Spring tides happen at new moon.", citation: null }], {
        title: "",
        language: "en",
      }),
    ).toBeNull();
  });
});

describe("what a chart draws", () => {
  const chart = fromTable(QUARTERS) as ChartAttributes;

  test("its cells are the columns chosen, row by row, whatever is drawn of them", () => {
    const pie: ChartAttributes = { ...chart, kind: "pie" };
    const cells = chartCells(pie);
    expect(cells.map((cell) => `${cell.row}:${cell.series}`).slice(0, 4)).toEqual([
      "0:0",
      "0:1",
      "1:0",
      "1:1",
    ]);
    expect(cells).toHaveLength(8);
  });

  test("bars and lines draw the columns chosen, a pie the last, a scatter chart the first two", () => {
    expect(drawnSeries(chart)).toEqual([0, 1]);
    expect(drawnSeries({ ...chart, drawn: [1] })).toEqual([1]);
    expect(drawnSeries({ ...chart, kind: "pie" })).toEqual([1]);
    expect(drawnSeries({ ...chart, kind: "scatter" })).toEqual([0, 1]);
    expect(chartPoints({ ...chart, kind: "scatter" }).map((point) => point.row)).toEqual([
      0, 1, 2, 3,
    ]);
  });

  test("its sources: the places quoted from, one Document, and the value with none", () => {
    const sources = chartSources(chart);
    expect(sources.values).toBe(8);
    expect(sources.uncited).toBe(1);
    expect(sources.documents).toBe(1);
    expect(sources.places.map((place) => [place.citation.pageFrom, place.count])).toEqual([
      [22, 4],
      [23, 3],
    ]);
  });

  test("its sources count the same cells whatever the kind, a scatter chart's both across and up", () => {
    const bar = chartSources(chart);
    const as = (kind: ChartAttributes["kind"], drawn = chart.drawn): ChartAttributes => ({
      ...chart,
      kind,
      drawn,
    });
    for (const kind of ["line", "pie", "scatter"] as const) {
      expect(chartSources(as(kind))).toEqual(bar);
    }
    // A scatter point is two cells: Q4 across (2024) is cited, up (2025) has none.
    const scatter = chartSources(as("scatter"));
    expect([scatter.values, scatter.uncited]).toEqual([8, 1]);
    // Columns chosen change it; the kind doesn't.
    expect(chartSources(as("pie", [1])).values).toBe(4);
  });

  test("a place's footnote is its Citation whose quote wasn't found, if any", () => {
    const missing = { ...cited(23), check: "not-found" as const, quote: "Not there" };
    const grid: SourceGrid = [
      [head("Quarter"), head("2025")],
      [cell("Q3"), cell("377.9", cited(23))],
      [cell("Q4"), cell("401.2", missing)],
    ];
    const places = chartSources(fromTable(grid) as ChartAttributes).places;
    expect(places).toHaveLength(1);
    expect(places[0]?.citation.check).toBe("not-found");
  });

  test("reads back what a Mind stores, leaving out what is malformed rather than guessing", () => {
    const stored = JSON.parse(JSON.stringify(chart)) as Record<string, unknown>;
    expect(readChart(stored)).toEqual(chart);
    const broken = readChart({
      kind: "radar",
      title: 7,
      drawn: [1, 9, "x", 1],
      data: {
        labelHeading: "Quarter",
        series: ["2024", "2025"],
        rows: [{ label: "Q1", values: [{ shown: "1", value: "1", from: { kind: "cell" } }, null] }],
        cutFrom: 0,
      },
    });
    expect(broken.kind).toBe("bar");
    expect(broken.title).toBe("");
    expect(broken.drawn).toEqual([1]);
    expect(broken.data.rows[0]?.values).toEqual([null, null]);
    expect(broken.data.cutFrom).toBeNull();
    expect(broken.source).toEqual({ kind: "table", blockId: null });
  });
});

describe("a number with no unit (#579)", () => {
  /** Revenue in "$… million", and a fifth quarter typed in by hand without its unit. */
  const MILLIONS: SourceGrid = [
    [head("Quarter"), head("2024"), head("2025")],
    [cell("Q1"), cell("$289.4 million", cited(22)), cell("$341.5 million", cited(22))],
    [cell("Q2"), cell("$301.2 million", cited(22)), cell("$362.0 million", cited(22))],
    [cell("Q3"), cell("$309.8 million", cited(23)), cell("$377.9 million", cited(23))],
    [cell("Q4"), cell("$330.6 million", cited(23)), cell("$401.2 million", cited(23))],
    [cell("Q5"), cell(""), cell("120.0")],
  ];

  test("in a column of '$… million', isn't drawn, and the foot counts it apart", () => {
    const chart = { ...(fromTable(MILLIONS) as ChartAttributes), kind: "pie" as const };
    expect(chart.data.rows[4]?.values[1]).toMatchObject({
      value: 120,
      unit: { kind: "plain" },
      scale: 1,
    });
    expect(chart.data.rows[0]?.values[1]).toMatchObject({ value: 341_500_000, scale: 1e6 });
    expect([...unitlessRows(chart.data, 1)]).toEqual([4]);
    const slices = chartPoints(chart);
    expect(slices.map((point) => point.row)).toEqual([0, 1, 2, 3]);
    // It has no Citation either, but isn't counted as a value with none: it isn't drawn.
    expect(chartSources(chart)).toMatchObject({ values: 8, uncited: 0, notDrawn: 0, noUnit: 1 });
    const bars = chartPoints({ ...chart, kind: "bar" });
    expect(bars.some((point) => point.row === 4)).toBe(false);
  });

  test("a column of bare numbers, or one with different units, is drawn as it is", () => {
    const bare = fromTable(QUARTERS) as ChartAttributes;
    expect(chartPoints(bare)).toHaveLength(8);
    expect(chartSources(bare).noUnit).toBe(0);
    const mixed = fromTable([
      [head("Item"), head("Value")],
      [cell("Revenue"), cell("$1.2 million")],
      [cell("Margin"), cell("41.2%")],
      [cell("Staff"), cell("120")],
    ]) as ChartAttributes;
    expect(chartPoints(mixed)).toHaveLength(3);
  });

  test("a scale alone counts as a unit: '1.5亿' beside a bare '8000'", () => {
    const chart = fromTable(
      [
        [head("季度"), head("营业收入")],
        [cell("一季度"), cell("1.5亿")],
        [cell("二季度"), cell("1.8亿")],
        [cell("三季度"), cell("8000")],
      ],
      "zh-CN",
    ) as ChartAttributes;
    expect(chartPoints(chart).map((point) => point.row)).toEqual([0, 1]);
    expect(chartSources(chart).noUnit).toBe(1);
  });

  test("a scatter chart leaves out a row whose number across or up has no unit", () => {
    const chart = { ...(fromTable(MILLIONS) as ChartAttributes), kind: "scatter" as const };
    const grid = MILLIONS.map((row, index) =>
      index === 5 ? [cell("Q5"), cell("$310.0 million"), cell("120.0")] : row,
    );
    const both = { ...(fromTable(grid) as ChartAttributes), kind: "scatter" as const };
    expect(chartPoints(chart).map((point) => point.row)).toEqual([0, 1, 2, 3]);
    expect(chartPoints(both).map((point) => point.row)).toEqual([0, 1, 2, 3]);
  });

  test("a chart made before values kept their scale still leaves out a bare number beside a currency", () => {
    const made = fromTable(MILLIONS) as ChartAttributes;
    const stored = JSON.parse(JSON.stringify(made)) as {
      data: { rows: { values: ({ scale?: number } | null)[] }[] };
    };
    for (const row of stored.data.rows)
      for (const value of row.values) if (value) delete value.scale;
    const older = readChart(stored as unknown as Record<string, unknown>);
    expect(older.data.rows[0]?.values[1]?.scale).toBeNull();
    expect([...unitlessRows(older.data, 1)]).toEqual([4]);
  });
});
