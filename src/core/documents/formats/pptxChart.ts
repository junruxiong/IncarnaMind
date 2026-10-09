/**
 * A chart part (DrawingML charts, c:chartSpace) read from its caches for the
 * slide renderer: its kind, title, categories, series with their values and
 * colours, legend and data labels. Charts are drawn from the numbers the
 * file caches; the embedded workbook is never opened. Kinds the renderer
 * doesn't draw (scatter, radar, stock, 3-D surfaces…) come back null, and the
 * slide shows a labelled box with the chart's text instead.
 */
import {
  type ColourContext,
  containerColour,
  css,
  fillElement,
  numberAttr,
  readFill,
} from "./drawingml";
import { child, descendants, elements, local, parseXml, textOf, type XmlElement } from "./xml";

export interface ChartSeries {
  name: string;
  values: (number | null)[];
  colour: string;
  /** Per-point colours (a pie's slices, or points coloured by hand). */
  pointColours: (string | null)[];
}

export interface ChartData {
  kind: "bar" | "column" | "line" | "area" | "pie" | "doughnut";
  grouping: "clustered" | "stacked" | "percentStacked" | "standard";
  title: string | null;
  categories: string[];
  series: ChartSeries[];
  legend: "right" | "left" | "top" | "bottom" | null;
  /** Whether each value is written on its bar, point or slice. */
  dataLabels: boolean;
  /** Whether the value axis draws its major gridlines. */
  gridlines: boolean;
  /** The text colour of the axes and labels, if the chart sets one. */
  textColour: string | null;
}

/** Office's default series colours: accent 1–6, then the same shaded. */
const ACCENTS = ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6"];

function accentColour(index: number, context: ColourContext): string {
  const name = ACCENTS[index % ACCENTS.length] as string;
  const rgb = context.theme.colours[name] ?? { r: 68, g: 114, b: 196 };
  const round = Math.floor(index / ACCENTS.length);
  const factor = round === 0 ? 1 : round % 2 === 1 ? 0.6 : 0.8;
  return css({ r: rgb.r * factor, g: rgb.g * factor, b: rgb.b * factor, a: 1 });
}

/** The values in a c:numCache / c:strCache, by point index. */
function cached(element: XmlElement | undefined): string[] {
  if (!element) return [];
  const count = numberAttr(descendants(element, "c:ptCount")[0], "val", 0);
  const values: string[] = Array.from({ length: count }, () => "");
  // Multi-level categories: the first level is the one next to the axis.
  const level = descendants(element, "c:lvl")[0] ?? element;
  for (const point of descendants(level, "c:pt")) {
    const index = numberAttr(point, "idx", values.length);
    if (index > 10_000) continue;
    values[index] = textOf(child(point, "c:v") ?? point);
  }
  return values;
}

function richText(element: XmlElement | undefined): string {
  if (!element) return "";
  return descendants(element, "a:p")
    .map((paragraph) => descendants(paragraph, "a:t").map(textOf).join(""))
    .join(" ")
    .trim();
}

const KINDS: Record<string, ChartData["kind"]> = {
  barChart: "column",
  bar3DChart: "column",
  lineChart: "line",
  line3DChart: "line",
  areaChart: "area",
  area3DChart: "area",
  pieChart: "pie",
  pie3DChart: "pie",
  ofPieChart: "pie",
  doughnutChart: "doughnut",
};

/** A chart part's XML as `ChartData`; null for a kind the renderer doesn't draw. */
export function readChart(xml: string, context: ColourContext): ChartData | null {
  const root = parseXml(xml);
  const chart = child(root, "c:chart");
  const plot = chart && child(chart, "c:plotArea");
  if (!chart || !plot) return null;
  const group = elements(plot).find((each) => KINDS[local(each.name)] !== undefined);
  if (!group) return null;
  let kind = KINDS[local(group.name)] as ChartData["kind"];
  if (kind === "column" && child(group, "c:barDir")?.attrs.val === "bar") kind = "bar";
  const groupingValue = child(group, "c:grouping")?.attrs.val;
  const grouping: ChartData["grouping"] =
    groupingValue === "stacked" || groupingValue === "percentStacked"
      ? groupingValue
      : kind === "column" || kind === "bar"
        ? "clustered"
        : "standard";
  const round = kind === "pie" || kind === "doughnut";
  const allSeries = elements(group).filter((each) => local(each.name) === "ser");
  // Office colours a pie's slices one by one unless told not to; a lone series' bars when told to.
  const varyAttribute = child(group, "c:varyColors")?.attrs.val;
  const varyColours = round
    ? varyAttribute !== "0"
    : varyAttribute === "1" && allSeries.length === 1;

  let categories: string[] = [];
  const series: ChartSeries[] = [];
  for (const [index, ser] of allSeries.entries()) {
    const tx = child(ser, "c:tx");
    const name = tx ? (cached(tx)[0] ?? textOf(descendants(tx, "c:v")[0] ?? tx)).trim() : "";
    const cat = child(ser, "c:cat");
    if (categories.length === 0 && cat) categories = cached(cat);
    const val = child(ser, "c:val");
    const values = cached(val).map((value) => {
      const number = Number(value);
      return value === "" || !Number.isFinite(number) ? null : number;
    });
    const order = numberAttr(child(ser, "c:idx"), "val", index);
    const fill = readFill(fillElement(child(ser, "c:spPr")), context);
    const line = child(child(ser, "c:spPr") ?? ser, "a:ln");
    const lineColour = line ? containerColour(fillElement(line), context) : null;
    const colour =
      kind === "line" && lineColour
        ? lineColour
        : fill?.kind === "solid"
          ? fill.colour
          : fill?.kind === "gradient"
            ? (fill.stops[0]?.colour ?? accentColour(order, context))
            : accentColour(order, context);
    const pointColours: (string | null)[] = values.map((_, at) =>
      varyColours ? accentColour(at, context) : null,
    );
    for (const point of elements(ser).filter((each) => local(each.name) === "dPt")) {
      const at = numberAttr(child(point, "c:idx"), "val", -1);
      const pointFill = readFill(fillElement(child(point, "c:spPr")), context);
      if (at >= 0 && at < pointColours.length && pointFill?.kind === "solid") {
        pointColours[at] = pointFill.colour;
      }
    }
    series.push({ name, values, colour, pointColours });
  }
  if (series.length === 0) return null;
  const longest = Math.max(...series.map((each) => each.values.length));
  if (categories.length < longest) {
    categories = Array.from({ length: longest }, (_, at) => categories[at] || String(at + 1));
  }

  // The title: its own text, or the only series' name when the chart titles itself.
  const titleElement = child(chart, "c:title");
  const deleted = child(chart, "c:autoTitleDeleted")?.attrs.val === "1";
  let title: string | null = titleElement ? richText(child(titleElement, "c:tx")) : null;
  if (titleElement && !title && series.length === 1) title = series[0]?.name || null;
  if (!titleElement && !deleted && series.length === 1 && (kind === "pie" || kind === "doughnut")) {
    title = series[0]?.name || null;
  }

  const legendElement = child(chart, "c:legend");
  const position = child(legendElement ?? chart, "c:legendPos")?.attrs.val ?? "r";
  const legend: ChartData["legend"] = legendElement
    ? position === "b"
      ? "bottom"
      : position === "t"
        ? "top"
        : position === "l"
          ? "left"
          : "right"
    : null;
  const labels = descendants(group, "c:dLbls").some(
    (each) =>
      child(each, "c:showVal")?.attrs.val === "1" ||
      child(each, "c:showPercent")?.attrs.val === "1",
  );
  const valueAxis = child(plot, "c:valAx");
  const gridlines = valueAxis ? child(valueAxis, "c:majorGridlines") !== undefined : false;
  const axisText = valueAxis && descendants(valueAxis, "a:defRPr")[0];
  const textColour = axisText ? containerColour(fillElement(axisText), context) : null;
  return {
    kind,
    grouping,
    title,
    categories,
    series,
    legend,
    dataLabels: labels,
    gridlines,
    textColour: textColour ?? defaultTextColour(context),
  };
}

/** Office's chart text: the text colour, a third of the way to the background (tx1 at lumMod 65%, lumOff 35%). */
function defaultTextColour(context: ColourContext): string {
  const text = context.theme.colours[context.map.tx1 ?? "dk1"] ?? { r: 0, g: 0, b: 0 };
  const back = context.theme.colours[context.map.bg1 ?? "lt1"] ?? { r: 255, g: 255, b: 255 };
  const mix = (a: number, b: number) => a * 0.65 + b * 0.35;
  return css({ r: mix(text.r, back.r), g: mix(text.g, back.g), b: mix(text.b, back.b), a: 1 });
}

/** A chart's text for its labelled box: title, then each series with its values. */
export function chartLines(xml: string): string[] {
  const root = parseXml(xml);
  const lines: string[] = [];
  for (const title of descendants(root, "c:title")) {
    const text = descendants(title, "a:t").map(textOf).join("");
    if (text) lines.push(text);
  }
  for (const series of descendants(root, "c:ser")) {
    const name = descendants(child(series, "c:tx") ?? series, "c:v")[0];
    const categories = child(series, "c:cat");
    const values = child(series, "c:val") ?? child(series, "c:yVal");
    const labels = categories ? descendants(categories, "c:v").map(textOf) : [];
    const numbers = values ? descendants(values, "c:v").map(textOf) : [];
    const pairs = numbers.map((number, at) => (labels[at] ? `${labels[at]}: ${number}` : number));
    lines.push([name ? textOf(name) : "", pairs.join(", ")].filter(Boolean).join(" — "));
  }
  return lines.filter(Boolean);
}
