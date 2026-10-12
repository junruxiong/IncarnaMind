import { mergeAttributes, Node, type NodeViewRenderer } from "@tiptap/core";
import { CHART_NODE } from "../../../core/api";
import { CHART_KINDS, drawnSeries, readChart } from "../../../shared/charts";
import { citationReference } from "../../../shared/citations";

export interface ChartOptions {
  /** Draws the chart, with its menus, its points' cards and "Show as table". */
  view: NodeViewRenderer | null;
}

/**
 * An attribute kept in the HTML as JSON in `data-<name>`, so a chart copied
 * and pasted (into a Note, or another Mind) keeps everything it holds.
 */
function jsonAttribute(name: string) {
  return {
    default: null,
    parseHTML: (element: HTMLElement) => {
      const raw = element.getAttribute(`data-${name}`);
      if (raw === null) return null;
      try {
        return JSON.parse(raw) as unknown;
      } catch {
        return null;
      }
    },
    renderHTML: (attributes: Record<string, unknown>) =>
      attributes[name] === null || attributes[name] === undefined
        ? {}
        : { [`data-${name}`]: JSON.stringify(attributes[name]) },
  };
}

/** A chart's values as text, a row a line, each value with its source: what copying it as text gives. */
function chartText(attributes: Record<string, unknown>): string {
  const chart = readChart(attributes);
  const drawn = drawnSeries(chart);
  const lines = [
    chart.title,
    [chart.data.labelHeading, ...drawn.map((series) => chart.data.series[series] ?? "")].join("\t"),
    ...chart.data.rows.map((row) =>
      [
        row.label,
        ...drawn.map((series) => {
          const value = row.values[series];
          if (!value) return "";
          const source = value.citation ? citationReference(value.citation) : "";
          return source ? `${value.shown} ${source}` : value.shown;
        }),
      ].join("\t"),
    ),
  ];
  return lines.filter((line) => line.trim() !== "").join("\n");
}

/**
 * A chart (#436): a Block with no text of its own, drawn from the values it
 * holds (see `ChartAttributes` in src/shared/charts.ts), each kept with the
 * cell or line it came from and the Citation that was there. It moves and is
 * deleted as one Block. Copied, it keeps everything; as HTML for another
 * app, it is its title and a table of its values.
 */
export const Chart = Node.create<ChartOptions>({
  name: CHART_NODE,
  group: "block",
  atom: true,
  selectable: true,
  draggable: false,

  addOptions() {
    return { view: null };
  },

  addAttributes() {
    return {
      kind: {
        default: "bar",
        parseHTML: (element: HTMLElement) => {
          const kind = element.getAttribute("data-kind");
          return kind && (CHART_KINDS as readonly string[]).includes(kind) ? kind : "bar";
        },
        renderHTML: (attributes: Record<string, unknown>) => ({ "data-kind": attributes.kind }),
      },
      title: {
        default: "",
        parseHTML: (element: HTMLElement) => element.getAttribute("data-title") ?? "",
        renderHTML: (attributes: Record<string, unknown>) =>
          attributes.title ? { "data-title": attributes.title } : {},
      },
      drawn: jsonAttribute("drawn"),
      data: jsonAttribute("data"),
      source: jsonAttribute("source"),
    };
  },

  parseHTML() {
    return [{ tag: `div[data-type="${CHART_NODE}"]` }];
  },

  renderHTML({ node, HTMLAttributes }) {
    const chart = readChart(node.attrs);
    const drawn = drawnSeries(chart);
    const cell = (tag: "th" | "td", text: string) => [tag, {}, text] as const;
    return [
      "div",
      mergeAttributes(HTMLAttributes, { "data-type": CHART_NODE }),
      ...(chart.title ? [["p", {}, chart.title]] : []),
      [
        "table",
        {},
        [
          "tr",
          {},
          cell("th", chart.data.labelHeading),
          ...drawn.map((series) => cell("th", chart.data.series[series] ?? "")),
        ],
        ...chart.data.rows.map((row) => [
          "tr",
          {},
          cell("td", row.label),
          ...drawn.map((series) => cell("td", row.values[series]?.shown ?? "")),
        ]),
      ],
    ];
  },

  renderText({ node }) {
    return chartText(node.attrs);
  },

  addNodeView() {
    return this.options.view;
  },
});
