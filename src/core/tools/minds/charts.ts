/**
 * A chart the User asks the assistant for (#1091): add_to_mind and create_mind
 * draw it from the last table their content writes, as "Chart from table"
 * does (#436), and put it right after that table, in the same change. Every
 * value keeps its cell's Citation.
 */
import {
  CHART_KINDS,
  type ChartKind,
  chartFromTable,
  gridOfTable,
  kindFits,
} from "../../../shared/charts";
import { translate } from "../../../shared/i18n";
import type { NodeJSON } from "../../answers/blocks";
import { CHART_NODE, type ChartProblem } from "../../api";
import { isRecord } from "../../errors";
import type { Language } from "../../language";

export interface ChartAsked {
  /** Null: the kind the chart Block starts with. */
  kind: ChartKind | null;
  title: string;
}

export const chartProperty = {
  chart: {
    type: "object",
    description:
      "Only when the User asks for a chart: IncarnaMind draws it from the last table in content and puts it right after that table, in the same change. Write the figures as that table, each cell cited; chart only figures the Documents state, or ones calculate gives. Give it before content. To chart a table already in the Mind, give chart alone, with no content: IncarnaMind charts the table the User's cursor is in, or the Mind's only table.",
    properties: {
      kind: {
        type: "string",
        enum: [...CHART_KINDS],
        description: "The kind the User asked for. Leave it out when they named none.",
      },
      title: {
        type: "string",
        description: "A short title, from the User's words, in their language.",
      },
    },
    required: ["title"],
    additionalProperties: false,
  },
};

const isKind = (value: unknown): value is ChartKind =>
  (CHART_KINDS as readonly unknown[]).includes(value);

export function chartAsked(value: unknown): ChartAsked | null {
  if (!isRecord(value)) return null;
  return {
    kind: isKind(value.kind) ? value.kind : null,
    title: typeof value.title === "string" ? value.title.trim() : "",
  };
}

// A figure calculate gave, as content writes it: "[=2]".
const CALCULATED = /\[=\d+\]/;

/**
 * Which slots of the last table in `markdown` hold a figure from calculate, by row and column on
 * its grid, the heading row 0.
 */
export function calculatedCells(markdown: string): (row: number, column: number) => boolean {
  let table: string[] = [];
  let open: string[] = [];
  for (const line of markdown.split("\n")) {
    if (line.trim().startsWith("|")) open.push(line.trim());
    else if (open.length > 0) {
      table = open;
      open = [];
    }
  }
  if (open.length > 0) table = open;
  const rows = table
    .filter((line) => !/^\|?\s*:?-{3,}/.test(line))
    .map((line) => line.replace(/^\|/, "").replace(/\|$/, "").split("|"));
  return (row, column) => CALCULATED.test(rows[row]?.[column] ?? "");
}

/**
 * The chart Block of `table`, whose id is `tableId`; or, when none may be drawn, why, and what
 * to tell the model, which keeps the table. Every value it draws must be cited, or come from
 * calculate (`calculated`): a figure the model wrote with neither isn't drawn.
 */
export function chartOfTable(
  table: NodeJSON,
  tableId: string,
  asked: ChartAsked,
  language: Language,
  calculated: (row: number, column: number) => boolean = () => false,
): { chart: NodeJSON } | { problem: ChartProblem; why: string } {
  const chart = chartFromTable(gridOfTable(table), {
    blockId: tableId,
    title: asked.title,
    language,
  });
  if (!chart) {
    return {
      problem: "noNumbers",
      why: "No chart was drawn: the table holds no column of numbers. The table stays; say so in your reply.",
    };
  }
  const kind = asked.kind ?? chart.kind;
  if (!kindFits(kind, chart.data)) {
    return {
      problem: "needsTwo",
      why: `No chart was drawn: ${translate(language, "chart.scatter.needsTwo")}. The table stays; say that in your reply, in those words.`,
    };
  }
  const unbacked = [...chart.data.rows, ...chart.data.cutRows]
    .flatMap((row) => row.values)
    .filter(
      (value) =>
        value?.value != null &&
        !value.citation &&
        !(value.from.kind === "cell" && calculated(value.from.row, value.from.column)),
    ).length;
  if (unbacked > 0) {
    return {
      problem: "uncited",
      why: `No chart was drawn: ${unbacked === 1 ? "1 of the table's figures has" : `${unbacked} of the table's figures have`} no Citation and come from no calculation. A chart draws only figures the Documents state, cited, or ones calculate gives. The table stays; say so in your reply.`,
    };
  }
  return { chart: { type: CHART_NODE, attrs: { ...chart, kind } } };
}
