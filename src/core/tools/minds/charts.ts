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
import { CHART_NODE } from "../../api";
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
      "Only when the User asks for a chart: IncarnaMind draws it from the last table in content and puts it right after that table, in the same change. Write the figures as that table, each cell cited; chart only figures the Documents state, or ones calculate gives. Give it before content.",
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

/**
 * The chart Block to put after the table whose id is `after`; or, when none can be drawn, what
 * to tell the model, which keeps the table. `ids` are the written Blocks' ids, in `blocks`' order.
 */
export function chartOfTable(
  blocks: readonly NodeJSON[],
  ids: readonly string[],
  asked: ChartAsked,
  language: Language,
): { chart: NodeJSON; after: string } | { why: string } {
  const at = blocks.findLastIndex((block) => block.type === "table");
  const table = blocks[at];
  const after = ids[at];
  if (!table || !after) {
    return {
      why: "No chart was drawn: content holds no table. Say so in your reply; to chart figures, write them as a cited table with chart.",
    };
  }
  const chart = chartFromTable(gridOfTable(table), {
    blockId: after,
    title: asked.title,
    language,
  });
  if (!chart) {
    return {
      why: "No chart was drawn: the table holds no column of numbers. The table stays; say so in your reply.",
    };
  }
  const kind = asked.kind ?? chart.kind;
  if (!kindFits(kind, chart.data)) {
    return {
      why: `No chart was drawn: ${translate(language, "chart.scatter.needsTwo")}. The table stays; say that in your reply, in those words.`,
    };
  }
  return { chart: { type: CHART_NODE, attrs: { ...chart, kind } }, after };
}
