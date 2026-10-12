/**
 * An export's own words, in the interface's language: what the Markdown and
 * Word writers add around the Mind's text.
 */
import { translate } from "../../shared/i18n";
import type { Language } from "../language";

export interface ExportLabels {
  /** Marks a Question, e.g. "Question:". */
  question: string;
  /** Follows an unverified Citation's source in its footnote, e.g. "[unverified]". */
  unverified: string;
  /** A chart's caption: "Chart: Revenue by quarter", or "Chart" when it has no title. */
  chart(title: string): string;
  /** Before a chart's sources, with the space that follows it, if any: "Source: ", "数据来源：". */
  chartSources: string;
  /** After them, the values drawn with no Citation: "2 values have no Citation." */
  chartUncited(count: number): string;
  /** A chart of a table cut to fit: "Only the first 250 of 300 rows are charted." */
  chartCut(rows: number, total: number): string;
}

export function exportLabels(language: Language): ExportLabels {
  const t = (key: Parameters<typeof translate>[1], params?: Parameters<typeof translate>[2]) =>
    translate(language, key, params);
  return {
    question: t("export.question"),
    unverified: t("export.unverified"),
    chart: (title) => (title ? t("export.chart.titled", { title }) : t("export.chart")),
    // No space after full-width punctuation (docs/agents/zh-CN.md).
    chartSources: t("export.chart.sources").replace(/[^\u3000-\u303f\uff00-\uffef]$/, "$& "),
    chartUncited: (count) =>
      count === 1 ? t("export.chart.uncited.one") : t("export.chart.uncited.other", { count }),
    chartCut: (rows, total) => t("export.chart.cut", { rows, total }),
  };
}
