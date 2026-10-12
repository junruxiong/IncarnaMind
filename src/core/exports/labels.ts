import { translate } from "../../shared/i18n";
import type { Language } from "../language";

export interface ExportLabels {
  question: string;
  unverified: string;
  chart(title: string): string;
  /** With the space that follows it, if any: "Source: ", "数据来源：". */
  chartSources: string;
  chartUncited(count: number): string;
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
