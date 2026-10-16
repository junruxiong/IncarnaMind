import { describe, expect, test } from "vitest";
import { holdsPeriods, isPeriod } from "../../src/shared/periods";

describe("a label that is a period", () => {
  test.each([
    ["years", ["2023", "FY2024", "FY 2024", "Fiscal 2024", "2024E", "2023/24", "FY24", "FY'24"]],
    ["years of a plan", ["Year 1", "Yr 2", "第1年", "第三年"]],
    [
      "quarters",
      ["Q1", "Q1 2025", "2025Q1", "2025 Q1", "2024年Q1", "1Q25", "Q1 FY25", "Q1-2025", "Quarter 1"],
    ],
    ["halves", ["H1 2025", "1H25", "H2", "H1 FY25"]],
    ["months", ["Jan", "January 2025", "Jan-25", "Sept 2024", "2025-03", "2025/3", "03/2025"]],
    ["dates", ["2025-03-31", "31/03/2025", "30 June 2025", "June 30, 2025", "31 Dec"]],
    ["Chinese years", ["2024年", "2024年度", "2024财年", "二〇二四年"]],
    [
      "Chinese quarters",
      ["一季度", "第一季度", "2024年第一季度", "2024年一季度", "第3季度", "上半年", "全年"],
    ],
    [
      "Chinese months and dates",
      ["3月", "2025年3月", "2025年1-6月", "十二月", "一月份", "2024年12月31日", "12月31日"],
    ],
    ["a note after the label", ["2024 (restated)", "FY2025¹", "2024（未经审计）"]],
  ])("%s", (_name, labels) => {
    for (const label of labels) expect(isPeriod(label), label).toBe(true);
  });

  test("words, numbers and codes are not", () => {
    for (const label of [
      "",
      "Total",
      "Revenue",
      "Net income",
      "Q5",
      "13月",
      "1,234",
      "12.5%",
      "3.2x",
      "2 h",
      "10-K",
      "Metric",
      "营业收入",
      "同比",
      "Change vs 2023",
      "2023 revenue",
      "Year-on-year",
      "YoY",
      "第一名",
    ]) {
      expect(isPeriod(label), label).toBe(false);
    }
  });
});

describe("labels that are periods", () => {
  test("at least two, and more than half of those that are not empty", () => {
    expect(holdsPeriods(["2023", "2024", "2025"])).toBe(true);
    expect(holdsPeriods(["Q1", "Q2", "Q3", "Q4", "Total"])).toBe(true);
    expect(holdsPeriods(["2023", "2024", "Change"])).toBe(true);
    expect(holdsPeriods(["2023", "", "2024", " "])).toBe(true);
    expect(holdsPeriods(["2023", "Revenue", "Margin"])).toBe(false);
    expect(holdsPeriods(["2023"])).toBe(false);
    expect(holdsPeriods(["2023", "Notes"])).toBe(false);
    expect(holdsPeriods(["Revenue", "Margin", "Headcount"])).toBe(false);
    expect(holdsPeriods([])).toBe(false);
  });

  test("in Chinese", () => {
    expect(holdsPeriods(["2023年", "2024年", "同比"])).toBe(true);
    expect(holdsPeriods(["第一季度", "第二季度", "第三季度", "第四季度"])).toBe(true);
    expect(holdsPeriods(["营业收入", "净利润", "毛利率"])).toBe(false);
  });
});
