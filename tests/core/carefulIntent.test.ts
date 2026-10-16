import { describe, expect, test } from "vitest";
import { asksToCheckOrCalculate } from "../../src/core/answers/carefulIntent";

describe("A Question that asks to check or calculate", () => {
  // protects=only a Question that checks or works out figures gets the larger thinking budget; fails_when=a request to write or summarize matches, or a check or a calculation doesn't; no_claim=the words are a heuristic, not a classifier.
  test.each([
    "Calculate the leverage headroom after the June 2027 step-down",
    "Check whether Northwind stays inside its covenant",
    "Can you verify these figures?",
    "Does the revenue growth add up?",
    "What percentage of sales is online?",
    "What is the ratio of net debt to EBITDA?",
    "计算一下 2027 年的杠杆率",
    "核对这份报告里的数字",
    "在线销售占比是多少？",
  ])("%s", (question) => {
    expect(asksToCheckOrCalculate(question)).toBe(true);
  });

  test.each([
    "Write a summary page of these documents",
    "What does the facilities agreement say about dividends?",
    "Make it longer",
    "为这些文档写一页摘要",
    "客户对送货说了什么？",
  ])("%s isn't one", (question) => {
    expect(asksToCheckOrCalculate(question)).toBe(false);
  });
});
