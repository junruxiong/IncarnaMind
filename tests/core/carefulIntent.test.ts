import { describe, expect, test } from "vitest";
import { asksToCheckOrCalculate } from "../../src/core/answers/carefulIntent";

describe("A Question that asks to check or calculate", () => {
  // protects=only a Question that calculates or checks figures or claims gets the larger thinking budget; fails_when=a request to write, summarize or check a grammar matches, or a calculation or a check of figures doesn't; no_claim=the words are a heuristic, not a classifier.
  test.each([
    "Calculate the leverage headroom after the June 2027 step-down",
    "Check these figures",
    "Can you verify these figures?",
    "Please double-check the numbers in section 3",
    "Does the revenue growth add up?",
    "What percentage of sales is online?",
    "What is the ratio of net debt to EBITDA?",
    "Check whether the totals match the table",
    "计算增长率",
    "计算一下 2027 年的杠杆率",
    "核对这份报告里的数字",
    "在线销售占比是多少？",
  ])("%s", (question) => {
    expect(asksToCheckOrCalculate(question)).toBe(true);
  });

  // protects=everyday words that contain check, compute or 计算 don't send a Question to high effort; fails_when=one of them matches; no_claim=none beyond the examples.
  test.each([
    "Write a summary page of these documents",
    "What does the facilities agreement say about dividends?",
    "Make it longer",
    "Check my grammar",
    "Check whether Northwind stays inside its covenant",
    "Add a check-in section to the plan",
    "Summarize the notes on cloud computing",
    "What does a background check cover?",
    "Where does the data validation happen?",
    "为这些文档写一页摘要",
    "客户对送货说了什么？",
    "检查一下我的错别字",
    "计算机网络是什么？",
    "云计算的优点",
    "短信验证码收不到怎么办",
  ])("%s isn't one", (question) => {
    expect(asksToCheckOrCalculate(question)).toBe(false);
  });
});
