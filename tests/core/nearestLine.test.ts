import { describe, expect, test } from "vitest";
import { figuresMatch, nearestLine } from "../../src/core/answers/checks/nearest";

const passages = [
  {
    id: "P1",
    text: [
      "[p. 1] You are entitled to 27 days' paid holiday per year plus public holidays.",
      "Staff may work from home up to three days a week.",
    ].join("\n"),
  },
  { id: "P2", text: "Notice: 90 days." },
];

describe("the line of a Passage nearest a sentence (#770)", () => {
  test("takes the line that shares the most words, without its Unit mark", () => {
    expect(
      nearestLine("New hires get 30 days of paid holiday and 5 public holidays.", passages),
    ).toMatchObject({
      id: "P1",
      text: "You are entitled to 27 days' paid holiday per year plus public holidays.",
    });
  });

  test("none when no line shares enough words with it", () => {
    expect(nearestLine("Penguins nest on the cliffs.", passages)).toBeNull();
    expect(nearestLine("Termination notice is long.", passages)).toBeNull();
  });

  test("reads Chinese by pairs of characters", () => {
    expect(
      nearestLine("新员工每年有三十天带薪年假", [
        { id: "P1", text: "员工每年享有二十七天带薪年假。" },
      ]),
    ).toMatchObject({ id: "P1" });
  });

  test("two common words don't make an unrelated sentence's line near, in English and Chinese", () => {
    const company = [
      { id: "P1", text: "The company reported its results for the year 2025 in March." },
    ];
    expect(
      nearestLine("Our company will open three new offices in Singapore during 2025.", company),
    ).toBeNull();
    const chinese = [{ id: "P1", text: "公司发布了年度报告，营收增长明显。" }];
    expect(nearestLine("这家公司的年度旅行计划包括去云南和四川的长途徒步。", chinese)).toBeNull();
    // The same words in a sentence about the same thing still count.
    expect(nearestLine("The company reported results for the year 2025.", company)).toMatchObject({
      id: "P1",
    });
  });
});

describe("the figures a sentence states, in a text", () => {
  test("match across separators and percent forms, and differ when one isn't there", () => {
    const sheet = "Item,2024,2025\nRevenue,1231.0,1482.6\nOperating margin,0.18,0.21";
    expect(figuresMatch("Operating margin was 21% in 2025.", sheet)).toBe(true);
    expect(figuresMatch("Revenue was $1,482.6 million in 2025.", sheet)).toBe(true);
    expect(figuresMatch("Operating margin was 25% in 2025.", sheet)).toBe(false);
    expect(figuresMatch("Revenue was 1,482.0 in 2025.", sheet)).toBe(false);
    expect(figuresMatch("Margins improved.", sheet)).toBe(true);
  });
});
