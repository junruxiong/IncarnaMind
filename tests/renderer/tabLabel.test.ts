import { describe, expect, test } from "vitest";
import { TAB_LABEL_CHARACTERS, tabLabelEm } from "../../src/renderer/src/tabLabel";

describe("the room a tab keeps for its name", () => {
  test("is ten characters of a long name, and all of a short one", () => {
    expect(tabLabelEm("Weekly report to the board")).toBe(tabLabelEm("Weekly rep"));
    expect(tabLabelEm("Q3")).toBeLessThan(tabLabelEm("Weekly report"));
    expect(tabLabelEm("")).toBe(0);
  });

  test("is wider for Chinese, a character an em", () => {
    expect(tabLabelEm("茶叶产地与历史研究笔记")).toBeCloseTo(TAB_LABEL_CHARACTERS + 0.4, 5);
    expect(tabLabelEm("预算")).toBeCloseTo(2.4, 5);
    expect(tabLabelEm("Q3 预算")).toBeCloseTo(0.7 * 3 + 2 + 0.4, 5);
  });

  test("counts an emoji or a letter with its accent as it reads, not by code unit", () => {
    expect(tabLabelEm("🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊")).toBeCloseTo(0.7 * TAB_LABEL_CHARACTERS + 0.4, 5);
  });
});
