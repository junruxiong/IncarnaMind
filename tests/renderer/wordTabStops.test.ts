import { describe, expect, test } from "vitest";
import { alignsColumns, tabStopAfter, tabWidth } from "../../src/renderer/src/editor/wordTabStops";

describe("a tab's stop and width", () => {
  test("goes to the paragraph's next stop, its hanging indent, or the next default stop", () => {
    const tabs = [
      { at: 120, align: "left" as const, leader: "none" as const },
      { at: 481.9, align: "right" as const, leader: "dot" as const },
    ];
    expect(tabStopAfter(40, { tabs })).toMatchObject({ at: 120, align: "left" });
    expect(tabStopAfter(120, { tabs })).toMatchObject({ at: 481.9, leader: "dot" });
    expect(tabStopAfter(500, { tabs })).toMatchObject({ at: 504, align: "left" });
    expect(tabStopAfter(12, { left: 28.35, firstLine: -28.35 })).toMatchObject({ at: 28.35 });
    expect(tabStopAfter(30, { left: 28.35, firstLine: -28.35 })).toMatchObject({ at: 36 });
    expect(tabStopAfter(0, { tabs: [{ at: 50, align: "bar", leader: "none" }] })).toMatchObject({
      at: 36,
    });
  });

  test("reaches its stop, less what a right, centred or decimal stop lines up after it", () => {
    const measures = { x: 100, stopX: 400, following: 60, beforeDecimal: 40 };
    const stop = (align: "left" | "right" | "center" | "decimal") => ({
      at: 0,
      align,
      leader: "none" as const,
    });
    expect(tabWidth(stop("left"), measures)).toBe(300);
    expect(tabWidth(stop("right"), measures)).toBe(240);
    expect(tabWidth(stop("center"), measures)).toBe(270);
    expect(tabWidth(stop("decimal"), measures)).toBe(260);
    expect(tabWidth(stop("right"), { ...measures, following: 500 })).toBe(0);
  });

  test("only a paragraph with right, centred or decimal stops lines text up in columns", () => {
    expect(alignsColumns({ tabs: [{ at: 120, align: "left", leader: "none" }] })).toBe(false);
    expect(alignsColumns({ tabs: [{ at: 480, align: "right", leader: "none" }] })).toBe(true);
    expect(alignsColumns({})).toBe(false);
  });
});
