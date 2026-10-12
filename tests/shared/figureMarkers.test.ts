/**
 * Where an Answer writes its calculated figures (ADR-0021): a marker such as
 * "[=1]", which every reader of the Answer turns into the figure's shown text,
 * the number the model wrote right before it shown once or kept and flagged.
 */
import { describe, expect, test } from "vitest";
import { markdownToBlocks } from "../../src/core/answers/markdown";
import type { NodeJSON } from "../../src/shared/blockJson";
import {
  numberBefore,
  type PlacedFigure,
  withFigures,
  withFigureText,
} from "../../src/shared/figureMarkers";
import { formulaText } from "../../src/shared/formula";

const figure = (
  value: number | boolean,
  shown: string,
  language: "en" | "zh-CN" = "en",
): PlacedFigure => ({
  value,
  shown,
  show: { language, as: null, decimals: null, scale: null },
});

const GROWTH = figure(0.204387, "20.4%");
const POINTS = figure(0.023, "2.3 percentage points");
const AMOUNT = figure(251_612_345, "$251.6 million");
const INCREASE = figure(304_573_600, "30,457.36万元", "zh-CN");

describe("the number a model wrote before a figure's marker", () => {
  test("is the figure when it reads as its value to the decimals written, in the same unit", () => {
    expect(numberBefore("Revenue grew 20.4% ", GROWTH)).toMatchObject({
      chars: "20.4%",
      same: true,
    });
    expect(numberBefore("Revenue grew 20% ", GROWTH)).toMatchObject({ same: true });
    expect(numberBefore("Revenue grew 20.44% ", GROWTH)).toMatchObject({ same: true });
    expect(numberBefore("Revenue was $251.6 million ", AMOUNT)).toMatchObject({ same: true });
    // A number with no unit of its own reads as an amount.
    expect(numberBefore("up 251.6 million ", AMOUNT)).toMatchObject({ same: true });
    expect(numberBefore("比上年增加30,457.36万元", INCREASE)).toMatchObject({ same: true });
    expect(numberBefore("约3.05亿元", INCREASE)).toMatchObject({ same: true });
  });

  test("isn't, with another value or unit: a difference of percents written as a percent is flagged", () => {
    expect(numberBefore("Revenue grew 26% ", GROWTH)).toMatchObject({ chars: "26%", same: false });
    expect(numberBefore("margin rose 2.3% ", POINTS)).toMatchObject({ chars: "2.3%", same: false });
    expect(numberBefore("up 3.1亿元", INCREASE)).toMatchObject({ same: false });
    expect(numberBefore("yes ", figure(true, "yes"))).toBeNull();
  });

  test("is none when the text before the marker doesn't end with a number", () => {
    expect(numberBefore("Revenue grew ", GROWTH)).toBeNull();
    expect(numberBefore("Revenue grew in 2025 ", GROWTH)).toBeNull();
    expect(numberBefore("grew 25%[^1] ", GROWTH)).toBeNull();
    expect(numberBefore("", GROWTH)).toBeNull();
  });
});

const textOf = (nodes: readonly NodeJSON[]): string =>
  nodes
    .map((node) => (node.type === "text" ? (node.text ?? "") : textOf(node.content ?? [])))
    .join("");

describe("figure markers in an Answer", () => {
  const figures = new Map([
    [1, GROWTH],
    [2, POINTS],
  ]);
  const figureOf = (marker: number) => figures.get(marker) ?? null;

  test("become the figure's shown text in Blocks, in text and a table's cell, with the marks around them", () => {
    const blocks = withFigures(
      markdownToBlocks("**Revenue grew [=1]**, margin rose [=2].\n\n| A |\n|:--|\n| [=1] |"),
      figureOf,
      (each) => each.shown,
    );
    expect(blocks[0]?.content).toMatchObject([
      { type: "text", text: "Revenue grew 20.4%", marks: [{ type: "bold" }] },
      { type: "text", text: ", margin rose 2.3 percentage points." },
    ]);
    expect(textOf(blocks.slice(1))).toBe("A20.4%");
  });

  test("leave code alone, drop one with no figure with the space before it, and say what the model wrote when it differs", () => {
    const written: (string | null)[] = [];
    const blocks = withFigures(
      markdownToBlocks("Grew 26% [=1], fell [=9]. Write `[=1]`.\n\n```\n[=2]\n```"),
      figureOf,
      (each, _marker, wrote) => {
        written.push(wrote);
        return each.shown;
      },
    );
    expect(textOf(blocks)).toBe("Grew 26% 20.4%, fell. Write [=1].[=2]");
    expect(written).toEqual(["26%"]);
  });

  test("don't show the words around the number twice when the model writes them beside the marker", () => {
    const more = new Map<number, PlacedFigure>([
      [1, POINTS],
      [2, INCREASE],
      [3, AMOUNT],
      [4, figure(1.2e9, "about 1.2 billion")],
      [5, GROWTH],
    ]);
    expect(
      withFigureText(
        "rose [=1] percentage points; 增加[=2]万元; was $[=3] in 2025; about [=4]; grew [=5]% while roundabout [=4] millionaires",
        (marker) => more.get(marker) ?? null,
      ),
    ).toBe(
      "rose 2.3 percentage points; 增加30,457.36万元; was $251.6 million in 2025; about 1.2 billion; grew 20.4% while roundabout about 1.2 billion millionaires",
    );
  });

  test("read the same as text, for Question context, search and outside agents", () => {
    expect(
      withFigureText(
        "Revenue grew 20.4% [=1] and margin rose [=2] [=7].\n\nThen 26% [=1].",
        figureOf,
      ),
    ).toBe("Revenue grew 20.4% and margin rose 2.3 percentage points.\n\nThen 26% 20.4%.");
  });
});

describe("a formula as a person reads it", () => {
  test("names become labels or numbers as written, and operators the signs people write", () => {
    const values: Record<string, string> = {
      rev2025: "$1,482.6 million",
      rev2024: "$1,231.0 million",
    };
    expect(formulaText("rev2025 / rev2024 - 1", (name) => values[name] ?? null)).toBe(
      "$1,482.6 million ÷ $1,231.0 million − 1",
    );
    expect(formulaText("=(end/start)^(1/4)-1", () => null)).toBe("(end ÷ start) ^ (1 ÷ 4) − 1");
    expect(formulaText("SUM(q1,q2)*-x", () => null)).toBe("SUM(q1, q2) × −x");
    expect(formulaText("ROUND(a*5%,2)", () => null)).toBe("ROUND(a × 5%, 2)");
    expect(formulaText("a ++ ) ?", () => null)).toBe("a ++ ) ?");
  });
});
