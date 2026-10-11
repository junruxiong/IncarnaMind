import { describe, expect, test } from "vitest";
import { cleanShortTitle, cutTitle, mindName, NAME_LENGTH } from "../../src/shared/naming";

describe("cutting a title", () => {
  test("a short line is kept, without trailing punctuation", () => {
    expect(cutTitle("How much cooler are parks at night?")).toBe(
      "How much cooler are parks at night",
    );
    expect(cutTitle("公园夜间降温多少？")).toBe("公园夜间降温多少");
  });

  test("only the first line counts", () => {
    expect(cutTitle("\n\n  Park cooling  \nsecond line")).toBe("Park cooling");
  });

  test("a long English line is cut at a word boundary", () => {
    const question =
      "How much cooler are large irrigated parks than the surrounding built-up streets during clear windless summer nights?";
    const title = cutTitle(question);
    expect(title.length).toBeLessThanOrEqual(NAME_LENGTH);
    expect(title.length).toBeGreaterThan(NAME_LENGTH - 15);
    expect(question.startsWith(`${title} `)).toBe(true);
  });

  test("a long Chinese line is cut at a character, with none broken", () => {
    const question =
      "在无风晴朗的夏季夜晚，大型灌溉公园比周围建成区的街道最多能低多少摄氏度，这个差距在白天是否更小？".repeat(
        2,
      );
    const title = cutTitle(question);
    expect(Array.from(title).length).toBeLessThanOrEqual(NAME_LENGTH);
    expect(Array.from(title).length).toBeGreaterThan(NAME_LENGTH - 3);
    expect(question.startsWith(title)).toBe(true);
  });

  test("a character outside the basic plane is never split", () => {
    const title = cutTitle("𠮷".repeat(100));
    expect(title).toBe("𠮷".repeat(NAME_LENGTH));
  });

  test("a word longer than the limit is cut inside it", () => {
    expect(cutTitle("x".repeat(100))).toBe("x".repeat(NAME_LENGTH));
  });

  test("a model's short title loses its quotes and full stop", () => {
    expect(cleanShortTitle('"Park cooling at night."\n')).toBe("Park cooling at night");
    expect(cleanShortTitle("「公园夜间降温」")).toBe("公园夜间降温");
    expect(cleanShortTitle("  ")).toBe("");
  });
});

describe("the name of a Mind", () => {
  test("a title the User typed wins over the first line", () => {
    expect(mindName({ title: "Reading list", autoTitle: "Notes on tides" })).toBe("Reading list");
    expect(mindName({ title: "", autoTitle: "Notes on tides" })).toBe("Notes on tides");
    expect(mindName({ title: "", autoTitle: "" })).toBe("");
  });
});
