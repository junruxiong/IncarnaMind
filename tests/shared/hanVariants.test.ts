/**
 * The table the Citation check folds traditional Chinese characters into
 * simplified ones with (src/shared/hanVariants.ts): generated from OpenCC's
 * TSCharacters.txt in the opencc-data package, characters only.
 */
import { describe, expect, test } from "vitest";
import { hanVariantPairs, tsCharacters } from "../../scripts/hanVariants";
import { SIMPLIFIED } from "../../src/shared/hanVariants";

describe("The traditional-to-simplified table", () => {
  test("is what scripts/hanVariants.ts generates from the installed opencc-data", () => {
    const pairs = hanVariantPairs(tsCharacters().text);
    expect(new Map(pairs.map((pair) => [pair[0], pair[1]]))).toEqual(SIMPLIFIED);
  });

  test("maps one character to one, each a single UTF-16 code unit, and folding twice is folding once", () => {
    for (const [from, to] of SIMPLIFIED) {
      expect(from).toHaveLength(1);
      expect(to).toHaveLength(1);
      expect(SIMPLIFIED.has(to)).toBe(false);
    }
  });

  test("folds the evaluation's traditional characters, and leaves simplified ones and other scripts alone", () => {
    const fold = (text: string) => [...text].map((char) => SIMPLIFIED.get(char) ?? char).join("");
    expect(fold("讓全球因為交通事故而傷亡的人數減少一半")).toBe(
      "让全球因为交通事故而伤亡的人数减少一半",
    );
    expect(fold("直線搜索可能會產生一些問題")).toBe("直线搜索可能会产生一些问题");
    // One simplified character for two traditional ones: both fold to it.
    expect(fold("發髮")).toBe("发发");
    expect(fold("优化过程 ABC 123 日本語")).toBe("优化过程 ABC 123 日本语");
  });
});
