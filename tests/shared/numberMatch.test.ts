/**
 * Quotes from spreadsheets (ADR-0011): number formatting is normalised in the
 * quote and the cells alike, while digits, decimals and signs must agree.
 * A block of rows is stored one line per row, its cells separated by tabs.
 */
import { describe, expect, test } from "vitest";
import { findQuote } from "../../src/shared/quoteMatch";

/** What a match covers in the cells' text, or null if the quote isn't found. */
const covered = (text: string, quote: string) => {
  const ranges = findQuote(text, quote, { numbers: true });
  return ranges ? ranges.map((range) => text.slice(range.start, range.end)).join(" … ") : null;
};

const ROW = "West\t4812\t-1250\t3.5\t£350,200";

describe("matching numbers in a sheet's rows", () => {
  test('"4,812" matches 4812, and so do "4 812" and "4812"', () => {
    expect(covered(ROW, "West 4,812")).toBe("West\t4812");
    expect(covered(ROW, "West 4 812")).toBe("West\t4812");
    expect(covered(ROW, "West 4812")).toBe("West\t4812");
    // And the other way round: a cell written with separators matches a plain quote.
    expect(covered("Total\t4,812\t1 250 000", "Total 4812 1250000")).toBe(
      "Total\t4,812\t1 250 000",
    );
  });

  test("currency symbols around a number don't count, but its digits do", () => {
    // The match takes in the cell's formatting, for the viewer to wash.
    expect(covered(ROW, "£350,200")).toBe("£350,200");
    expect(covered(ROW, "350200")).toBe("£350,200");
    expect(covered(ROW, "$350,200")).toBe("£350,200");
    expect(covered(ROW, "£350,201")).toBeNull();
  });

  test("decimals stay correct: trailing zeros don't count, other digits do", () => {
    expect(covered(ROW, "3.50")).toBe("3.5");
    expect(covered("Rate\t4812.5", "4,812.50")).toBe("4812.5");
    expect(covered("Rate\t4812.05", "4,812.5")).toBeNull();
    expect(covered("Rate\t4812", "4,812.00")).toBe("4812");
    // A number is matched whole: not inside a longer one, nor before its decimals.
    expect(covered("Rate\t4812.5", "4812")).toBeNull();
    expect(covered("Rate\t48125", "4812")).toBeNull();
    expect(covered("Rate\t14812", "4812")).toBeNull();
  });

  test("negatives stay correct: a sign, or an accounting negative, must agree", () => {
    expect(covered(ROW, "-1,250")).toBe("-1250");
    expect(covered(ROW, "(1,250)")).toBe("-1250");
    expect(covered(ROW, "1,250")).toBeNull();
    expect(covered(ROW, "4812 1250")).toBeNull();
    expect(covered("Change\t(1,250.0)", "-1250")).toBe("(1,250.0)");
  });

  test("cells are read in order: separate cells never join into one number", () => {
    const small = "Q1\t12\t345\t678";
    expect(covered(small, "12 345 678")).toBe("12\t345\t678");
    expect(covered(small, "345")).toBe("345");
    expect(covered(small, "12345678")).toBeNull();
    // A space after a number's decimals separates cells, never groups of digits.
    expect(covered("£350,200\t£389,600", "350,200.00 389 600")).toBe("£350,200\t£389,600");
    expect(covered("3.5\t389\t600", "3.5 389 600")).toBe("3.5\t389\t600");
  });

  test("numbers that differ aren't found, and nor are reworded cells", () => {
    expect(covered(ROW, "West 4,813")).toBeNull();
    expect(covered(ROW, "West 48,12")).toBeNull();
    expect(covered(ROW, "Western 4812")).toBeNull();
    expect(covered(ROW, "West revenue was 4,812")).toBeNull();
  });

  test("text that isn't a sheet's is matched as before: numbers there aren't normalised", () => {
    expect(findQuote("Total 4812 tonnes", "Total 4,812 tonnes")).toBeNull();
    expect(findQuote("Total 4812 tonnes", "Total 4,812 tonnes", { numbers: true })).not.toBeNull();
  });
});
