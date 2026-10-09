/**
 * Showing a cell's number in the sheet preview as Excel shows it (ADR-0011):
 * dates in their own pattern, times, scientific notation, fractions,
 * accounting formats with their fill, colours and conditions. What is
 * indexed (`formatNumber`) is unchanged: these are for the eye only.
 */
import { describe, expect, test } from "vitest";
import {
  displayNumber,
  displayText,
  shortDatePattern,
} from "../../src/core/documents/formats/numberDisplay";
import { formatNumber } from "../../src/core/documents/formats/numbers";

const show = (value: number, code: string | undefined, shortDate?: string) =>
  displayNumber(value, code, { shortDate }).text;

describe("numbers shown as Excel shows them", () => {
  test("number formats that indexing already knows look the same", () => {
    for (const [value, code] of [
      [350200, '"£"#,##0'],
      [-1250, "#,##0 ;(#,##0)"],
      [0.042, "0.0%"],
      [-0.0125, "0.0%"],
      [1234.5, "#,##0.00"],
      [1234.5, "General"],
    ] as const) {
      expect(show(value, code)).toBe(formatNumber(value, code));
    }
  });

  test("dates and times follow their own pattern, not ISO", () => {
    // 46114 is 2 April 2026, a Thursday.
    expect(show(46114, "d-mmm-yy")).toBe("2-Apr-26");
    expect(show(46114, "dd/mm/yyyy")).toBe("02/04/2026");
    expect(show(46114, "mmmm d, yyyy")).toBe("April 2, 2026");
    expect(show(46114, "dddd")).toBe("Thursday");
    expect(show(46114, "ddd d mmm")).toBe("Thu 2 Apr");
    expect(show(46114, "mmm-yy")).toBe("Apr-26");
    expect(show(46114, "yyyy-mm-dd")).toBe("2026-04-02");
    expect(show(0.75, "h:mm AM/PM")).toBe("6:00 PM");
    expect(show(0.75, "h:mm")).toBe("18:00");
    expect(show(46114.5, "yyyy-mm-dd h:mm:ss")).toBe("2026-04-02 12:00:00");
    expect(show(1.5, "[h]:mm")).toBe("36:00");
    expect(show(0.000_694_444, "mm:ss")).toBe("01:00");
    // m after h, or before s, is minutes.
    expect(show(0.5 + 5 / 1440, "h:m")).toBe("12:5");
  });

  test("the system short date (format 14) follows the User's locale", () => {
    expect(show(46114, "yyyy-mm-dd", "m/d/yyyy")).toBe("2026-04-02");
    expect(displayNumber(46114, "yyyy-mm-dd", { shortDate: "dd/mm/yyyy", builtIn: 14 }).text).toBe(
      "02/04/2026",
    );
    expect(shortDatePattern("en-US")).toBe("m/d/yyyy");
    expect(shortDatePattern("en-GB")).toBe("dd/mm/yyyy");
    expect(shortDatePattern("zh-CN")).toBe("yyyy/m/d");
  });

  test("scientific notation and fractions", () => {
    expect(show(12345, "0.00E+00")).toBe("1.23E+04");
    expect(show(0.000123, "0.0E+00")).toBe("1.2E-04");
    expect(show(1.5, "# ?/?")).toBe("1 1/2");
    expect(show(0.75, "# ??/??")).toBe("  3/4 ");
    expect(show(2.125, "# ?/8")).toBe("2 1/8");
  });

  test("General fits a standard column: at most 11 characters", () => {
    expect(show(1 / 3, "General")).toBe("0.333333333");
    expect(show(0.1 + 0.2, undefined)).toBe("0.3");
    expect(show(123456789012, "General")).toBe("1.23457E+11");
    expect(show(-1 / 3, "General")).toBe("-0.33333333");
    expect(displayNumber(1 / 3, "General", { generalWidth: 6 }).text).toBe("0.3333");
  });

  test("digits fill placeholders between literals, as in codes and phone numbers", () => {
    expect(show(123456789, "000-00-0000")).toBe("123-45-6789");
    expect(show(5551234567, "(###) ###-####")).toBe("(555) 123-4567");
    expect(show(7, "000")).toBe("007");
    expect(show(1500000, '#,##0.0,,"M"')).toBe("1.5M");
  });

  test("colours and conditions choose how a value looks", () => {
    expect(displayNumber(-1250, "#,##0;[Red]-#,##0", {})).toEqual({
      text: "-1,250",
      color: "#FF0000",
    });
    expect(displayNumber(1250, "[Blue]#,##0;[Red]-#,##0", {}).color).toBe("#0000FF");
    const scaled = '[>=1000000]0.0,,"M";[>=1000]0.0,"K";0';
    expect(show(1_500_000, scaled)).toBe("1.5M");
    expect(show(2500, scaled)).toBe("2.5K");
    expect(show(12, scaled)).toBe("12");
  });

  test("accounting formats push the currency left and the number right", () => {
    const accounting = '_("$"* #,##0.00_);_("$"* \\(#,##0.00\\);_("$"* "-"??_);_(@_)';
    const positive = displayNumber(1234.5, accounting, {});
    expect(positive.text).toBe(" $1,234.50 ");
    expect(positive.fill).toBe(2);
    expect(displayNumber(-1234.5, accounting, {}).text).toBe(" $(1,234.50)");
    expect(displayNumber(0, accounting, {}).text).toBe(" $-   ");
    expect(displayText("Total", accounting).text).toBe(" Total ");
  });

  test("text cells take a format's text section, and keep their text otherwise", () => {
    expect(displayText("abc", '0;-0;0;"Code: "@')).toEqual({ text: "Code: abc" });
    expect(displayText("abc", "0.00")).toEqual({ text: "abc" });
    expect(displayText("abc", undefined)).toEqual({ text: "abc" });
  });
});
