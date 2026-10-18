import { describe, expect, test } from "vitest";
import { Calculator } from "../../src/core/documents/formats/xlsxCalc";
import { xlsxPackage } from "../helpers/office";

/*
 * The workbook calculator (spike: excel-edit) on small workbooks written as XML, with the stored
 * results the generators leave and the ones Excel does.
 */

const number = (ref: string, value: number) => `<c r="${ref}"><v>${value}</v></c>`;
const formula = (ref: string, text: string, stored: number | null, attrs = "") =>
  `<c r="${ref}"><f${attrs}>${text}</f>${stored === null ? "" : `<v>${stored}</v>`}</c>`;
const follower = (ref: string, si: number, stored: number | null) =>
  `<c r="${ref}"><f t="shared" si="${si}"/>${stored === null ? "" : `<v>${stored}</v>`}</c>`;

/** What a ledger stores for its formulas: D2:D4, E2:E4 and G1; null stores none. */
interface Stored {
  d: [number | null, number | null, number | null];
  e: [number | null, number | null, number | null];
  total: number | null;
}

const PLACEHOLDERS: Stored = { d: [0, 0, 0], e: [0, 0, 0], total: 0 };
const NONE: Stored = { d: [null, null, null], e: [null, null, null], total: null };
/** What Excel stores: the results, but for D3, which an edit elsewhere left behind. */
const EXCELS: Stored = { d: [6, 999, 14], e: [6, 16, 30], total: 30 };

/**
 * A ledger: quantity (B) times a price kept in C1 (absolute) as a shared formula in D, a running
 * total in E, and the total in G1.
 */
function ledger(stored: Stored, calcPr?: string): Uint8Array {
  const [d2, d3, d4] = stored.d;
  const [e2, e3, e4] = stored.e;
  const rows = [
    `<row r="1">${number("C1", 2)}${formula("G1", "SUM(D2:D4)", stored.total)}</row>`,
    `<row r="2">${number("B2", 3)}${formula("D2", "B2*$C$1", d2, ' t="shared" ref="D2:D4" si="0"')}${formula("E2", "D2", e2)}</row>`,
    `<row r="3">${number("B3", 5)}${follower("D3", 0, d3)}${formula("E3", "E2+D3", e3)}</row>`,
    `<row r="4">${number("B4", 7)}${follower("D4", 0, d4)}${formula("E4", "E3+D4", e4)}</row>`,
  ];
  return new Uint8Array(
    xlsxPackage({
      sheets: [{ name: "Ledger", xml: `<sheetData>${rows.join("")}</sheetData>` }],
      ...(calcPr && { calcPr }),
    }),
  );
}

const D = 3;
function value(calc: Calculator, row: number, column: number) {
  const sheet = calc.sheetNamed("Ledger");
  if (!sheet) throw new Error("The workbook has no Ledger.");
  return calc.cellValue(sheet, row, column);
}
const written = (calc: Calculator) => calc.writes().map((write) => `${write.row}:${write.column}`);

describe("a shared formula", () => {
  test("is written once and read by every cell of its group, relative references moving and absolute ones staying", async () => {
    const calc = await Calculator.open(ledger(PLACEHOLDERS));

    expect(calc.formulaText("Ledger", 2, D)).toBe("=B2*$C$1");
    expect(calc.formulaText("Ledger", 3, D)).toBe("=B3*$C$1");
    expect(calc.formulaText("Ledger", 4, D)).toBe("=B4*$C$1");
    calc.recalculateAll();
    expect([2, 3, 4].map((row) => value(calc, row, D))).toEqual([6, 10, 14]);
    expect(value(calc, 1, 6)).toBe(30);
  });

  test("keeps its other cells' formulas when its first cell is changed", async () => {
    const calc = await Calculator.open(ledger(PLACEHOLDERS));

    calc.edit("Ledger", "D2", "=B2*3");

    expect(calc.formulaText("Ledger", 3, D)).toBe("=B3*$C$1");
    const writes = calc.writes();
    expect(writes.find((write) => write.row === 3 && write.column === D)?.formula).toBe("B3*$C$1");
  });
});

describe("an edit", () => {
  test.each([
    [
      "a file that asks Excel to recalculate on opening",
      ledger(EXCELS, '<calcPr calcId="124519" fullCalcOnLoad="1"/>'),
    ],
    ["a file whose stored results are all 0, as ExcelJS writes them", ledger(PLACEHOLDERS)],
    ["a file with no stored results, as openpyxl writes it", ledger(NONE)],
  ])("recalculates every formula first in %s", async (_name, bytes) => {
    const calc = await Calculator.open(bytes);
    expect(calc.book.placeholders).toBe(true);

    calc.edit("Ledger", "B2", "4");

    // D3 and D4 don't depend on B2, but the total and the running totals read them.
    expect([2, 3, 4].map((row) => value(calc, row, D))).toEqual([8, 10, 14]);
    expect(value(calc, 1, 6)).toBe(32);
    expect([2, 3, 4].map((row) => value(calc, row, 4))).toEqual([8, 18, 32]);
    expect(written(calc)).toContain("3:3");
  });

  test("recalculates only what depends on it in a file that keeps real results, which are trusted", async () => {
    const calc = await Calculator.open(ledger(EXCELS));
    expect(calc.book.placeholders).toBe(false);

    calc.edit("Ledger", "B2", "4");

    expect(value(calc, 2, D)).toBe(8);
    // D3 keeps what the file stored for it, as Excel recalculates it on opening the copy.
    expect(value(calc, 3, D)).toBe(999);
    expect(written(calc)).not.toContain("3:3");
  });
});
