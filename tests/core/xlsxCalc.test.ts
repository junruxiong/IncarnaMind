import { describe, expect, test } from "vitest";
import { Calculator } from "../../src/core/documents/formats/xlsxCalc";
import { xlsxPackage } from "../helpers/office";

/*
 * The workbook calculator (spike: excel-edit) on small workbooks written as XML, with the stored
 * results the generators leave and the ones Excel does.
 */

const number = (ref: string, value: number) => `<c r="${ref}"><v>${value}</v></c>`;
const formula = (ref: string, text: string, stored: number | string, attrs = "") =>
  `<c r="${ref}"><f${attrs}>${text}</f><v>${stored}</v></c>`;
const follower = (ref: string, si: number, stored: number) =>
  `<c r="${ref}"><f t="shared" si="${si}"/><v>${stored}</v></c>`;

/**
 * A ledger: quantity (B) times a price kept in C1 (absolute) as a shared formula in D, a running
 * total in E, and the total in G1, with `stored` as every formula's stored result.
 */
function ledger(options: { stored: number; calcPr?: string }): Uint8Array {
  const { stored } = options;
  const rows = [
    `<row r="1">${number("C1", 2)}${formula("G1", "SUM(D2:D4)", stored)}</row>`,
    `<row r="2">${number("B2", 3)}${formula("D2", "B2*$C$1", stored, ' t="shared" ref="D2:D4" si="0"')}${formula("E2", "D2", stored)}</row>`,
    `<row r="3">${number("B3", 5)}${follower("D3", 0, stored)}${formula("E3", "E2+D3", stored)}</row>`,
    `<row r="4">${number("B4", 7)}${follower("D4", 0, stored)}${formula("E4", "E3+D4", stored)}</row>`,
  ];
  return new Uint8Array(
    xlsxPackage({
      sheets: [{ name: "Ledger", xml: `<sheetData>${rows.join("")}</sheetData>` }],
      ...(options.calcPr && { calcPr: options.calcPr }),
    }),
  );
}

const D = 3;
function value(calc: Calculator, row: number, column: number) {
  const sheet = calc.sheetNamed("Ledger");
  if (!sheet) throw new Error("The workbook has no Ledger.");
  return calc.cellValue(sheet, row, column);
}

describe("a shared formula", () => {
  test("is written once and read by every cell of its group, relative references moving and absolute ones staying", async () => {
    const calc = await Calculator.open(ledger({ stored: 0 }));

    expect(calc.formulaText("Ledger", 2, D)).toBe("=B2*$C$1");
    expect(calc.formulaText("Ledger", 3, D)).toBe("=B3*$C$1");
    expect(calc.formulaText("Ledger", 4, D)).toBe("=B4*$C$1");
    calc.recalculateAll();
    expect([2, 3, 4].map((row) => value(calc, row, D))).toEqual([6, 10, 14]);
    expect(value(calc, 1, 6)).toBe(30);
  });

  test("keeps its other cells' formulas when its first cell is changed", async () => {
    const calc = await Calculator.open(ledger({ stored: 0 }));

    calc.edit("Ledger", "D2", "=B2*3");

    expect(calc.formulaText("Ledger", 3, D)).toBe("=B3*$C$1");
    const writes = calc.writes();
    expect(writes.find((write) => write.row === 3 && write.column === D)?.formula).toBe("B3*$C$1");
  });
});

describe("an edit", () => {
  test("recalculates every formula first in a file that asks Excel to recalculate on opening, whose stored results are placeholders", async () => {
    const calc = await Calculator.open(
      ledger({ stored: 0, calcPr: '<calcPr calcId="124519" fullCalcOnLoad="1"/>' }),
    );
    expect(calc.book.fullCalcOnLoad).toBe(true);

    calc.edit("Ledger", "B2", "4");

    // D3 and D4 don't depend on B2, but the total and the running totals read them.
    expect([2, 3, 4].map((row) => value(calc, row, D))).toEqual([8, 10, 14]);
    expect(value(calc, 1, 6)).toBe(32);
    expect([2, 3, 4].map((row) => value(calc, row, 4))).toEqual([8, 18, 32]);
    const written = calc.writes().map((write) => `${write.row}:${write.column}`);
    expect(written).toContain("3:3");
  });

  test("recalculates only what depends on it in a file that keeps its results, which are trusted", async () => {
    const calc = await Calculator.open(ledger({ stored: 0 }));
    expect(calc.book.fullCalcOnLoad).toBe(false);

    calc.edit("Ledger", "B2", "4");

    expect(value(calc, 2, D)).toBe(8);
    // D3 and D4 keep what the file stored for them.
    expect([3, 4].map((row) => value(calc, row, D))).toEqual([0, 0]);
    expect(calc.writes().map((write) => `${write.row}:${write.column}`)).not.toContain("3:3");
  });
});
