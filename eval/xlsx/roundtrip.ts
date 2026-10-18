/**
 * The Excel editing spike's measurements on a folder of workbooks: open,
 * recalculate everything against the file's stored results, save with no
 * edit, and save after one edit, writing the copies and what the calculator
 * worked out, for the judges (judge.py, and LibreOffice in CI) to compare.
 *
 *   node --import ./scripts/tsResolve.mjs eval/xlsx/roundtrip.ts <in folder> <out folder>
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { deflateRawSync } from "node:zlib";
import { readWorkbook } from "../../src/core/documents/formats/xlsx";
import { isError, type Scalar } from "../../src/core/documents/formats/xlsxBook";
import { Calculator } from "../../src/core/documents/formats/xlsxCalc";
import { writeCopy } from "../../src/core/documents/formats/xlsxCopy";
import { columnName } from "../../src/shared/units";

const [input, output] = process.argv.slice(2) as [string, string];
const deflate = (data: Uint8Array) => new Uint8Array(deflateRawSync(data));
for (const folder of ["noedit", "edit"]) mkdirSync(join(output, folder), { recursive: true });

const close = (a: Scalar, b: Scalar) => {
  if (typeof a === "number" && typeof b === "number") {
    return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
  }
  if (isError(a) || isError(b)) return isError(a) && isError(b) && a.error === b.error;
  if ((a === null || a === "") && (b === null || b === "")) return true;
  return a === b;
};
const json = (value: Scalar) => (isError(value) ? { error: value.error } : value);
const now = () => performance.now();

const results: unknown[] = [];
for (const name of readdirSync(input)
  .filter((each) => each.endsWith(".xlsx"))
  .sort()) {
  const bytes = new Uint8Array(readFileSync(join(input, name)));
  const result: Record<string, unknown> = { file: name, bytes: bytes.length };
  try {
    let start = now();
    await readWorkbook(bytes, undefined, { shortDate: "yyyy-mm-dd" });
    result.previewMs = Math.round(now() - start);
    start = now();
    const calc = await Calculator.open(bytes);
    result.calcOpenMs = Math.round(now() - start);
    let formulas = 0;
    const cached = new Map<string, Scalar>();
    for (const sheet of calc.book.sheets) {
      for (const cell of sheet.cells.values()) {
        if (!cell.formula) continue;
        formulas++;
        cached.set(`${sheet.name}!${columnName(cell.column)}${cell.row}`, cell.value);
      }
    }
    result.formulas = formulas;
    const reasons: Record<string, number> = {};
    for (const reason of calc.cannot.values()) {
      const kind = reason.replace(/^function (\S+)$/, "function $1");
      reasons[kind] = (reasons[kind] ?? 0) + 1;
    }
    result.cannot = calc.cannot.size;
    result.cannotReasons = reasons;

    // No edit: the copy as the editor saves it untouched.
    start = now();
    writeFileSync(join(output, "noedit", name), await writeCopy(bytes, [], deflate));
    result.saveMs = Math.round(now() - start);

    // Every formula worked out again, against what the file stored.
    start = now();
    const all = await Calculator.open(bytes);
    const volatile = all.volatileCells();
    const recalc = all.recalculateAll();
    result.recalcAllMs = Math.round(now() - start);
    let compared = 0;
    let matched = 0;
    const mismatches: unknown[] = [];
    for (const sheet of all.book.sheets) {
      for (const cell of sheet.cells.values()) {
        if (!cell.formula) continue;
        const ref = `${sheet.name}!${columnName(cell.column)}${cell.row}`;
        const before = cached.get(ref) ?? null;
        if (before === null || volatile.has(ref)) continue;
        if (
          recalc.stale.some(
            (each) =>
              each.sheet === sheet.name && each.row === cell.row && each.column === cell.column,
          )
        )
          continue;
        compared++;
        if (close(before, cell.value)) matched++;
        else if (mismatches.length < 8)
          mismatches.push({
            ref,
            formula: cell.formula.text,
            stored: json(before),
            worked: json(cell.value),
          });
      }
    }
    result.fidelity = {
      compared,
      matched,
      stale: recalc.stale.length,
      volatile: volatile.size,
      mismatches,
    };

    // One edit: the number with the most formulas depending on it, among the first candidates.
    const edit = await Calculator.open(bytes);
    let best: { sheet: string; row: number; column: number; value: number; count: number } | null =
      null;
    let tried = 0;
    for (const sheet of edit.book.sheets) {
      for (const cell of sheet.cells.values()) {
        if (cell.formula || typeof cell.value !== "number" || cell.arrayPart !== undefined)
          continue;
        if (tried++ > 400) break;
        const count = edit.dependentCount(sheet.name, cell.row, cell.column);
        if (count > 0 && (!best || count > best.count)) {
          best = {
            sheet: sheet.name,
            row: cell.row,
            column: cell.column,
            value: cell.value,
            count,
          };
        }
      }
    }
    if (best) {
      const ref = `${columnName(best.column)}${best.row}`;
      const typed = best.value === 0 ? 5 : Math.round((best.value * 1.1 + 1) * 10_000) / 10_000;
      const dependents = edit.dependentRefs(best.sheet, best.row, best.column);
      start = now();
      const changed = edit.edit(best.sheet, ref, String(typed));
      result.editMs = Math.round(now() - start);
      const copy = await writeCopy(bytes, edit.writes(), deflate);
      writeFileSync(join(output, "edit", name), copy);
      const expected: Record<string, unknown> = {};
      for (const each of dependents) {
        const sheet = edit.sheetNamed(each.sheet);
        const cell = sheet?.cells.get(each.row * 16_384 + each.column);
        if (!cell?.formula) continue;
        expected[`${each.sheet}!${columnName(each.column)}${each.row}`] = json(cell.value);
      }
      const stale = changed.stale.map(
        (each) => `${each.sheet}!${columnName(each.column)}${each.row}`,
      );
      writeFileSync(
        join(output, "edit", `${basename(name, ".xlsx")}.json`),
        JSON.stringify({
          edit: { sheet: best.sheet, ref, from: best.value, to: typed },
          expected,
          stale,
        }),
      );
      // The indexer reads the copy, as it will when the copy is added as a Document.
      await readWorkbook(copy);
      result.edit = {
        sheet: best.sheet,
        ref,
        dependents: Object.keys(expected).length,
        changed: changed.changed.length,
        stale: stale.length,
      };
    }
  } catch (error) {
    result.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  }
  results.push(result);
  console.log(JSON.stringify(result));
}
writeFileSync(join(output, "ours.json"), JSON.stringify(results, null, 1));
