import { readFileSync } from "node:fs";
import { isError } from "../../src/core/documents/formats/xlsxBook";
import { Calculator } from "../../src/core/documents/formats/xlsxCalc";
import { columnName } from "../../src/shared/units";

const calc = await Calculator.open(new Uint8Array(readFileSync(process.argv[2] as string)));
const before = new Map<string, unknown>();
for (const sheet of calc.book.sheets)
  for (const cell of sheet.cells.values())
    if (cell.formula) before.set(`${sheet.name}!${columnName(cell.column)}${cell.row}`, cell.value);
calc.recalculateAll();
let shown = 0;
for (const sheet of calc.book.sheets)
  for (const cell of sheet.cells.values()) {
    if (!cell.formula) continue;
    const ref = `${sheet.name}!${columnName(cell.column)}${cell.row}`;
    const a = before.get(ref) as never;
    const b = cell.value as never;
    const same =
      typeof a === "number" && typeof b === "number"
        ? Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a))
        : isError(a) && isError(b)
          ? (a as { error: string }).error === (b as { error: string }).error
          : a === b;
    if (!same && a !== null && shown++ < Number(process.argv[3] ?? 12))
      console.log(
        ref,
        cell.formula.text.slice(0, 90),
        "| stored",
        JSON.stringify(a),
        "| worked",
        JSON.stringify(b),
      );
  }
