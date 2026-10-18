/**
 * Values in the workbook calculator (spike: excel-edit): ranges, arrays and
 * cells' values, and Excel's rules for turning one kind into another. Pure.
 */
import type { BookSheet, CalcError, Scalar } from "./xlsxBook";
import { isError } from "./xlsxBook";
import type { ErrorCode } from "./xlsxFormula";

export interface RangeValue {
  kind: "range";
  sheet: BookSheet;
  top: number;
  left: number;
  bottom: number;
  right: number;
}

export interface ArrayValue {
  kind: "array";
  rows: Scalar[][];
}

export type Value = Scalar | RangeValue | ArrayValue;

export const isRange = (value: Value): value is RangeValue =>
  typeof value === "object" && value !== null && "kind" in value && value.kind === "range";
export const isArray = (value: Value): value is ArrayValue =>
  typeof value === "object" && value !== null && "kind" in value && value.kind === "array";

export const err = (code: ErrorCode): CalcError => ({ error: code });

/** Thrown while evaluating what the calculator doesn't do; the cell keeps its stored value. */
export class CannotCalculate extends Error {}

export function sameValue(a: Scalar, b: Scalar): boolean {
  if (isError(a) || isError(b)) return isError(a) && isError(b) && a.error === b.error;
  return a === b;
}

/** What the User typed, as a value: a number ("1,234.5", "12%", "-3"), TRUE or FALSE, else text. */
export function typedValue(input: string): Scalar {
  const trimmed = input.trim();
  if (trimmed === "") return null;
  if (/^(true|false)$/i.test(trimmed)) return trimmed.toUpperCase() === "TRUE";
  const number = numberFromText(trimmed);
  return number === null ? input : number;
}

/** Text as a number the way Excel reads one typed or in a formula: "1,234", "5%", "(3)", "$4"; null if it isn't one. */
export function numberFromText(text: string): number | null {
  let body = text.trim();
  if (body === "") return null;
  let sign = 1;
  if (/^\(.*\)$/.test(body)) {
    sign = -1;
    body = body.slice(1, -1).trim();
  }
  if (body.startsWith("-")) {
    sign = -sign;
    body = body.slice(1).trim();
  } else if (body.startsWith("+")) body = body.slice(1).trim();
  body = body.replace(/^[$£€¥]/, "");
  let scale = 1;
  if (body.endsWith("%")) {
    scale = 0.01;
    body = body.slice(0, -1).trim();
  }
  if (!/^(\d{1,3}(,\d{3})+|\d+)?(\.\d*)?([eE][+-]?\d+)?$/.test(body) || !/\d/.test(body))
    return null;
  return sign * Number(body.replace(/,/g, "")) * scale;
}

export function toNumber(value: Scalar): number | CalcError {
  if (typeof value === "number") return value;
  if (value === null) return 0;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (isError(value)) return value;
  const number = numberFromText(value);
  if (number !== null) return number;
  const date = dateFromText(value);
  return date ?? err("#VALUE!");
}

/** "2026-01-31" (and "2026-01-31 12:00") as a 1900-system serial; null if it isn't one. */
function dateFromText(text: string): number | null {
  const match = /^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$/.exec(
    text.trim(),
  );
  if (!match) return null;
  const days =
    (Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])) - Date.UTC(1899, 11, 30)) /
    86_400_000;
  const time =
    (Number(match[4] ?? 0) * 3600 + Number(match[5] ?? 0) * 60 + Number(match[6] ?? 0)) / 86_400;
  return days + time;
}

/** A number as text the way Excel joins it with "&": up to 15 significant digits. */
export function numberText(value: number): string {
  if (value === 0) return "0";
  const rounded = Number(value.toPrecision(15));
  const abs = Math.abs(rounded);
  if (abs >= 1e21 || abs < 1e-9) {
    return rounded
      .toExponential()
      .replace(/\.?0+e/, "e")
      .replace("e", "E")
      .replace(/E\+?(-?)(\d)$/, "E+$1$2")
      .replace("E+-", "E-");
  }
  return String(rounded);
}

export function toText(value: Scalar): string | CalcError {
  if (value === null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
  if (typeof value === "number") return numberText(value);
  return value;
}

/** Excel's ordering of mixed values: numbers, then text (no case), then logical values. */
export function compareValues(a: Scalar, b: Scalar): number {
  const rank = (value: Scalar) =>
    typeof value === "number"
      ? 0
      : typeof value === "string"
        ? 1
        : typeof value === "boolean"
          ? 2
          : 3;
  let x = a;
  let y = b;
  if (x === null) x = typeof y === "string" ? "" : typeof y === "boolean" ? false : 0;
  if (y === null) y = typeof x === "string" ? "" : typeof x === "boolean" ? false : 0;
  const rx = rank(x);
  const ry = rank(y);
  if (rx !== ry) return rx - ry;
  if (typeof x === "number" && typeof y === "number") return x === y ? 0 : x < y ? -1 : 1;
  if (typeof x === "string" && typeof y === "string") {
    const p = x.toLowerCase();
    const q = y.toLowerCase();
    return p === q ? 0 : p < q ? -1 : 1;
  }
  if (typeof x === "boolean" && typeof y === "boolean") return Number(x) - Number(y);
  return 0;
}

export function scalarOp(op: string, a: Scalar, b: Scalar): Scalar {
  if (isError(a)) return a;
  if (isError(b)) return b;
  switch (op) {
    case "&": {
      const x = toText(a);
      const y = toText(b);
      if (isError(x)) return x;
      if (isError(y)) return y;
      return x + y;
    }
    case "=":
      return compareValues(a, b) === 0;
    case "<>":
      return compareValues(a, b) !== 0;
    case "<":
      return compareValues(a, b) < 0;
    case ">":
      return compareValues(a, b) > 0;
    case "<=":
      return compareValues(a, b) <= 0;
    case ">=":
      return compareValues(a, b) >= 0;
  }
  const x = toNumber(a);
  const y = toNumber(b);
  if (isError(x)) return x;
  if (isError(y)) return y;
  switch (op) {
    case "+":
      return x + y;
    case "-":
      return x - y;
    case "*":
      return x * y;
    case "/":
      return y === 0 ? err("#DIV/0!") : x / y;
    case "^": {
      if (x === 0 && y === 0) return err("#NUM!");
      const result = x ** y;
      return Number.isFinite(result) ? result : err("#NUM!");
    }
  }
  return err("#VALUE!");
}
