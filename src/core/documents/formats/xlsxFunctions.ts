/**
 * Excel's worksheet functions for the workbook calculator (./xlsxCalc), the
 * ones found in the spike's corpus first. Each takes its arguments unevaluated,
 * so IF, IFERROR and CHOOSE evaluate only what they need, as Excel does.
 * A function not here makes its cell one the calculator can't recalculate.
 */
import { displayNumber } from "./numberDisplay";
import { type CalcError, isError, type Scalar } from "./xlsxBook";
import type { Context } from "./xlsxCalc";
import type { Node } from "./xlsxFormula";
import {
  type ArrayValue,
  CannotCalculate,
  compareValues,
  err,
  isArray,
  isRange,
  numberFromText,
  numberText,
  type RangeValue,
  toNumber,
  toText,
  type Value,
} from "./xlsxValues";

type Fn = (args: Node[], context: Context) => Value;

/** An error that ends a function's evaluation: what the function returns. */
class Fail extends Error {
  constructor(readonly value: CalcError) {
    super(value.error);
  }
}

const fail = (code: CalcError["error"]): never => {
  throw new Fail(err(code));
};

export const FUNCTIONS = new Map<string, Fn>();
/** Functions whose result can change with nothing they read changing: always recalculated. */
export const VOLATILE = new Set(["NOW", "TODAY", "RAND", "RANDBETWEEN", "OFFSET", "INDIRECT"]);

function define(names: string | string[], fn: Fn, arity?: [number, number]) {
  for (const name of Array.isArray(names) ? names : [names]) {
    FUNCTIONS.set(name, (args, context) => {
      if (arity && (args.length < arity[0] || args.length > arity[1])) return err("#VALUE!");
      try {
        return fn(args, context);
      } catch (error) {
        if (error instanceof Fail) return error.value;
        throw error;
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Arguments

const value = (node: Node, context: Context): Value => context.calc.evaluate(node, context);
const arrayValue = (node: Node, context: Context): Value =>
  context.calc.evaluate(node, { ...context, array: true });

function scalar(node: Node | undefined, context: Context): Scalar {
  if (!node || node.kind === "missing") return null;
  return context.calc.toScalar(value(node, context), context);
}

function check<T>(result: T | CalcError): T {
  if (isError(result)) throw new Fail(result);
  return result;
}

function num(node: Node | undefined, context: Context, fallback?: number): number {
  if (!node || node.kind === "missing") {
    if (fallback === undefined) return 0;
    return fallback;
  }
  return check(toNumber(check(scalar(node, context))));
}

function text(node: Node | undefined, context: Context): string {
  return check(toText(check(scalar(node, context))));
}

function bool(node: Node | undefined, context: Context, fallback: boolean): boolean {
  if (!node || node.kind === "missing") return fallback;
  const each = check(scalar(node, context));
  if (typeof each === "boolean") return each;
  if (typeof each === "number") return each !== 0;
  if (each === null) return false;
  const upper = each.toUpperCase();
  if (upper === "TRUE") return true;
  if (upper === "FALSE") return false;
  return fail("#VALUE!");
}

const integer = (node: Node | undefined, context: Context, fallback?: number) =>
  Math.trunc(num(node, context, fallback));

/**
 * The numbers of the arguments, as SUM reads them: numbers in ranges and
 * arrays, typed numbers, numeric text and logical values; text, logical
 * values and blanks in ranges skipped; any error returned.
 */
function numbers(
  args: Node[],
  context: Context,
  options: { logicalInRanges?: boolean; textInRanges?: boolean } = {},
): number[] {
  const found: number[] = [];
  for (const arg of args) {
    if (arg.kind === "missing") continue;
    const each = value(arg, context);
    if (isRange(each) || isArray(each)) {
      context.calc.eachValue(each, (cell) => {
        if (isError(cell)) throw new Fail(cell);
        if (typeof cell === "number") found.push(cell);
        else if (typeof cell === "boolean" && options.logicalInRanges) found.push(cell ? 1 : 0);
        else if (typeof cell === "string" && options.textInRanges) found.push(0);
      });
    } else {
      const single = check(each);
      if (single === null) continue;
      found.push(check(toNumber(single)));
    }
  }
  return found;
}

/** A range's or array's values as rows. */
const rowsOf = (each: Value, context: Context) => context.calc.toRows(each);

/** A 1-D range or array as a list, and whether it ran across. */
function listOf(each: Value, context: Context): Scalar[] {
  if (isRange(each) && each.left === each.right) return lineOf(each, 0, false, context);
  if (isRange(each) && each.top === each.bottom) return lineOf(each, 0, true, context);
  const rows = rowsOf(each, context);
  if (rows.length === 1) return rows[0] ?? [];
  return rows.map((row) => row[0] ?? null);
}

const arrayOf = (rows: Scalar[][]): ArrayValue => ({ kind: "array", rows });

// ---------------------------------------------------------------------------
// Rounding as Excel does: on the 15-digit decimal, halves away from zero

function roundTo(x: number, digits: number, mode: "round" | "up" | "down"): number {
  const factor = 10 ** digits;
  const scaled = Number((Math.abs(x) * factor).toPrecision(15));
  const whole =
    mode === "round" ? Math.round(scaled) : mode === "up" ? Math.ceil(scaled) : Math.floor(scaled);
  const result = (Math.sign(x) * whole) / factor;
  return Object.is(result, -0) ? 0 : result;
}

// ---------------------------------------------------------------------------
// Math

define("SUM", (args, context) => numbers(args, context).reduce((a, b) => a + b, 0));
define("PRODUCT", (args, context) => numbers(args, context).reduce((a, b) => a * b, 1));
define("SUMSQ", (args, context) => numbers(args, context).reduce((a, b) => a + b * b, 0));
define(
  "ROUND",
  (args, context) => roundTo(num(args[0], context), integer(args[1], context), "round"),
  [2, 2],
);
define(
  "ROUNDUP",
  (args, context) => roundTo(num(args[0], context), integer(args[1], context), "up"),
  [2, 2],
);
define(
  "ROUNDDOWN",
  (args, context) => roundTo(num(args[0], context), integer(args[1], context), "down"),
  [2, 2],
);
define("INT", (args, context) => Math.floor(num(args[0], context)), [1, 1]);
define(
  "TRUNC",
  (args, context) => roundTo(num(args[0], context), integer(args[1], context, 0), "down"),
  [1, 2],
);
define("ABS", (args, context) => Math.abs(num(args[0], context)), [1, 1]);
define("SIGN", (args, context) => Math.sign(num(args[0], context)), [1, 1]);
define(
  "SQRT",
  (args, context) => {
    const x = num(args[0], context);
    return x < 0 ? err("#NUM!") : Math.sqrt(x);
  },
  [1, 1],
);
define(
  "POWER",
  (args, context) => {
    const result = num(args[0], context) ** num(args[1], context);
    return Number.isFinite(result) ? result : err("#NUM!");
  },
  [2, 2],
);
define("EXP", (args, context) => Math.exp(num(args[0], context)), [1, 1]);
define(
  "LN",
  (args, context) => {
    const x = num(args[0], context);
    return x <= 0 ? err("#NUM!") : Math.log(x);
  },
  [1, 1],
);
define(
  "LOG10",
  (args, context) => {
    const x = num(args[0], context);
    return x <= 0 ? err("#NUM!") : Math.log10(x);
  },
  [1, 1],
);
define(
  "LOG",
  (args, context) => {
    const x = num(args[0], context);
    const base = num(args[1], context, 10);
    return x <= 0 || base <= 0 || base === 1 ? err("#NUM!") : Math.log(x) / Math.log(base);
  },
  [1, 2],
);
define("PI", () => Math.PI, [0, 0]);
define(
  "MOD",
  (args, context) => {
    const a = num(args[0], context);
    const b = num(args[1], context);
    if (b === 0) return err("#DIV/0!");
    return a - b * Math.floor(a / b);
  },
  [2, 2],
);
define(
  "QUOTIENT",
  (args, context) => {
    const b = num(args[1], context);
    return b === 0 ? err("#DIV/0!") : Math.trunc(num(args[0], context) / b);
  },
  [2, 2],
);
define(
  "FACT",
  (args, context) => {
    const n = Math.floor(num(args[0], context));
    if (n < 0) return err("#NUM!");
    let result = 1;
    for (let k = 2; k <= n; k++) result *= k;
    return result;
  },
  [1, 1],
);
define(
  "COMBIN",
  (args, context) => {
    const n = Math.floor(num(args[0], context));
    const k = Math.floor(num(args[1], context));
    if (n < 0 || k < 0 || k > n) return err("#NUM!");
    let result = 1;
    for (let i = 1; i <= k; i++) result = (result * (n - k + i)) / i;
    return Math.round(result);
  },
  [2, 2],
);
define(
  "EVEN",
  (args, context) => {
    const x = num(args[0], context);
    const up = Math.ceil(Math.abs(x) / 2) * 2;
    return Math.sign(x) * up;
  },
  [1, 1],
);
define(
  "ODD",
  (args, context) => {
    const x = num(args[0], context);
    let up = Math.ceil(Math.abs(x));
    if (up % 2 === 0) up += 1;
    return x < 0 ? -up : up;
  },
  [1, 1],
);

function ceilingFloor(x: number, significance: number, up: boolean): Scalar {
  if (significance === 0) return up ? 0 : x === 0 ? 0 : err("#DIV/0!");
  if (x > 0 && significance < 0) return err("#NUM!");
  const ratio = Number((x / significance).toPrecision(15));
  const step = up ? Math.ceil(ratio) : Math.floor(ratio);
  // A negative number with a negative significance rounds away from zero (CEILING) or towards it (FLOOR).
  if (x < 0 && significance < 0) {
    const away = up
      ? Math.ceil(Math.abs(x) / Math.abs(significance))
      : Math.floor(Math.abs(x) / Math.abs(significance));
    return -away * Math.abs(significance);
  }
  return step * significance;
}
define(
  "CEILING",
  (args, context) => ceilingFloor(num(args[0], context), num(args[1], context, 1), true),
  [1, 2],
);
define(
  "FLOOR",
  (args, context) => ceilingFloor(num(args[0], context), num(args[1], context, 1), false),
  [1, 2],
);
define(
  ["CEILING.MATH", "CEILING.PRECISE", "ISO.CEILING"],
  (args, context) => {
    const x = num(args[0], context);
    const significance = Math.abs(num(args[1], context, 1));
    if (significance === 0) return 0;
    const away = num(args[2], context, 0) !== 0;
    if (x < 0 && away)
      return -Math.ceil(Number((-x / significance).toPrecision(15))) * significance;
    return Math.ceil(Number((x / significance).toPrecision(15))) * significance;
  },
  [1, 3],
);
define(
  ["FLOOR.MATH", "FLOOR.PRECISE"],
  (args, context) => {
    const x = num(args[0], context);
    const significance = Math.abs(num(args[1], context, 1));
    if (significance === 0) return 0;
    const towards = num(args[2], context, 0) !== 0;
    if (x < 0 && towards)
      return -Math.floor(Number((-x / significance).toPrecision(15))) * significance;
    return Math.floor(Number((x / significance).toPrecision(15))) * significance;
  },
  [1, 3],
);
define(
  "MROUND",
  (args, context) => {
    const x = num(args[0], context);
    const multiple = num(args[1], context);
    if (multiple === 0) return 0;
    if (Math.sign(x) * Math.sign(multiple) < 0) return err("#NUM!");
    return roundTo(x / multiple, 0, "round") * multiple;
  },
  [2, 2],
);
define("RAND", () => Math.random(), [0, 0]);
define(
  "RANDBETWEEN",
  (args, context) => {
    const low = Math.ceil(num(args[0], context));
    const high = Math.floor(num(args[1], context));
    return low + Math.floor(Math.random() * (high - low + 1));
  },
  [2, 2],
);

for (const [name, fn] of [
  ["SIN", Math.sin],
  ["COS", Math.cos],
  ["TAN", Math.tan],
  ["ATAN", Math.atan],
  ["SINH", Math.sinh],
  ["COSH", Math.cosh],
  ["TANH", Math.tanh],
  ["ASINH", Math.asinh],
] as const) {
  define(name, (args, context) => fn(num(args[0], context)), [1, 1]);
}
for (const [name, fn, ok] of [
  ["ASIN", Math.asin, (x: number) => x >= -1 && x <= 1],
  ["ACOS", Math.acos, (x: number) => x >= -1 && x <= 1],
  ["ACOSH", Math.acosh, (x: number) => x >= 1],
  ["ATANH", Math.atanh, (x: number) => x > -1 && x < 1],
] as const) {
  define(
    name,
    (args, context) => {
      const x = num(args[0], context);
      return ok(x) ? fn(x) : err("#NUM!");
    },
    [1, 1],
  );
}
define(
  "ATAN2",
  (args, context) => {
    const x = num(args[0], context);
    const y = num(args[1], context);
    return x === 0 && y === 0 ? err("#DIV/0!") : Math.atan2(y, x);
  },
  [2, 2],
);
define("RADIANS", (args, context) => (num(args[0], context) * Math.PI) / 180, [1, 1]);
define("DEGREES", (args, context) => (num(args[0], context) * 180) / Math.PI, [1, 1]);

// ---------------------------------------------------------------------------
// Statistics

const mean = (list: number[]) => list.reduce((a, b) => a + b, 0) / list.length;
function variance(list: number[], sample: boolean): Scalar {
  if (list.length < (sample ? 2 : 1)) return err("#DIV/0!");
  const m = mean(list);
  return list.reduce((sum, x) => sum + (x - m) ** 2, 0) / (list.length - (sample ? 1 : 0));
}

define("AVERAGE", (args, context) => {
  const list = numbers(args, context);
  return list.length === 0 ? err("#DIV/0!") : mean(list);
});
define("AVERAGEA", (args, context) => {
  const list = numbers(args, context, { logicalInRanges: true, textInRanges: true });
  return list.length === 0 ? err("#DIV/0!") : mean(list);
});
define("MAX", (args, context) => {
  const list = numbers(args, context);
  return list.length === 0 ? 0 : Math.max(...list);
});
define("MIN", (args, context) => {
  const list = numbers(args, context);
  return list.length === 0 ? 0 : Math.min(...list);
});
define("MAXA", (args, context) => {
  const list = numbers(args, context, { logicalInRanges: true, textInRanges: true });
  return list.length === 0 ? 0 : Math.max(...list);
});
define("MINA", (args, context) => {
  const list = numbers(args, context, { logicalInRanges: true, textInRanges: true });
  return list.length === 0 ? 0 : Math.min(...list);
});
define("MEDIAN", (args, context) => {
  const list = numbers(args, context).sort((a, b) => a - b);
  if (list.length === 0) return err("#NUM!");
  const middle = list.length >> 1;
  return list.length % 2
    ? (list[middle] as number)
    : ((list[middle - 1] as number) + (list[middle] as number)) / 2;
});
define(["MODE", "MODE.SNGL"], (args, context) => {
  const list = numbers(args, context);
  const counts = new Map<number, number>();
  let best: number | null = null;
  let bestCount = 1;
  for (const x of list) {
    const count = (counts.get(x) ?? 0) + 1;
    counts.set(x, count);
    if (count > bestCount) {
      best = x;
      bestCount = count;
    }
  }
  return best === null ? err("#N/A") : best;
});
define(["STDEV", "STDEV.S"], (args, context) => {
  const v = variance(numbers(args, context), true);
  return typeof v === "number" ? Math.sqrt(v) : v;
});
define(["STDEVP", "STDEV.P"], (args, context) => {
  const v = variance(numbers(args, context), false);
  return typeof v === "number" ? Math.sqrt(v) : v;
});
define(["VAR", "VAR.S"], (args, context) => variance(numbers(args, context), true));
define(["VARP", "VAR.P"], (args, context) => variance(numbers(args, context), false));
define("DEVSQ", (args, context) => {
  const list = numbers(args, context);
  if (list.length === 0) return err("#NUM!");
  const m = mean(list);
  return list.reduce((sum, x) => sum + (x - m) ** 2, 0);
});
define("AVEDEV", (args, context) => {
  const list = numbers(args, context);
  if (list.length === 0) return err("#NUM!");
  const m = mean(list);
  return list.reduce((sum, x) => sum + Math.abs(x - m), 0) / list.length;
});
define(
  "LARGE",
  (args, context) => {
    const list = numbers([args[0] as Node], context).sort((a, b) => b - a);
    const k = Math.ceil(num(args[1], context));
    return k < 1 || k > list.length ? err("#NUM!") : (list[k - 1] as number);
  },
  [2, 2],
);
define(
  "SMALL",
  (args, context) => {
    const list = numbers([args[0] as Node], context).sort((a, b) => a - b);
    const k = Math.ceil(num(args[1], context));
    return k < 1 || k > list.length ? err("#NUM!") : (list[k - 1] as number);
  },
  [2, 2],
);
define(
  ["RANK", "RANK.EQ"],
  (args, context) => {
    const x = num(args[0], context);
    const list = numbers([args[1] as Node], context);
    const ascending = num(args[2], context, 0) !== 0;
    if (!list.includes(x)) return err("#N/A");
    return 1 + list.filter((each) => (ascending ? each < x : each > x)).length;
  },
  [2, 3],
);
function percentile(list: number[], p: number): Scalar {
  if (list.length === 0 || p < 0 || p > 1) return err("#NUM!");
  const sorted = [...list].sort((a, b) => a - b);
  const at = p * (sorted.length - 1);
  const low = Math.floor(at);
  const fraction = at - low;
  const a = sorted[low] as number;
  const b = sorted[Math.min(low + 1, sorted.length - 1)] as number;
  return a + fraction * (b - a);
}
define(
  ["PERCENTILE", "PERCENTILE.INC"],
  (args, context) => percentile(numbers([args[0] as Node], context), num(args[1], context)),
  [2, 2],
);
function percentileExclusive(list: number[], p: number): Scalar {
  const sorted = [...list].sort((a, b) => a - b);
  const at = p * (sorted.length + 1);
  if (at < 1 || at > sorted.length) return err("#NUM!");
  const low = Math.floor(at);
  const a = sorted[low - 1] as number;
  const b = sorted[Math.min(low, sorted.length - 1)] as number;
  return a + (at - low) * (b - a);
}
define(
  "PERCENTILE.EXC",
  (args, context) =>
    percentileExclusive(numbers([args[0] as Node], context), num(args[1], context)),
  [2, 2],
);
define(
  "QUARTILE.EXC",
  (args, context) => {
    const quart = Math.floor(num(args[1], context));
    return quart <= 0 || quart >= 4
      ? err("#NUM!")
      : percentileExclusive(numbers([args[0] as Node], context), quart / 4);
  },
  [2, 2],
);
define(
  ["QUARTILE", "QUARTILE.INC"],
  (args, context) => {
    const quart = Math.floor(num(args[1], context));
    return quart < 0 || quart > 4
      ? err("#NUM!")
      : percentile(numbers([args[0] as Node], context), quart / 4);
  },
  [2, 2],
);
define(
  "CORREL",
  (args, context) => {
    const xs = listOf(value(args[0] as Node, context), context);
    const ys = listOf(value(args[1] as Node, context), context);
    if (xs.length !== ys.length) return err("#N/A");
    const pairs: [number, number][] = [];
    xs.forEach((x, index) => {
      const y = ys[index];
      if (typeof x === "number" && typeof y === "number") pairs.push([x, y]);
    });
    if (pairs.length < 2) return err("#DIV/0!");
    const mx = mean(pairs.map((pair) => pair[0]));
    const my = mean(pairs.map((pair) => pair[1]));
    let sxy = 0;
    let sxx = 0;
    let syy = 0;
    for (const [x, y] of pairs) {
      sxy += (x - mx) * (y - my);
      sxx += (x - mx) ** 2;
      syy += (y - my) ** 2;
    }
    return sxx === 0 || syy === 0 ? err("#DIV/0!") : sxy / Math.sqrt(sxx * syy);
  },
  [2, 2],
);

// ---------------------------------------------------------------------------
// Counting, and sums with conditions

define("COUNT", (args, context) => {
  let count = 0;
  for (const arg of args) {
    if (arg.kind === "missing") continue;
    const each = value(arg, context);
    if (isRange(each) || isArray(each)) {
      context.calc.eachValue(each, (cell) => {
        if (typeof cell === "number") count++;
      });
    } else {
      const single = context.calc.toScalar(each, context);
      if (typeof single === "number" || typeof single === "boolean") count++;
      else if (typeof single === "string" && numberFromText(single) !== null) count++;
    }
  }
  return count;
});
define("COUNTA", (args, context) => {
  let count = 0;
  for (const arg of args) {
    if (arg.kind === "missing") continue;
    const each = value(arg, context);
    if (isRange(each) || isArray(each)) {
      context.calc.eachValue(each, (cell) => {
        if (cell !== null) count++;
      });
    } else if (context.calc.toScalar(each, context) !== null) count++;
  }
  return count;
});
define(
  "COUNTBLANK",
  (args, context) => {
    const each = value(args[0] as Node, context);
    if (!isRange(each)) return err("#VALUE!");
    let filled = 0;
    context.calc.eachValue(each, (cell) => {
      if (cell !== null && cell !== "") filled++;
    });
    return (each.bottom - each.top + 1) * (each.right - each.left + 1) - filled;
  },
  [1, 1],
);

/** A criterion of COUNTIF and SUMIF: ">1000", "<>x", "a*", 4, TRUE. */
export function criterion(given: Scalar): (cell: Scalar) => boolean {
  if (typeof given === "number")
    return (cell) =>
      typeof cell === "number"
        ? cell === given
        : typeof cell === "string" && numberFromText(cell) === given;
  if (typeof given === "boolean") return (cell) => cell === given;
  if (given === null) return (cell) => cell === null || cell === "";
  if (isError(given)) return (cell) => isError(cell) && cell.error === given.error;
  const match = /^(<=|>=|<>|<|>|=)?([\s\S]*)$/.exec(given) as RegExpExecArray;
  const op = match[1] ?? "=";
  const rest = match[2] as string;
  if (rest === "") {
    if (op === "=") return (cell) => cell === null || cell === "";
    if (op === "<>") return (cell) => cell !== null && cell !== "";
  }
  const number =
    numberFromText(rest) ?? (/^\d{4}-\d\d?-\d\d?$/.test(rest) ? (toNumber(rest) as number) : null);
  if (number !== null) {
    const compare = (cell: number) => (cell === number ? 0 : cell < number ? -1 : 1);
    return (cell) => {
      if (typeof cell !== "number") return op === "<>";
      const order = compare(cell);
      return op === "="
        ? order === 0
        : op === "<>"
          ? order !== 0
          : op === "<"
            ? order < 0
            : op === ">"
              ? order > 0
              : op === "<="
                ? order <= 0
                : order >= 0;
    };
  }
  const upper = rest.toUpperCase();
  if (upper === "TRUE" || upper === "FALSE") {
    const flag = upper === "TRUE";
    return (cell) => (op === "<>" ? cell !== flag : op === "=" ? cell === flag : false);
  }
  if (op === "=" || op === "<>") {
    const pattern = wildcard(rest);
    return (cell) => {
      const matched = typeof cell === "string" && pattern.test(cell);
      return op === "=" ? matched : !matched;
    };
  }
  return (cell) => {
    if (typeof cell !== "string") return false;
    const order = compareValues(cell, rest);
    return op === "<" ? order < 0 : op === ">" ? order > 0 : op === "<=" ? order <= 0 : order >= 0;
  };
}

/** Excel's wildcards (* and ?, ~ escaping) as an anchored, case-insensitive pattern. */
export function wildcard(pattern: string): RegExp {
  let source = "";
  for (let index = 0; index < pattern.length; index++) {
    const ch = pattern[index] as string;
    if (ch === "~" && index + 1 < pattern.length) {
      source += escapeRegex(pattern[++index] as string);
    } else if (ch === "*") source += "[\\s\\S]*";
    else if (ch === "?") source += "[\\s\\S]";
    else source += escapeRegex(ch);
  }
  return new RegExp(`^${source}$`, "i");
}
const escapeRegex = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** The cells of a criteria range matching, as offsets from its first cell; blanks included when they match. */
function matchingOffsets(
  range: RangeValue,
  test: (cell: Scalar) => boolean,
  context: Context,
): [number, number][] {
  const found: [number, number][] = [];
  if (test(null)) {
    const used = context.calc.usedBounds(range.sheet);
    const bottom = Math.min(range.bottom, Math.max(used.rows, range.top));
    const right = Math.min(range.right, Math.max(used.columns - 1, range.left));
    for (let row = range.top; row <= bottom; row++) {
      for (let column = range.left; column <= right; column++) {
        if (test(context.calc.cellValue(range.sheet, row, column)))
          found.push([row - range.top, column - range.left]);
      }
    }
    return found;
  }
  context.calc.eachKey(range.sheet, range, (key) => {
    const cell = range.sheet.cells.get(key);
    if (cell && test(cell.value)) found.push([cell.row - range.top, cell.column - range.left]);
  });
  return found;
}

function criteriaRange(node: Node | undefined, context: Context): RangeValue {
  const each = node ? value(node, context) : null;
  if (!isRange(each)) return fail("#VALUE!");
  return each;
}

/** The cells matching every (range, criterion) pair, as offsets. */
function matchingAll(
  pairs: Node[],
  context: Context,
): { offsets: [number, number][]; first: RangeValue } {
  const ranges: RangeValue[] = [];
  const tests: ((cell: Scalar) => boolean)[] = [];
  for (let index = 0; index + 1 < pairs.length; index += 2) {
    ranges.push(criteriaRange(pairs[index], context));
    tests.push(criterion(scalar(pairs[index + 1], context)));
  }
  const first = ranges[0];
  if (!first) return fail("#VALUE!");
  const height = first.bottom - first.top;
  const width = first.right - first.left;
  for (const range of ranges) {
    if (range.bottom - range.top !== height || range.right - range.left !== width) fail("#VALUE!");
  }
  const offsets = matchingOffsets(first, tests[0] as (cell: Scalar) => boolean, context).filter(
    ([r, c]) =>
      ranges.every(
        (range, index) =>
          index === 0 ||
          (tests[index] as (cell: Scalar) => boolean)(
            context.calc.cellValue(range.sheet, range.top + r, range.left + c),
          ),
      ),
  );
  return { offsets, first };
}

function valuesAt(range: RangeValue, offsets: [number, number][], context: Context): number[] {
  const found: number[] = [];
  for (const [r, c] of offsets) {
    const cell = context.calc.cellValue(range.sheet, range.top + r, range.left + c);
    if (isError(cell)) throw new Fail(cell);
    if (typeof cell === "number") found.push(cell);
  }
  return found;
}

define("COUNTIF", (args, context) => matchingAll(args, context).offsets.length, [2, 2]);
define("COUNTIFS", (args, context) => matchingAll(args, context).offsets.length);
define(
  "SUMIF",
  (args, context) => {
    const { offsets, first } = matchingAll(args.slice(0, 2), context);
    const sum = args[2] && args[2].kind !== "missing" ? criteriaRange(args[2], context) : first;
    return valuesAt(sum, offsets, context).reduce((a, b) => a + b, 0);
  },
  [2, 3],
);
define("SUMIFS", (args, context) => {
  const sum = criteriaRange(args[0], context);
  const { offsets } = matchingAll(args.slice(1), context);
  return valuesAt(sum, offsets, context).reduce((a, b) => a + b, 0);
});
define(
  "AVERAGEIF",
  (args, context) => {
    const { offsets, first } = matchingAll(args.slice(0, 2), context);
    const range = args[2] && args[2].kind !== "missing" ? criteriaRange(args[2], context) : first;
    const list = valuesAt(range, offsets, context);
    return list.length === 0 ? err("#DIV/0!") : mean(list);
  },
  [2, 3],
);
define("AVERAGEIFS", (args, context) => {
  const list = valuesAt(
    criteriaRange(args[0], context),
    matchingAll(args.slice(1), context).offsets,
    context,
  );
  return list.length === 0 ? err("#DIV/0!") : mean(list);
});
define("MAXIFS", (args, context) => {
  const list = valuesAt(
    criteriaRange(args[0], context),
    matchingAll(args.slice(1), context).offsets,
    context,
  );
  return list.length === 0 ? 0 : Math.max(...list);
});
define("MINIFS", (args, context) => {
  const list = valuesAt(
    criteriaRange(args[0], context),
    matchingAll(args.slice(1), context).offsets,
    context,
  );
  return list.length === 0 ? 0 : Math.min(...list);
});

define("SUMPRODUCT", (args, context) => {
  const arrays = args.map((arg) => rowsOf(arrayValue(arg, context), context));
  const first = arrays[0];
  if (!first) return err("#VALUE!");
  let total = 0;
  for (let r = 0; r < first.length; r++) {
    const row = first[r] as Scalar[];
    for (let c = 0; c < row.length; c++) {
      let product = 1;
      for (const array of arrays) {
        const cell = array[r]?.[c];
        if (cell === undefined) return err("#VALUE!");
        if (isError(cell)) return cell;
        product *= typeof cell === "number" ? cell : 0;
      }
      total += product;
    }
  }
  return total;
});
function pairSum(args: Node[], context: Context, term: (x: number, y: number) => number): Scalar {
  const xs = rowsOf(arrayValue(args[0] as Node, context), context).flat();
  const ys = rowsOf(arrayValue(args[1] as Node, context), context).flat();
  if (xs.length !== ys.length) return err("#N/A");
  let total = 0;
  xs.forEach((x, index) => {
    const y = ys[index];
    if (typeof x === "number" && typeof y === "number") total += term(x, y);
  });
  return total;
}
define("SUMX2MY2", (args, context) => pairSum(args, context, (x, y) => x * x - y * y), [2, 2]);
define("SUMX2PY2", (args, context) => pairSum(args, context, (x, y) => x * x + y * y), [2, 2]);
define("SUMXMY2", (args, context) => pairSum(args, context, (x, y) => (x - y) ** 2), [2, 2]);

const SUBTOTALS: Record<number, string> = {
  1: "AVERAGE",
  2: "COUNT",
  3: "COUNTA",
  4: "MAX",
  5: "MIN",
  6: "PRODUCT",
  7: "STDEV",
  8: "STDEVP",
  9: "SUM",
  10: "VAR",
  11: "VARP",
};
define("SUBTOTAL", (args, context) => {
  const code = integer(args[0], context);
  const name = SUBTOTALS[code > 100 ? code - 100 : code];
  const fn = name ? FUNCTIONS.get(name) : undefined;
  if (!fn) return err("#VALUE!");
  // SUBTOTAL leaves out other SUBTOTALs in its ranges, so sections and a grand total don't count twice.
  const refs = args.slice(1);
  const values: Scalar[] = [];
  for (const ref of refs) {
    const each = value(ref, context);
    if (!isRange(each)) {
      values.push(context.calc.toScalar(each, context));
      continue;
    }
    // 101–111 leave out every hidden row; 1–11 only the rows a filter hides.
    const skipHidden = code > 100 || each.sheet.filtered;
    context.calc.eachKey(each.sheet, each, (key) => {
      const cell = each.sheet.cells.get(key);
      if (!cell || /\b(SUBTOTAL|AGGREGATE)\(/i.test(cell.formula?.text ?? "")) return;
      if (skipHidden && each.sheet.hiddenRows.has(cell.row)) return;
      if (cell.value !== null) values.push(cell.value);
    });
  }
  const literal: Node = { kind: "array", rows: [values.map(toLiteral)] };
  return fn([literal], context);
});
function toLiteral(each: Scalar): Node {
  if (typeof each === "number") return { kind: "number", value: each };
  if (typeof each === "string") return { kind: "string", value: each };
  if (typeof each === "boolean") return { kind: "boolean", value: each };
  if (each === null) return { kind: "missing" };
  return { kind: "error", code: each.error };
}

// ---------------------------------------------------------------------------
// Logic and information

define(
  "IF",
  (args, context) => {
    if (context.array) {
      const condition = value(args[0] as Node, context);
      if (
        isArray(condition) ||
        (isRange(condition) &&
          (condition.top !== condition.bottom || condition.left !== condition.right))
      ) {
        const rows = rowsOf(condition, context);
        const branch = (node: Node | undefined) =>
          node === undefined
            ? null
            : node.kind === "missing"
              ? null
              : rowsOf(value(node, context), context);
        const yes = branch(args[1]);
        const no = branch(args[2]);
        const at = (grid: Scalar[][] | null, r: number, c: number, fallback: Scalar): Scalar => {
          if (grid === null) return fallback;
          const row = grid.length === 1 ? grid[0] : grid[r];
          return (row?.length === 1 ? row[0] : row?.[c]) ?? err("#N/A");
        };
        return arrayOf(
          rows.map((row, r) =>
            row.map((cell, c) => {
              if (isError(cell)) return cell;
              const truth = typeof cell === "string" ? null : Boolean(cell);
              if (truth === null) return err("#VALUE!");
              return truth ? at(yes, r, c, args[1] ? 0 : true) : at(no, r, c, args[2] ? 0 : false);
            }),
          ),
        );
      }
    }
    const condition = check(scalar(args[0], context));
    const truth = typeof condition === "string" ? fail("#VALUE!") : Boolean(condition);
    const branch = truth ? args[1] : args[2];
    if (!branch) return truth;
    if (branch.kind === "missing") return 0;
    return value(branch, context);
  },
  [1, 3],
);
define("IFS", (args, context) => {
  for (let index = 0; index + 1 < args.length; index += 2) {
    if (bool(args[index], context, false)) return value(args[index + 1] as Node, context);
  }
  return err("#N/A");
});
define(
  "IFERROR",
  (args, context) => {
    const first = value(args[0] as Node, context);
    const single = isRange(first) || isArray(first) ? context.calc.toScalar(first, context) : first;
    return isError(single) ? value(args[1] as Node, context) : first;
  },
  [2, 2],
);
define(
  "IFNA",
  (args, context) => {
    const first = value(args[0] as Node, context);
    const single = isRange(first) || isArray(first) ? context.calc.toScalar(first, context) : first;
    return isError(single) && single.error === "#N/A" ? value(args[1] as Node, context) : first;
  },
  [2, 2],
);
function logicals(args: Node[], context: Context): boolean[] {
  const found: boolean[] = [];
  for (const arg of args) {
    const each = value(arg, context);
    if (isRange(each) || isArray(each)) {
      context.calc.eachValue(each, (cell) => {
        if (isError(cell)) throw new Fail(cell);
        if (typeof cell === "boolean") found.push(cell);
        else if (typeof cell === "number") found.push(cell !== 0);
      });
    } else found.push(bool(arg, context, false));
  }
  if (found.length === 0) fail("#VALUE!");
  return found;
}
define("AND", (args, context) => logicals(args, context).every(Boolean));
define("OR", (args, context) => logicals(args, context).some(Boolean));
define("XOR", (args, context) => logicals(args, context).filter(Boolean).length % 2 === 1);
define("NOT", (args, context) => !bool(args[0], context, false), [1, 1]);
define("TRUE", () => true, [0, 0]);
define("FALSE", () => false, [0, 0]);
define("SWITCH", (args, context) => {
  const subject = check(scalar(args[0], context));
  let index = 1;
  for (; index + 1 < args.length; index += 2) {
    if (compareValues(subject, check(scalar(args[index], context))) === 0)
      return value(args[index + 1] as Node, context);
  }
  return index < args.length ? value(args[index] as Node, context) : err("#N/A");
});
define("CHOOSE", (args, context) => {
  const index = integer(args[0], context);
  const chosen = args[index];
  return index < 1 || !chosen ? err("#VALUE!") : value(chosen, context);
});

const single = (args: Node[], context: Context): Scalar => {
  const each = value(args[0] as Node, context);
  return context.calc.toScalar(each, context);
};
define("ISBLANK", (args, context) => single(args, context) === null, [1, 1]);
define("ISNUMBER", (args, context) => typeof single(args, context) === "number", [1, 1]);
define("ISTEXT", (args, context) => typeof single(args, context) === "string", [1, 1]);
define("ISNONTEXT", (args, context) => typeof single(args, context) !== "string", [1, 1]);
define("ISLOGICAL", (args, context) => typeof single(args, context) === "boolean", [1, 1]);
define("ISERROR", (args, context) => isError(single(args, context)), [1, 1]);
define(
  "ISERR",
  (args, context) => {
    const each = single(args, context);
    return isError(each) && each.error !== "#N/A";
  },
  [1, 1],
);
define(
  "ISNA",
  (args, context) => {
    const each = single(args, context);
    return isError(each) && each.error === "#N/A";
  },
  [1, 1],
);
define("ISEVEN", (args, context) => Math.floor(Math.abs(num(args[0], context))) % 2 === 0, [1, 1]);
define("ISODD", (args, context) => Math.floor(Math.abs(num(args[0], context))) % 2 === 1, [1, 1]);
define("ISREF", (args, context) => isRange(value(args[0] as Node, context)), [1, 1]);
define("NA", () => err("#N/A"), [0, 0]);
define(
  "ERROR.TYPE",
  (args, context) => {
    const each = single(args, context);
    const codes = [
      "#NULL!",
      "#DIV/0!",
      "#VALUE!",
      "#REF!",
      "#NAME?",
      "#NUM!",
      "#N/A",
      "#GETTING_DATA",
    ];
    return isError(each) ? codes.indexOf(each.error) + 1 || err("#N/A") : err("#N/A");
  },
  [1, 1],
);
define(
  "N",
  (args, context) => {
    const each = single(args, context);
    if (isError(each)) return each;
    return typeof each === "number" ? each : typeof each === "boolean" ? Number(each) : 0;
  },
  [1, 1],
);
define(
  "T",
  (args, context) => {
    const each = single(args, context);
    if (isError(each)) return each;
    return typeof each === "string" ? each : "";
  },
  [1, 1],
);
define(
  "TYPE",
  (args, context) => {
    const each = value(args[0] as Node, context);
    if (isArray(each) || (isRange(each) && (each.top !== each.bottom || each.left !== each.right)))
      return 64;
    const one = context.calc.toScalar(each, context);
    return typeof one === "number" || one === null
      ? 1
      : typeof one === "string"
        ? 2
        : typeof one === "boolean"
          ? 4
          : 16;
  },
  [1, 1],
);

// ---------------------------------------------------------------------------
// Text

define(["CONCATENATE"], (args, context) => args.map((arg) => text(arg, context)).join(""));
define(["CONCAT"], (args, context) => {
  let out = "";
  for (const arg of args) {
    const each = value(arg, context);
    if (isRange(each) || isArray(each)) {
      for (const row of rowsOf(each, context)) for (const cell of row) out += check(toText(cell));
    } else out += check(toText(check(context.calc.toScalar(each, context))));
  }
  return out;
});
define("TEXTJOIN", (args, context) => {
  const delimiter = text(args[0], context);
  const skipEmpty = bool(args[1], context, true);
  const parts: string[] = [];
  for (const arg of args.slice(2)) {
    const each = value(arg, context);
    const cells =
      isRange(each) || isArray(each)
        ? rowsOf(each, context).flat()
        : [context.calc.toScalar(each, context)];
    for (const cell of cells) {
      const part = check(toText(cell));
      if (!(skipEmpty && part === "")) parts.push(part);
    }
  }
  return parts.join(delimiter);
});
define(
  "LEFT",
  (args, context) => {
    const n = integer(args[1], context, 1);
    return n < 0 ? err("#VALUE!") : [...text(args[0], context)].slice(0, n).join("");
  },
  [1, 2],
);
define(
  "RIGHT",
  (args, context) => {
    const n = integer(args[1], context, 1);
    if (n < 0) return err("#VALUE!");
    const chars = [...text(args[0], context)];
    return n === 0 ? "" : chars.slice(-n).join("");
  },
  [1, 2],
);
define(
  "MID",
  (args, context) => {
    const start = integer(args[1], context);
    const n = integer(args[2], context);
    if (start < 1 || n < 0) return err("#VALUE!");
    return [...text(args[0], context)].slice(start - 1, start - 1 + n).join("");
  },
  [3, 3],
);
define("LEN", (args, context) => [...text(args[0], context)].length, [1, 1]);
define("UPPER", (args, context) => text(args[0], context).toUpperCase(), [1, 1]);
define("LOWER", (args, context) => text(args[0], context).toLowerCase(), [1, 1]);
define(
  "PROPER",
  (args, context) =>
    text(args[0], context)
      .toLowerCase()
      .replace(
        /(^|[^\p{L}])(\p{L})/gu,
        (_, before: string, letter: string) => before + letter.toUpperCase(),
      ),
  [1, 1],
);
define(
  "TRIM",
  (args, context) => text(args[0], context).replace(/ +/g, " ").replace(/^ | $/g, ""),
  [1, 1],
);
define(
  "CLEAN",
  (args, context) =>
    text(args[0], context).replace(/\p{Cc}/gu, (ch) => (ch.charCodeAt(0) < 32 ? "" : ch)),
  [1, 1],
);
define(
  "REPT",
  (args, context) => {
    const n = integer(args[1], context);
    return n < 0 ? err("#VALUE!") : text(args[0], context).repeat(n);
  },
  [2, 2],
);
define("EXACT", (args, context) => text(args[0], context) === text(args[1], context), [2, 2]);
define(
  "SUBSTITUTE",
  (args, context) => {
    const source = text(args[0], context);
    const old = text(args[1], context);
    const replacement = text(args[2], context);
    if (old === "") return source;
    if (!args[3] || args[3].kind === "missing") return source.split(old).join(replacement);
    const instance = integer(args[3], context);
    if (instance < 1) return err("#VALUE!");
    let at = -1;
    for (let k = 0; k < instance; k++) {
      at = source.indexOf(old, at + 1);
      if (at < 0) return source;
    }
    return source.slice(0, at) + replacement + source.slice(at + old.length);
  },
  [3, 4],
);
define(
  "REPLACE",
  (args, context) => {
    const source = [...text(args[0], context)];
    const start = integer(args[1], context);
    const n = integer(args[2], context);
    if (start < 1 || n < 0) return err("#VALUE!");
    return [
      ...source.slice(0, start - 1),
      text(args[3], context),
      ...source.slice(start - 1 + n),
    ].join("");
  },
  [4, 4],
);
define(
  "FIND",
  (args, context) => {
    const find = text(args[0], context);
    const within = text(args[1], context);
    const start = integer(args[2], context, 1);
    if (start < 1 || start > within.length + 1) return err("#VALUE!");
    const at = within.indexOf(find, start - 1);
    return at < 0 ? err("#VALUE!") : at + 1;
  },
  [2, 3],
);
define(
  "SEARCH",
  (args, context) => {
    const find = text(args[0], context);
    const within = text(args[1], context);
    const start = integer(args[2], context, 1);
    if (start < 1 || start > within.length + 1) return err("#VALUE!");
    const pattern = wildcard(find).source.slice(1, -1);
    const match = new RegExp(pattern, "i").exec(within.slice(start - 1));
    return match ? match.index + start : err("#VALUE!");
  },
  [2, 3],
);
define(
  "CHAR",
  (args, context) => {
    const code = integer(args[0], context);
    return code < 1 || code > 255 ? err("#VALUE!") : String.fromCharCode(code);
  },
  [1, 1],
);
define(
  "CODE",
  (args, context) => {
    const source = text(args[0], context);
    return source === "" ? err("#VALUE!") : (source.codePointAt(0) as number);
  },
  [1, 1],
);
define("UNICHAR", (args, context) => String.fromCodePoint(integer(args[0], context)), [1, 1]);
define(
  "UNICODE",
  (args, context) => text(args[0], context).codePointAt(0) ?? err("#VALUE!"),
  [1, 1],
);
define(
  "VALUE",
  (args, context) => {
    const each = check(scalar(args[0], context));
    if (typeof each === "number") return each;
    const number = toNumber(typeof each === "string" ? each.trim() : each);
    return isError(number) ? err("#VALUE!") : number;
  },
  [1, 1],
);
define(
  "TEXT",
  (args, context) => {
    const each = check(scalar(args[0], context));
    const format = text(args[1], context);
    const number =
      typeof each === "number"
        ? each
        : typeof each === "string"
          ? numberFromText(each)
          : typeof each === "boolean"
            ? null
            : 0;
    if (number === null) return typeof each === "string" ? each : check(toText(each));
    if (/^general$/i.test(format.trim())) return numberText(number);
    return displayNumber(number, format, {
      date1904: context.calc.book.date1904,
      generalWidth: 255,
    }).text;
  },
  [2, 2],
);
define(
  "FIXED",
  (args, context) => {
    const x = num(args[0], context);
    const decimals = integer(args[1], context, 2);
    const noCommas = bool(args[2], context, false);
    const rounded = roundTo(x, decimals, "round");
    const shown = rounded.toFixed(Math.max(0, decimals));
    if (noCommas) return shown;
    const [whole, fraction] = shown.split(".");
    return `${(whole as string).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}${fraction ? `.${fraction}` : ""}`;
  },
  [1, 3],
);
define(
  "HYPERLINK",
  (args, context) => (args[1] ? value(args[1], context) : value(args[0] as Node, context)),
  [1, 2],
);

// ---------------------------------------------------------------------------
// Dates: serial numbers in the 1900 system (with Lotus's 29 February 1900) or the 1904 one

const DAY = 86_400_000;

/** Excel's serial for a date, its day counted on from the 1st of its month as Excel does (DATE(1900,2,29) is 60). */
function serialOf(year: number, month: number, day: number, date1904: boolean): number {
  const months = year * 12 + (month - 1);
  const y = Math.floor(months / 12);
  const m = months - y * 12;
  const first = Date.UTC(y, m, 1);
  if (date1904) return (first - Date.UTC(1904, 0, 1)) / DAY + day - 1;
  const n = (first - Date.UTC(1899, 11, 31)) / DAY;
  return (n >= 60 ? n + 1 : n) + day - 1;
}

function partsOf(serial: number, date1904: boolean): { year: number; month: number; day: number } {
  const whole = Math.floor(serial);
  if (whole === 0 && !date1904) return { year: 1900, month: 1, day: 0 };
  let date: Date;
  if (date1904) date = new Date(Date.UTC(1904, 0, 1) + whole * DAY);
  else if (whole === 60) return { year: 1900, month: 2, day: 29 };
  else
    date = new Date((whole < 60 ? Date.UTC(1899, 11, 31) : Date.UTC(1899, 11, 30)) + whole * DAY);
  return { year: date.getUTCFullYear(), month: date.getUTCMonth() + 1, day: date.getUTCDate() };
}

const dateSerial = (node: Node | undefined, context: Context) => {
  const serial = num(node, context);
  if (serial < 0) fail("#NUM!");
  return serial;
};

define(
  "DATE",
  (args, context) => {
    let year = integer(args[0], context);
    if (year < 1900) year += 1900;
    const serial = serialOf(
      year,
      integer(args[1], context),
      integer(args[2], context),
      context.calc.book.date1904,
    );
    return serial < 0 ? err("#NUM!") : serial;
  },
  [3, 3],
);
define(
  "TIME",
  (args, context) => {
    const seconds =
      integer(args[0], context) * 3600 + integer(args[1], context) * 60 + integer(args[2], context);
    return seconds < 0 ? err("#NUM!") : (seconds % 86_400) / 86_400;
  },
  [3, 3],
);
define(
  "YEAR",
  (args, context) => partsOf(dateSerial(args[0], context), context.calc.book.date1904).year,
  [1, 1],
);
define(
  "MONTH",
  (args, context) => partsOf(dateSerial(args[0], context), context.calc.book.date1904).month,
  [1, 1],
);
define(
  "DAY",
  (args, context) => partsOf(dateSerial(args[0], context), context.calc.book.date1904).day,
  [1, 1],
);
const timePart = (serial: number) => Math.round((serial - Math.floor(serial)) * 86_400);
define(
  "HOUR",
  (args, context) => Math.floor(timePart(dateSerial(args[0], context)) / 3600) % 24,
  [1, 1],
);
define(
  "MINUTE",
  (args, context) => Math.floor(timePart(dateSerial(args[0], context)) / 60) % 60,
  [1, 1],
);
define("SECOND", (args, context) => timePart(dateSerial(args[0], context)) % 60, [1, 1]);
define(
  "TODAY",
  (_args, context) => {
    const now = new Date();
    return serialOf(
      now.getFullYear(),
      now.getMonth() + 1,
      now.getDate(),
      context.calc.book.date1904,
    );
  },
  [0, 0],
);
define(
  "NOW",
  (_args, context) => {
    const now = new Date();
    const day = serialOf(
      now.getFullYear(),
      now.getMonth() + 1,
      now.getDate(),
      context.calc.book.date1904,
    );
    return day + (now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds()) / 86_400;
  },
  [0, 0],
);
function addMonths(serial: number, months: number, date1904: boolean, endOfMonth: boolean): number {
  const { year, month, day } = partsOf(serial, date1904);
  const target = month - 1 + months;
  const y = year + Math.floor(target / 12);
  const m = ((target % 12) + 12) % 12;
  const last = new Date(Date.UTC(y, m + 1, 0)).getUTCDate();
  return serialOf(y, m + 1, endOfMonth ? last : Math.min(day, last), date1904);
}
define(
  "EDATE",
  (args, context) =>
    addMonths(
      dateSerial(args[0], context),
      integer(args[1], context),
      context.calc.book.date1904,
      false,
    ),
  [2, 2],
);
define(
  "EOMONTH",
  (args, context) =>
    addMonths(
      dateSerial(args[0], context),
      integer(args[1], context),
      context.calc.book.date1904,
      true,
    ),
  [2, 2],
);
define(
  "WEEKDAY",
  (args, context) => {
    const serial =
      Math.floor(dateSerial(args[0], context)) + (context.calc.book.date1904 ? 1462 : 0);
    const type = integer(args[1], context, 1);
    const sunday = (((serial - 1) % 7) + 7) % 7; // 0 for Sunday
    if (type === 1 || type === 17) return sunday + 1;
    if (type === 2 || type === 11) return ((sunday + 6) % 7) + 1;
    if (type === 3) return (sunday + 6) % 7;
    if (type >= 12 && type <= 16) return ((sunday - (type - 10) + 14) % 7) + 1;
    return err("#NUM!");
  },
  [1, 2],
);
define(
  "DAYS",
  (args, context) => Math.floor(num(args[0], context)) - Math.floor(num(args[1], context)),
  [2, 2],
);
define(
  "DAYS360",
  (args, context) => {
    const date1904 = context.calc.book.date1904;
    const a = partsOf(dateSerial(args[0], context), date1904);
    const b = partsOf(dateSerial(args[1], context), date1904);
    const european = bool(args[2], context, false);
    let d1 = a.day;
    let d2 = b.day;
    if (european) {
      if (d1 === 31) d1 = 30;
      if (d2 === 31) d2 = 30;
    } else {
      const lastOfFeb = (p: { year: number; month: number; day: number }) =>
        p.month === 2 && p.day === new Date(Date.UTC(p.year, 2, 0)).getUTCDate();
      if (lastOfFeb(a)) {
        if (lastOfFeb(b)) d2 = 30;
        d1 = 30;
      }
      if (d1 === 31) d1 = 30;
      if (d2 === 31 && d1 >= 30) d2 = 30;
    }
    return (b.year - a.year) * 360 + (b.month - a.month) * 30 + (d2 - d1);
  },
  [2, 3],
);
define(
  "DATEDIF",
  (args, context) => {
    const date1904 = context.calc.book.date1904;
    const start = dateSerial(args[0], context);
    const end = dateSerial(args[1], context);
    if (start > end) return err("#NUM!");
    const unit = text(args[2], context).toUpperCase();
    const a = partsOf(start, date1904);
    const b = partsOf(end, date1904);
    let months = (b.year - a.year) * 12 + (b.month - a.month);
    if (b.day < a.day) months--;
    switch (unit) {
      case "D":
        return Math.floor(end) - Math.floor(start);
      case "M":
        return months;
      case "Y":
        return Math.floor(months / 12);
      case "YM":
        return months % 12;
      case "MD":
        return b.day >= a.day
          ? b.day - a.day
          : b.day + new Date(Date.UTC(b.year, b.month - 1, 0)).getUTCDate() - a.day;
      case "YD": {
        const anniversary = serialOf(b.year, a.month, a.day, date1904);
        const days = Math.floor(end) - anniversary;
        return days >= 0 ? days : Math.floor(end) - serialOf(b.year - 1, a.month, a.day, date1904);
      }
    }
    return err("#NUM!");
  },
  [3, 3],
);
function holidays(node: Node | undefined, context: Context): Set<number> {
  if (!node) return new Set();
  return new Set(numbers([node], context).map(Math.floor));
}
const isWeekend = (serial: number, date1904: boolean) => {
  const sunday = (((Math.floor(serial) + (date1904 ? 1462 : 0) - 1) % 7) + 7) % 7;
  return sunday === 0 || sunday === 6;
};
define(
  "NETWORKDAYS",
  (args, context) => {
    const date1904 = context.calc.book.date1904;
    let start = Math.floor(num(args[0], context));
    let end = Math.floor(num(args[1], context));
    const off = holidays(args[2], context);
    let sign = 1;
    if (start > end) {
      [start, end] = [end, start];
      sign = -1;
    }
    let count = 0;
    for (let day = start; day <= end; day++)
      if (!isWeekend(day, date1904) && !off.has(day)) count++;
    return sign * count;
  },
  [2, 3],
);
define(
  "WORKDAY",
  (args, context) => {
    const date1904 = context.calc.book.date1904;
    let day = Math.floor(num(args[0], context));
    let left = integer(args[1], context);
    const off = holidays(args[2], context);
    const step = left < 0 ? -1 : 1;
    while (left !== 0) {
      day += step;
      if (!isWeekend(day, date1904) && !off.has(day)) left -= step;
    }
    return day;
  },
  [2, 3],
);
define(
  "DATEVALUE",
  (args, context) => {
    const each = text(args[0], context);
    const number = toNumber(each);
    return isError(number) ? err("#VALUE!") : Math.floor(number);
  },
  [1, 1],
);

// ---------------------------------------------------------------------------
// Money

define(
  "PMT",
  (args, context) => {
    const rate = num(args[0], context);
    const periods = num(args[1], context);
    const pv = num(args[2], context);
    const fv = num(args[3], context, 0);
    const type = num(args[4], context, 0) !== 0 ? 1 : 0;
    if (periods === 0) return err("#NUM!");
    if (rate === 0) return -(pv + fv) / periods;
    const growth = (1 + rate) ** periods;
    return -(rate * (fv + pv * growth)) / ((1 + rate * type) * (growth - 1));
  },
  [3, 5],
);
define(
  "FV",
  (args, context) => {
    const rate = num(args[0], context);
    const periods = num(args[1], context);
    const payment = num(args[2], context);
    const pv = num(args[3], context, 0);
    const type = num(args[4], context, 0) !== 0 ? 1 : 0;
    if (rate === 0) return -(pv + payment * periods);
    const growth = (1 + rate) ** periods;
    return -(pv * growth + (payment * (1 + rate * type) * (growth - 1)) / rate);
  },
  [3, 5],
);
define(
  "PV",
  (args, context) => {
    const rate = num(args[0], context);
    const periods = num(args[1], context);
    const payment = num(args[2], context);
    const fv = num(args[3], context, 0);
    const type = num(args[4], context, 0) !== 0 ? 1 : 0;
    if (rate === 0) return -(fv + payment * periods);
    const growth = (1 + rate) ** periods;
    return -(fv + (payment * (1 + rate * type) * (growth - 1)) / rate) / growth;
  },
  [3, 5],
);
define(
  "NPER",
  (args, context) => {
    const rate = num(args[0], context);
    const payment = num(args[1], context);
    const pv = num(args[2], context);
    const fv = num(args[3], context, 0);
    const type = num(args[4], context, 0) !== 0 ? 1 : 0;
    if (rate === 0) return payment === 0 ? err("#NUM!") : -(pv + fv) / payment;
    const a = payment * (1 + rate * type) - fv * rate;
    const b = payment * (1 + rate * type) + pv * rate;
    if (a / b <= 0) return err("#NUM!");
    return Math.log(a / b) / Math.log(1 + rate);
  },
  [3, 5],
);
define("NPV", (args, context) => {
  const rate = num(args[0], context);
  const list = numbers(args.slice(1), context);
  return list.reduce((sum, x, index) => sum + x / (1 + rate) ** (index + 1), 0);
});
function newton(f: (x: number) => number, guess: number): number | null {
  let x = guess;
  for (let step = 0; step < 100; step++) {
    const y = f(x);
    if (Math.abs(y) < 1e-10) return x;
    const h = 1e-7 * Math.max(1, Math.abs(x));
    const slope = (f(x + h) - y) / h;
    if (slope === 0 || !Number.isFinite(slope)) return null;
    const next = x - y / slope;
    if (!Number.isFinite(next)) return null;
    if (Math.abs(next - x) < 1e-12) return next;
    x = next;
  }
  return Math.abs(f(x)) < 1e-6 ? x : null;
}
define(
  "IRR",
  (args, context) => {
    const list = numbers([args[0] as Node], context);
    const guess = num(args[1], context, 0.1);
    if (!list.some((x) => x > 0) || !list.some((x) => x < 0)) return err("#NUM!");
    const rate = newton(
      (r) => list.reduce((sum, x, index) => sum + x / (1 + r) ** index, 0),
      guess,
    );
    return rate === null ? err("#NUM!") : rate;
  },
  [1, 2],
);
define(
  "RATE",
  (args, context) => {
    const periods = num(args[0], context);
    const payment = num(args[1], context);
    const pv = num(args[2], context);
    const fv = num(args[3], context, 0);
    const type = num(args[4], context, 0) !== 0 ? 1 : 0;
    const guess = num(args[5], context, 0.1);
    const f = (r: number) =>
      r === 0
        ? pv + payment * periods + fv
        : pv * (1 + r) ** periods + (payment * (1 + r * type) * ((1 + r) ** periods - 1)) / r + fv;
    const rate = newton(f, guess);
    return rate === null ? err("#NUM!") : rate;
  },
  [3, 6],
);

// ---------------------------------------------------------------------------
// Looking up and references

/** The index of `wanted` in `list`: exact (with wildcards for text), or by Excel's binary search. */
function lookupIndex(list: Scalar[], wanted: Scalar, mode: 0 | 1 | -1): number {
  if (mode === 0) {
    if (typeof wanted === "string" && /[*?~]/.test(wanted)) {
      const pattern = wildcard(wanted);
      return list.findIndex((each) => typeof each === "string" && pattern.test(each));
    }
    return list.findIndex(
      (each) => each !== null && compareValues(each, wanted) === 0 && typeof each === typeof wanted,
    );
  }
  if (mode === -1) {
    let found = -1;
    for (let index = 0; index < list.length; index++) {
      const each = list[index] as Scalar;
      if (each === null || typeof each !== typeof wanted) continue;
      if (compareValues(each, wanted) >= 0) found = index;
      else break;
    }
    return found;
  }
  // Ascending: the last value not greater than the one wanted, as Excel's binary search finds it.
  let low = 0;
  let high = list.length - 1;
  let found = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    let probe = middle;
    while (probe <= high && (list[probe] === null || typeof list[probe] !== typeof wanted)) probe++;
    if (probe > high) {
      high = middle - 1;
      continue;
    }
    const order = compareValues(list[probe] as Scalar, wanted);
    if (order <= 0) {
      found = probe;
      low = probe + 1;
    } else high = middle - 1;
  }
  return found;
}

/** A range's line of cells (a column, or a row with `across`), down to the last stored row or column. */
function lineOf(range: RangeValue, index: number, across: boolean, context: Context): Scalar[] {
  const used = context.calc.usedBounds(range.sheet);
  const line: Scalar[] = [];
  if (across) {
    const right = Math.min(range.right, Math.max(used.columns - 1, range.left));
    for (let column = range.left; column <= right; column++)
      line.push(context.calc.cellValue(range.sheet, range.top + index, column));
  } else {
    const bottom = Math.min(range.bottom, Math.max(used.rows, range.top));
    for (let row = range.top; row <= bottom; row++)
      line.push(context.calc.cellValue(range.sheet, row, range.left + index));
  }
  return line;
}

for (const across of [false, true]) {
  define(
    across ? "HLOOKUP" : "VLOOKUP",
    (args, context) => {
      const wanted = check(scalar(args[0], context));
      const table = value(args[1] as Node, context);
      const index = integer(args[2], context);
      const approximate = bool(args[3], context, true);
      if (index < 1) return err("#VALUE!");
      if (isRange(table)) {
        const width = across ? table.bottom - table.top + 1 : table.right - table.left + 1;
        if (index > width) return err("#REF!");
        const at = lookupIndex(lineOf(table, 0, across, context), wanted, approximate ? 1 : 0);
        if (at < 0) return err("#N/A");
        return across
          ? context.calc.cellValue(table.sheet, table.top + index - 1, table.left + at)
          : context.calc.cellValue(table.sheet, table.top + at, table.left + index - 1);
      }
      const rows = rowsOf(table, context);
      const lines = across
        ? (rows[0] ?? []).map((_, c) => rows.map((row) => row[c] ?? null))
        : rows;
      if (index > (lines[0]?.length ?? 0)) return err("#REF!");
      const at = lookupIndex(
        lines.map((line) => line[0] ?? null),
        wanted,
        approximate ? 1 : 0,
      );
      if (at < 0) return err("#N/A");
      return (lines[at] as Scalar[])[index - 1] ?? null;
    },
    [3, 4],
  );
}
define(
  "MATCH",
  (args, context) => {
    const wanted = check(scalar(args[0], context));
    const type = integer(args[2], context, 1);
    const list = listOf(value(args[1] as Node, context), context);
    const at = lookupIndex(list, wanted, type === 0 ? 0 : type > 0 ? 1 : -1);
    return at < 0 ? err("#N/A") : at + 1;
  },
  [2, 3],
);
define(
  "LOOKUP",
  (args, context) => {
    const wanted = check(scalar(args[0], context));
    const keys = listOf(value(args[1] as Node, context), context);
    const results = args[2] ? listOf(value(args[2], context), context) : keys;
    const at = lookupIndex(keys, wanted, 1);
    return at < 0 ? err("#N/A") : (results[at] ?? err("#N/A"));
  },
  [2, 3],
);
define(
  "XLOOKUP",
  (args, context) => {
    const wanted = check(scalar(args[0], context));
    const keysValue = value(args[1] as Node, context);
    const resultsValue = value(args[2] as Node, context);
    const mode = integer(args[4], context, 0);
    const searchLast = integer(args[5], context, 1) === -1;
    const keys = listOf(keysValue, context);
    const order = keys.map((_, index) => index);
    if (searchLast) order.reverse();
    let found = -1;
    if (mode === 0 || mode === 2) {
      const pattern = mode === 2 && typeof wanted === "string" ? wildcard(wanted) : null;
      found =
        order.find((index) => {
          const each = keys[index] as Scalar;
          return pattern
            ? typeof each === "string" && pattern.test(each)
            : each !== null && typeof each === typeof wanted && compareValues(each, wanted) === 0;
        }) ?? -1;
    } else {
      let best: number | null = null;
      for (const index of order) {
        const each = keys[index] as Scalar;
        if (each === null || typeof each !== typeof wanted) continue;
        const cmp = compareValues(each, wanted);
        if (cmp === 0) {
          best = index;
          break;
        }
        if (
          mode === -1 &&
          cmp < 0 &&
          (best === null || compareValues(each, keys[best] as Scalar) > 0)
        )
          best = index;
        if (
          mode === 1 &&
          cmp > 0 &&
          (best === null || compareValues(each, keys[best] as Scalar) < 0)
        )
          best = index;
      }
      found = best ?? -1;
    }
    if (found < 0)
      return args[3] && args[3].kind !== "missing" ? value(args[3], context) : err("#N/A");
    if (isRange(resultsValue)) {
      const across =
        keysValue !== null &&
        isRange(keysValue) &&
        keysValue.top === keysValue.bottom &&
        keysValue.left !== keysValue.right;
      return across
        ? { ...resultsValue, left: resultsValue.left + found, right: resultsValue.left + found }
        : { ...resultsValue, top: resultsValue.top + found, bottom: resultsValue.top + found };
    }
    const rows = rowsOf(resultsValue, context);
    return rows.length === 1
      ? (rows[0]?.[found] ?? err("#N/A"))
      : (rows[found]?.[0] ?? err("#N/A"));
  },
  [3, 6],
);
define(
  "INDEX",
  (args, context) => {
    const source = value(args[0] as Node, context);
    let row = integer(args[1], context, 0);
    let column = integer(args[2], context, 0);
    if (isRange(source)) {
      const height = source.bottom - source.top + 1;
      const width = source.right - source.left + 1;
      if (args.length === 2 && height === 1) {
        column = row;
        row = 0;
      }
      if (row < 0 || column < 0 || row > height || column > width) return err("#REF!");
      return {
        kind: "range",
        sheet: source.sheet,
        top: row === 0 ? source.top : source.top + row - 1,
        bottom: row === 0 ? source.bottom : source.top + row - 1,
        left: column === 0 ? source.left : source.left + column - 1,
        right: column === 0 ? source.right : source.left + column - 1,
      } satisfies RangeValue;
    }
    const rows = rowsOf(source, context);
    if (rows.length === 1 && args.length === 2) {
      column = row;
      row = 1;
    }
    const picked = rows[Math.max(row, 1) - 1]?.[Math.max(column, 1) - 1];
    return picked === undefined ? err("#REF!") : picked;
  },
  [2, 4],
);
define(
  "ROW",
  (args, context) => {
    if (!args[0]) return context.row;
    const each = value(args[0], context);
    return isRange(each) ? each.top : err("#VALUE!");
  },
  [0, 1],
);
define(
  "COLUMN",
  (args, context) => {
    if (!args[0]) return context.column + 1;
    const each = value(args[0], context);
    return isRange(each) ? each.left + 1 : err("#VALUE!");
  },
  [0, 1],
);
define(
  "ROWS",
  (args, context) => {
    const each = value(args[0] as Node, context);
    if (isRange(each)) return each.bottom - each.top + 1;
    return rowsOf(each, context).length;
  },
  [1, 1],
);
define(
  "COLUMNS",
  (args, context) => {
    const each = value(args[0] as Node, context);
    if (isRange(each)) return each.right - each.left + 1;
    return rowsOf(each, context)[0]?.length ?? 1;
  },
  [1, 1],
);
define(
  "OFFSET",
  (args, context) => {
    const base = value(args[0] as Node, context);
    if (!isRange(base)) return err("#VALUE!");
    const rows = integer(args[1], context);
    const columns = integer(args[2], context);
    const height =
      args[3] && args[3].kind !== "missing"
        ? integer(args[3], context)
        : base.bottom - base.top + 1;
    const width =
      args[4] && args[4].kind !== "missing"
        ? integer(args[4], context)
        : base.right - base.left + 1;
    const top = base.top + rows;
    const left = base.left + columns;
    if (top < 1 || left < 0 || height < 1 || width < 1) return err("#REF!");
    return {
      kind: "range",
      sheet: base.sheet,
      top,
      left,
      bottom: top + height - 1,
      right: left + width - 1,
    } satisfies RangeValue;
  },
  [3, 5],
);
define(
  "INDIRECT",
  (args, context) => {
    const reference = text(args[0], context);
    if (!bool(args[1], context, true)) throw new CannotCalculate("INDIRECT in R1C1 style");
    const bang = reference.lastIndexOf("!");
    const sheetName =
      bang >= 0
        ? reference
            .slice(0, bang)
            .replace(/^'(.*)'$/, "$1")
            .replace(/''/g, "'")
        : null;
    const body = (bang >= 0 ? reference.slice(bang + 1) : reference).replace(/\$/g, "");
    const sheet = sheetName ? context.calc.sheetNamed(sheetName) : context.sheet;
    const match = /^([A-Za-z]{1,3})(\d+)(?::([A-Za-z]{1,3})(\d+))?$/.exec(body);
    if (!sheet || !match) {
      const named = context.calc.findName(body, sheetName, context.sheet);
      if (named?.node) return context.calc.evaluate(named.node, context);
      return err("#REF!");
    }
    const col = (letters: string) =>
      [...letters.toUpperCase()].reduce((n, ch) => n * 26 + ch.charCodeAt(0) - 64, 0) - 1;
    const top = Number(match[2]);
    const left = col(match[1] as string);
    const bottom = match[4] ? Number(match[4]) : top;
    const right = match[3] ? col(match[3]) : left;
    return {
      kind: "range",
      sheet,
      top: Math.min(top, bottom),
      left: Math.min(left, right),
      bottom: Math.max(top, bottom),
      right: Math.max(left, right),
    } satisfies RangeValue;
  },
  [1, 2],
);
define(
  "ADDRESS",
  (args, context) => {
    const row = integer(args[0], context);
    const column = integer(args[1], context);
    const kind = integer(args[2], context, 1);
    const a1 = bool(args[3], context, true);
    const sheet = args[4] && args[4].kind !== "missing" ? text(args[4], context) : null;
    if (row < 1 || column < 1) return err("#VALUE!");
    let letters = "";
    for (let n = column; n > 0; n = Math.floor((n - 1) / 26))
      letters = String.fromCharCode(65 + ((n - 1) % 26)) + letters;
    const fixedRow = kind === 1 || kind === 2;
    const fixedColumn = kind === 1 || kind === 3;
    const ref = a1
      ? `${fixedColumn ? "$" : ""}${letters}${fixedRow ? "$" : ""}${row}`
      : `R${fixedRow ? row : `[${row}]`}C${fixedColumn ? column : `[${column}]`}`;
    if (sheet === null) return ref;
    const quoted = /^[A-Za-z_][\w.]*$/.test(sheet) ? sheet : `'${sheet.replace(/'/g, "''")}'`;
    return `${quoted}!${ref}`;
  },
  [2, 5],
);
define(
  "TRANSPOSE",
  (args, context) => {
    const rows = rowsOf(arrayValue(args[0] as Node, context), context);
    return arrayOf((rows[0] ?? []).map((_, c) => rows.map((row) => row[c] ?? null)));
  },
  [1, 1],
);
define(
  "MMULT",
  (args, context) => {
    const a = rowsOf(arrayValue(args[0] as Node, context), context);
    const b = rowsOf(arrayValue(args[1] as Node, context), context);
    if ((a[0]?.length ?? 0) !== b.length) return err("#VALUE!");
    const out: Scalar[][] = [];
    for (const row of a) {
      const line: Scalar[] = [];
      for (let c = 0; c < (b[0]?.length ?? 0); c++) {
        let sum = 0;
        for (let k = 0; k < row.length; k++) {
          const x = row[k];
          const y = b[k]?.[c];
          if (typeof x !== "number" || typeof y !== "number") return err("#VALUE!");
          sum += x * y;
        }
        line.push(sum);
      }
      out.push(line);
    }
    return arrayOf(out);
  },
  [2, 2],
);
define(
  "FORMULATEXT",
  (args, context) => {
    const each = value(args[0] as Node, context);
    if (!isRange(each)) return err("#N/A");
    return context.calc.formulaText(each.sheet.name, each.top, each.left) ?? err("#N/A");
  },
  [1, 1],
);

// ---------------------------------------------------------------------------
// More of the corpus's functions

/** WEEKNUM's week start for each return type; 21 is ISO 8601. */
const WEEK_STARTS: Record<number, number> = {
  1: 0,
  2: 1,
  11: 1,
  12: 2,
  13: 3,
  14: 4,
  15: 5,
  16: 6,
  17: 0,
};
function isoWeek(serial: number, date1904: boolean): number {
  const { year, month, day } = partsOf(serial, date1904);
  const date = new Date(Date.UTC(year, month - 1, day));
  // The week's Thursday decides its year; week 1 holds the year's first Thursday.
  date.setUTCDate(date.getUTCDate() + 3 - ((date.getUTCDay() + 6) % 7));
  const thursday = date.getTime();
  const january4 = new Date(Date.UTC(date.getUTCFullYear(), 0, 4));
  january4.setUTCDate(january4.getUTCDate() + 3 - ((january4.getUTCDay() + 6) % 7));
  return 1 + Math.round((thursday - january4.getTime()) / (7 * DAY));
}
define(
  "WEEKNUM",
  (args, context) => {
    const date1904 = context.calc.book.date1904;
    const serial = Math.floor(dateSerial(args[0], context));
    const type = integer(args[1], context, 1);
    if (type === 21) return isoWeek(serial, date1904);
    const start = WEEK_STARTS[type];
    if (start === undefined) return err("#NUM!");
    const { year } = partsOf(serial, date1904);
    const january1 = serialOf(year, 1, 1, date1904);
    const weekdayOf = (s: number) =>
      new Date(Date.UTC(year, 0, 1) + (s - january1) * DAY).getUTCDay();
    const offset = (weekdayOf(january1) - start + 7) % 7;
    return Math.floor((serial - january1 + offset) / 7) + 1;
  },
  [1, 2],
);
define(
  "ISOWEEKNUM",
  (args, context) => isoWeek(Math.floor(dateSerial(args[0], context)), context.calc.book.date1904),
  [1, 1],
);

for (const [name, base] of [
  ["HEX2DEC", 16],
  ["BIN2DEC", 2],
  ["OCT2DEC", 8],
] as const) {
  define(
    name,
    (args, context) => {
      const digits = text(args[0], context).trim();
      if (
        digits === "" ||
        digits.length > 10 ||
        !new RegExp(`^[0-${base === 16 ? "9a-f" : base - 1}]+$`, "i").test(digits)
      )
        return err("#NUM!");
      const parsed = Number.parseInt(digits, base);
      const bits = base === 16 ? 40 : base === 8 ? 30 : 10;
      return digits.length === 10 && parsed >= 2 ** (bits - 1) ? parsed - 2 ** bits : parsed;
    },
    [1, 1],
  );
}
for (const [name, base] of [
  ["DEC2HEX", 16],
  ["DEC2BIN", 2],
  ["DEC2OCT", 8],
] as const) {
  define(
    name,
    (args, context) => {
      const x = Math.trunc(num(args[0], context));
      const bits = base === 16 ? 40 : base === 8 ? 30 : 10;
      if (x < -(2 ** (bits - 1)) || x >= 2 ** (bits - 1)) return err("#NUM!");
      let digits = (x < 0 ? 2 ** bits + x : x).toString(base).toUpperCase();
      if (args[1] && args[1].kind !== "missing") {
        const places = integer(args[1], context);
        if (places < digits.length) return err("#NUM!");
        digits = digits.padStart(places, "0");
      }
      return digits;
    },
    [1, 2],
  );
}
define(
  "DOLLAR",
  (args, context) => {
    const x = num(args[0], context);
    const decimals = integer(args[1], context, 2);
    const rounded = roundTo(Math.abs(x), decimals, "round");
    const [whole, fraction] = rounded.toFixed(Math.max(0, decimals)).split(".");
    const shown = `$${(whole as string).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}${fraction ? `.${fraction}` : ""}`;
    return x < 0 ? `(${shown})` : shown;
  },
  [1, 2],
);
define(
  "FREQUENCY",
  (args, context) => {
    const data = numbers([args[0] as Node], context);
    const bins = numbers([args[1] as Node], context);
    const sorted = bins.map((bin, index) => ({ bin, index })).sort((a, b) => a.bin - b.bin);
    const counts = new Array<number>(bins.length + 1).fill(0);
    for (const x of data) {
      const slot = sorted.find((each) => x <= each.bin);
      counts[slot ? slot.index : bins.length] = (counts[slot ? slot.index : bins.length] ?? 0) + 1;
    }
    return arrayOf(counts.map((count) => [count]));
  },
  [2, 2],
);
const AGGREGATES: Record<number, string> = {
  1: "AVERAGE",
  2: "COUNT",
  3: "COUNTA",
  4: "MAX",
  5: "MIN",
  6: "PRODUCT",
  7: "STDEV.S",
  8: "STDEV.P",
  9: "SUM",
  10: "VAR.S",
  11: "VAR.P",
  12: "MEDIAN",
  13: "MODE.SNGL",
};
define("AGGREGATE", (args, context) => {
  const code = integer(args[0], context);
  const option = integer(args[1], context, 0);
  const ignoreErrors = [2, 3, 6, 7].includes(option);
  const ignoreHidden = [1, 3, 5, 7].includes(option);
  const name =
    AGGREGATES[code] ??
    {
      14: "LARGE",
      15: "SMALL",
      16: "PERCENTILE.INC",
      17: "QUARTILE.INC",
      18: "PERCENTILE.EXC",
      19: "QUARTILE.EXC",
    }[code];
  const ignoreNested = option < 4;
  const fn = name ? FUNCTIONS.get(name) : undefined;
  if (!fn) return err("#VALUE!");
  const values: Scalar[] = [];
  const refs = code >= 14 ? [args[2] as Node] : args.slice(2);
  for (const ref of refs) {
    const each = value(ref, context);
    const take = (cell: BookCellLike) => {
      if (ignoreNested && /\b(SUBTOTAL|AGGREGATE)\(/i.test(cell.formula ?? "")) return;
      if (ignoreHidden && cell.hidden) return;
      if (ignoreErrors && isError(cell.value)) return;
      if (cell.value !== null) values.push(cell.value);
    };
    if (isRange(each)) {
      context.calc.eachKey(each.sheet, each, (key) => {
        const cell = each.sheet.cells.get(key);
        if (cell)
          take({
            value: cell.value,
            formula: cell.formula?.text,
            hidden: each.sheet.hiddenRows.has(cell.row),
          });
      });
    } else for (const cell of rowsOf(each, context).flat()) take({ value: cell });
  }
  const literal: Node = { kind: "array", rows: [values.map(toLiteral)] };
  return fn(code >= 14 ? [literal, args[3] as Node] : [literal], context);
});
interface BookCellLike {
  value: Scalar;
  formula?: string | undefined;
  hidden?: boolean;
}
function weekendMask(node: Node | undefined, context: Context): boolean[] {
  // Monday first, as Excel's weekend strings ("0000011") are.
  if (!node || node.kind === "missing") return [false, false, false, false, false, true, true];
  const given = check(scalar(node, context));
  if (typeof given === "string") {
    if (!/^[01]{7}$/.test(given)) return fail("#VALUE!");
    return [...given].map((ch) => ch === "1");
  }
  const code = Math.trunc(check(toNumber(given)));
  const mask = new Array<boolean>(7).fill(false);
  if (code >= 1 && code <= 7) {
    mask[(code + 4) % 7] = true;
    mask[(code + 5) % 7] = true;
  } else if (code >= 11 && code <= 17) mask[(code - 11 + 6) % 7] = true;
  else return fail("#NUM!");
  return mask;
}
const mondayIndex = (serial: number, date1904: boolean) =>
  (((Math.floor(serial) + (date1904 ? 1462 : 0) - 2) % 7) + 7) % 7;
define(
  "NETWORKDAYS.INTL",
  (args, context) => {
    const date1904 = context.calc.book.date1904;
    let start = Math.floor(num(args[0], context));
    let end = Math.floor(num(args[1], context));
    const mask = weekendMask(args[2], context);
    const off = holidays(args[3], context);
    let sign = 1;
    if (start > end) {
      [start, end] = [end, start];
      sign = -1;
    }
    let count = 0;
    for (let day = start; day <= end; day++)
      if (!mask[mondayIndex(day, date1904)] && !off.has(day)) count++;
    return sign * count;
  },
  [2, 4],
);
define(
  "WORKDAY.INTL",
  (args, context) => {
    const date1904 = context.calc.book.date1904;
    let day = Math.floor(num(args[0], context));
    let left = integer(args[1], context);
    const mask = weekendMask(args[2], context);
    if (mask.every(Boolean)) return err("#VALUE!");
    const off = holidays(args[3], context);
    const step = left < 0 ? -1 : 1;
    while (left !== 0) {
      day += step;
      if (!mask[mondayIndex(day, date1904)] && !off.has(day)) left -= step;
    }
    return day;
  },
  [2, 4],
);
function pairs(args: Node[], context: Context): [number, number][] {
  const ys = rowsOf(arrayValue(args[0] as Node, context), context).flat();
  const xs = rowsOf(arrayValue(args[1] as Node, context), context).flat();
  if (xs.length !== ys.length) return fail("#N/A");
  const found: [number, number][] = [];
  xs.forEach((x, index) => {
    const y = ys[index];
    if (typeof x === "number" && typeof y === "number") found.push([x, y]);
  });
  return found;
}
function line(points: [number, number][]): { slope: number; intercept: number } | null {
  if (points.length < 2) return null;
  const mx = mean(points.map((p) => p[0]));
  const my = mean(points.map((p) => p[1]));
  let sxy = 0;
  let sxx = 0;
  for (const [x, y] of points) {
    sxy += (x - mx) * (y - my);
    sxx += (x - mx) ** 2;
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}
define("SLOPE", (args, context) => line(pairs(args, context))?.slope ?? err("#DIV/0!"), [2, 2]);
define(
  "INTERCEPT",
  (args, context) => line(pairs(args, context))?.intercept ?? err("#DIV/0!"),
  [2, 2],
);
define(
  ["FORECAST", "FORECAST.LINEAR"],
  (args, context) => {
    const x = num(args[0], context);
    const fit = line(pairs(args.slice(1), context));
    return fit ? fit.intercept + fit.slope * x : err("#DIV/0!");
  },
  [3, 3],
);
