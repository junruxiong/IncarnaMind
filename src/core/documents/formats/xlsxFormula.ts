/**
 * Excel's formula language as a workbook stores it (en-US, comma separators,
 * no leading "="): parsed into a tree the workbook calculator evaluates, and
 * printed back with its relative references moved, which is how a shared
 * formula's followers get their own text (ECMA-376 §18.17).
 *
 * Spike (excel-edit): what isn't understood (3-D and external references,
 * the intersection and union operators) parses to an `unsupported` node, so
 * the calculator can say which cells it can't recalculate.
 */

export type ErrorCode =
  | "#NULL!"
  | "#DIV/0!"
  | "#VALUE!"
  | "#REF!"
  | "#NAME?"
  | "#NUM!"
  | "#N/A"
  | "#GETTING_DATA"
  | "#SPILL!"
  | "#CALC!";

const ERRORS: readonly ErrorCode[] = [
  "#NULL!",
  "#DIV/0!",
  "#VALUE!",
  "#REF!",
  "#NAME?",
  "#NUM!",
  "#N/A",
  "#GETTING_DATA",
  "#SPILL!",
  "#CALC!",
];

/** The largest row and column (0-based) a sheet has. */
export const MAX_ROW = 1_048_576;
export const MAX_COLUMN = 16_383;

/** A reference's corner: a row from 1 and a column from 0, each fixed ($) or relative. */
export interface Corner {
  row: number;
  column: number;
  rowFixed: boolean;
  columnFixed: boolean;
}

export type Node =
  | { kind: "number"; value: number }
  | { kind: "string"; value: string }
  | { kind: "boolean"; value: boolean }
  | { kind: "error"; code: ErrorCode }
  | { kind: "missing" }
  | { kind: "array"; rows: Node[][] }
  /** A cell or a range. `whole` says whether it was written as whole columns (A:C) or rows (1:3). */
  | {
      kind: "ref";
      sheet: string | null;
      from: Corner;
      to: Corner | null;
      whole: "columns" | "rows" | null;
    }
  | { kind: "name"; sheet: string | null; name: string }
  /** `Stock[Value]`, `Stock[[#This Row],[Qty]]`: the table, its items (#All…) and its columns. */
  | { kind: "table"; table: string; items: string[]; first: string | null; last: string | null }
  /** `name` as the calculator looks it up; `written` as the file has it ("_xlfn.XLOOKUP"). */
  | { kind: "call"; name: string; written: string; args: Node[] }
  | { kind: "unary"; op: "-" | "+"; operand: Node }
  | { kind: "percent"; operand: Node }
  | { kind: "binary"; op: BinaryOp; left: Node; right: Node }
  | { kind: "range"; left: Node; right: Node }
  | { kind: "unsupported"; reason: string; text: string };

export type BinaryOp = "+" | "-" | "*" | "/" | "^" | "&" | "=" | "<>" | "<" | ">" | "<=" | ">=";

export class FormulaSyntaxError extends Error {}

/** Parsed, but uses what the calculator doesn't do yet (a union, a 3-D reference…). */
class Unsupported extends Error {}

type Token =
  | { type: "number"; text: string }
  | { type: "string"; value: string }
  | { type: "error"; code: ErrorCode }
  | { type: "op"; value: string }
  | { type: "word"; text: string }
  | { type: "struct"; text: string }
  | { type: "end" };

const WORD =
  /^(?:'(?:[^']|'')+'|\[[^\]]*\][^!\s,()]*|[A-Za-z_\\\u0080-￿][\w.\\\u0080-￿]*)(?::(?:'(?:[^']|'')+'|[A-Za-z_\\\u0080-￿][\w.\\\u0080-￿]*))?!/;
const NAME = /^[A-Za-z_\\\u0080-￿$][\w.\\$\u0080-￿]*/;

function tokenize(formula: string): Token[] {
  const tokens: Token[] = [];
  let at = 0;
  const text = formula;
  while (at < text.length) {
    const ch = text[at] as string;
    if (ch === " " || ch === "\n" || ch === "\r" || ch === "\t") {
      at++;
      continue;
    }
    if (ch === '"') {
      let value = "";
      at++;
      for (;;) {
        if (at >= text.length) throw new FormulaSyntaxError("Unclosed string");
        if (text[at] === '"') {
          if (text[at + 1] === '"') {
            value += '"';
            at += 2;
            continue;
          }
          at++;
          break;
        }
        value += text[at];
        at++;
      }
      tokens.push({ type: "string", value });
      continue;
    }
    if (ch === "#") {
      const code = ERRORS.find((each) => text.startsWith(each, at));
      if (!code) throw new FormulaSyntaxError(`Unknown error literal at ${at}`);
      tokens.push({ type: "error", code });
      at += code.length;
      continue;
    }
    const number = /^(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/.exec(text.slice(at));
    if (number && !/^\d+:\d/.test(text.slice(at))) {
      // A row range ("1:3") is a reference, read below as a word.
      const after = text[at + number[0].length];
      if (!(after && /[A-Za-z_]/.test(after) && /^\d+$/.test(number[0]))) {
        tokens.push({ type: "number", text: number[0] });
        at += number[0].length;
        continue;
      }
    }
    const two = text.slice(at, at + 2);
    if (two === "<>" || two === "<=" || two === ">=") {
      tokens.push({ type: "op", value: two });
      at += 2;
      continue;
    }
    if ("+-*/^&=<>%(),:;{}".includes(ch)) {
      tokens.push({ type: "op", value: ch });
      at++;
      continue;
    }
    const external =
      /^\[\d+\](?:[^!\s,()]*|'(?:[^']|'')*')!(?:\$?[A-Za-z]{1,3}\$?\d*(?::\$?[A-Za-z]{1,3}\$?\d*)?|\$?\d+:\$?\d+|[A-Za-z_][\w.]*)/.exec(
        text.slice(at),
      );
    if (external) {
      // Another workbook's cells or names ("[1]Sheet1!A1", "[1]!Total"): kept, not calculated.
      tokens.push({ type: "word", text: external[0] });
      at += external[0].length;
      continue;
    }
    if (ch === "[") {
      // A structured reference with no table, as in a table's own formulas: [@Qty].
      const end = structEnd(text, at);
      tokens.push({ type: "struct", text: text.slice(at, end) });
      at = end;
      continue;
    }
    // A sheet prefix, then the reference or name after it, read as one word.
    const prefix = WORD.exec(text.slice(at));
    const start = at;
    if (prefix) at += prefix[0].length;
    if (prefix && text.startsWith("#REF!", at)) {
      at += 5;
      tokens.push({ type: "error", code: "#REF!" });
      continue;
    }
    const rowRange = /^\$?\d+:\$?\d+/.exec(text.slice(at));
    const name = rowRange ?? NAME.exec(text.slice(at));
    if (!name && !prefix) throw new FormulaSyntaxError(`Unexpected "${ch}" at ${at}`);
    at += name ? name[0].length : 0;
    // A range written as one word (A1:B2, A:C, $A$1:$B$9).
    const rest = /^:(\$?[A-Za-z]{1,3}\$?\d+|\$?[A-Za-z]{1,3}(?![\w(])|\$?\d+)(?![\w(!])/.exec(
      text.slice(at),
    );
    if (rest && name && /^\$?[A-Za-z]{0,3}\$?\d*$/.test(name[0]) && !rowRange) at += rest[0].length;
    if (text[at] === "[") {
      at = structEnd(text, at);
      tokens.push({ type: "struct", text: text.slice(start, at) });
      continue;
    }
    tokens.push({ type: "word", text: text.slice(start, at) });
  }
  tokens.push({ type: "end" });
  return tokens;
}

/** Where a structured reference's brackets close. */
function structEnd(text: string, at: number): number {
  let depth = 0;
  for (let index = at; index < text.length; index++) {
    const ch = text[index];
    if (ch === "'") {
      index++;
      continue;
    }
    if (ch === "[") depth++;
    else if (ch === "]") {
      depth--;
      if (depth === 0) return index + 1;
    }
  }
  throw new FormulaSyntaxError("Unclosed structured reference");
}

const columnIndex = (letters: string) => {
  let column = 0;
  for (const letter of letters.toUpperCase()) column = column * 26 + (letter.charCodeAt(0) - 64);
  return column - 1;
};

const unquoteSheet = (sheet: string) =>
  sheet.startsWith("'") ? sheet.slice(1, -1).replace(/''/g, "'") : sheet;

/** "A1", "$B$2" → a corner; a column alone ("C") or a row alone ("3") for whole ranges. */
function cornerOf(text: string): { corner: Corner; part: "cell" | "column" | "row" } | null {
  const cell = /^(\$?)([A-Za-z]{1,3})(\$?)(\d+)$/.exec(text);
  if (cell) {
    const column = columnIndex(cell[2] as string);
    const row = Number(cell[4]);
    if (column > MAX_COLUMN || row < 1 || row > MAX_ROW) return null;
    return {
      corner: { row, column, rowFixed: cell[3] === "$", columnFixed: cell[1] === "$" },
      part: "cell",
    };
  }
  const column = /^(\$?)([A-Za-z]{1,3})$/.exec(text);
  if (column) {
    const index = columnIndex(column[2] as string);
    if (index > MAX_COLUMN) return null;
    return {
      corner: { row: 1, column: index, rowFixed: true, columnFixed: column[1] === "$" },
      part: "column",
    };
  }
  const row = /^(\$?)(\d+)$/.exec(text);
  if (row) {
    const index = Number(row[2]);
    if (index < 1 || index > MAX_ROW) return null;
    return {
      corner: { row: index, column: 0, rowFixed: row[1] === "$", columnFixed: true },
      part: "row",
    };
  }
  return null;
}

/** A word token as a reference, a name, or a function's name (when a bracket follows). */
function wordNode(text: string): Node {
  let sheet: string | null = null;
  let body = text;
  const bang = text.lastIndexOf("!");
  if (bang >= 0) {
    const prefix = text.slice(0, bang);
    body = text.slice(bang + 1);
    if (prefix.startsWith("[") || /^'?\[/.test(prefix) || body === "") {
      return { kind: "unsupported", reason: "external reference", text };
    }
    const unquoted = unquoteSheet(prefix);
    if (/^[^']+:[^']+$/.test(prefix) || (prefix.startsWith("'") && unquoted.includes(":"))) {
      return { kind: "unsupported", reason: "3-D reference", text };
    }
    sheet = unquoted;
  }
  const [first, second] = body.split(":");
  const a = first ? cornerOf(first) : null;
  if (a && second === undefined && a.part === "cell") {
    return { kind: "ref", sheet, from: a.corner, to: null, whole: null };
  }
  if (a && second !== undefined) {
    const b = cornerOf(second);
    if (b && a.part === b.part) {
      if (a.part === "cell")
        return { kind: "ref", sheet, from: a.corner, to: b.corner, whole: null };
      if (a.part === "column") {
        return {
          kind: "ref",
          sheet,
          from: a.corner,
          to: { ...b.corner, row: MAX_ROW },
          whole: "columns",
        };
      }
      return {
        kind: "ref",
        sheet,
        from: a.corner,
        to: { ...b.corner, column: MAX_COLUMN },
        whole: "rows",
      };
    }
  }
  if (body.toUpperCase() === "TRUE" || body.toUpperCase() === "FALSE") {
    return { kind: "boolean", value: body.toUpperCase() === "TRUE" };
  }
  if (body.includes(":")) return { kind: "unsupported", reason: "range of names", text };
  return { kind: "name", sheet, name: body };
}

/** "Stock[[#This Row],[Qty]]" → its table, items and columns. */
function tableNode(text: string, currentTable: string | null): Node {
  const open = text.indexOf("[");
  const table = text.slice(0, open) || currentTable;
  if (!table) return { kind: "unsupported", reason: "structured reference outside a table", text };
  let inner = text.slice(open + 1, -1).trim();
  const items: string[] = [];
  const columns: string[] = [];
  if (inner.startsWith("@")) {
    items.push("#This Row");
    inner = inner.slice(1).trim();
    if (inner.startsWith("[")) inner = inner.slice(1, -1);
    if (inner) columns.push(unescapeColumn(inner));
  } else if (!inner.startsWith("[")) {
    if (inner.startsWith("#")) items.push(inner);
    else if (inner) columns.push(unescapeColumn(inner));
  } else {
    // [[#Headers],[Qty]:[Unit]] — a list of bracketed parts, the columns perhaps a range.
    const parts = splitTopLevel(inner);
    for (const part of parts) {
      const trimmed = part.trim();
      const range = /^\[((?:[^\]']|'.)*)\]\s*:\s*\[((?:[^\]']|'.)*)\]$/.exec(trimmed);
      if (range) {
        columns.push(unescapeColumn(range[1] as string), unescapeColumn(range[2] as string));
        continue;
      }
      const name = trimmed.replace(/^\[|\]$/g, "");
      if (name.startsWith("#")) items.push(name);
      else if (name === "@") items.push("#This Row");
      else columns.push(unescapeColumn(name));
    }
  }
  return {
    kind: "table",
    table,
    items: items.map((item) => item.replace(/\s+/g, " ")),
    first: columns[0] ?? null,
    last: columns[1] ?? columns[0] ?? null,
  };
}

const unescapeColumn = (name: string) => name.replace(/'(.)/g, "$1");

function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < text.length; index++) {
    const ch = text[index];
    if (ch === "'") {
      index++;
      continue;
    }
    if (ch === "[") depth++;
    else if (ch === "]") depth--;
    else if (ch === "," && depth === 0) {
      parts.push(text.slice(start, index));
      start = index + 1;
    }
  }
  parts.push(text.slice(start));
  return parts;
}

const BINARY_POWER: Record<string, number> = {
  "=": 1,
  "<>": 1,
  "<": 1,
  ">": 1,
  "<=": 1,
  ">=": 1,
  "&": 2,
  "+": 3,
  "-": 3,
  "*": 4,
  "/": 4,
  "^": 5,
};
const PREFIX_POWER = 6;
const PERCENT_POWER = 7;
const RANGE_POWER = 8;

/**
 * Parses a formula as a workbook stores it, without its leading "=". Inside a
 * table, `table` names it, for references such as `[@Qty]`.
 */
export function parseFormula(formula: string, table: string | null = null): Node {
  const tokens = tokenize(formula.startsWith("=") ? formula.slice(1) : formula);
  let at = 0;
  const peek = () => tokens[at] as Token;
  const next = () => tokens[at++] as Token;
  const isOp = (token: Token, value: string) => token.type === "op" && token.value === value;

  function expression(minPower: number): Node {
    let left = prefix();
    for (;;) {
      const token = peek();
      if (token.type !== "op") break;
      if (token.value === "%" && PERCENT_POWER >= minPower) {
        next();
        left = { kind: "percent", operand: left };
        continue;
      }
      if (token.value === ":" && RANGE_POWER >= minPower) {
        next();
        const right = expression(RANGE_POWER + 1);
        left = { kind: "range", left, right };
        continue;
      }
      const power = BINARY_POWER[token.value];
      if (power === undefined || power < minPower) break;
      next();
      const right = expression(power + 1);
      left = { kind: "binary", op: token.value as BinaryOp, left, right };
    }
    return left;
  }

  function prefix(): Node {
    const token = next();
    switch (token.type) {
      case "number":
        return { kind: "number", value: Number(token.text) };
      case "string":
        return { kind: "string", value: token.value };
      case "error":
        return { kind: "error", code: token.code };
      case "struct":
        return tableNode(token.text, table);
      case "word": {
        if (isOp(peek(), "(")) {
          next();
          const written = token.text;
          return { kind: "call", name: functionName(written), written, args: argumentsList() };
        }
        return wordNode(token.text);
      }
      case "op": {
        if (token.value === "-" || token.value === "+") {
          return { kind: "unary", op: token.value, operand: expression(PREFIX_POWER) };
        }
        if (token.value === "(") {
          const inner = expression(0);
          if (isOp(peek(), ",")) throw new Unsupported("union");
          expect(")");
          return inner;
        }
        if (token.value === "{") return arrayLiteral();
        throw new FormulaSyntaxError(`Unexpected "${token.value}"`);
      }
      case "end":
        throw new FormulaSyntaxError("The formula ends too early");
    }
  }

  function argumentsList(): Node[] {
    const args: Node[] = [];
    if (isOp(peek(), ")")) {
      next();
      return args;
    }
    for (;;) {
      const token = peek();
      if (isOp(token, ",") || isOp(token, ")")) args.push({ kind: "missing" });
      else args.push(expression(0));
      const after = next();
      if (after.type === "op" && after.value === ")") return args;
      if (!(after.type === "op" && after.value === ",")) {
        throw new FormulaSyntaxError("Expected , or ) in a function's arguments");
      }
    }
  }

  function arrayLiteral(): Node {
    const rows: Node[][] = [[]];
    for (;;) {
      const token = next();
      let value: Node;
      if (token.type === "op" && (token.value === "-" || token.value === "+")) {
        const number = next();
        if (number.type !== "number") throw new FormulaSyntaxError("Bad array constant");
        value = { kind: "number", value: (token.value === "-" ? -1 : 1) * Number(number.text) };
      } else if (token.type === "number") value = { kind: "number", value: Number(token.text) };
      else if (token.type === "string") value = { kind: "string", value: token.value };
      else if (token.type === "error") value = { kind: "error", code: token.code };
      else if (token.type === "word" && /^(TRUE|FALSE)$/i.test(token.text)) {
        value = { kind: "boolean", value: token.text.toUpperCase() === "TRUE" };
      } else throw new FormulaSyntaxError("Bad array constant");
      (rows[rows.length - 1] as Node[]).push(value);
      const separator = next();
      if (separator.type !== "op") throw new FormulaSyntaxError("Bad array constant");
      if (separator.value === "}") return { kind: "array", rows };
      if (separator.value === ";") rows.push([]);
      else if (separator.value !== ",") throw new FormulaSyntaxError("Bad array constant");
    }
  }

  function expect(value: string) {
    const token = next();
    if (!(token.type === "op" && token.value === value)) {
      throw new FormulaSyntaxError(`Expected "${value}"`);
    }
  }

  try {
    const node = expression(0);
    if (peek().type !== "end") throw new FormulaSyntaxError("Unexpected text after the formula");
    return node;
  } catch (error) {
    if (error instanceof Unsupported)
      return { kind: "unsupported", reason: error.message, text: formula };
    throw error;
  }
}

/** A function's name as the calculator looks it up: upper case, without Excel's _xlfn. prefixes. */
export function functionName(name: string): string {
  return name.replace(/^(?:_xlfn\.|_xlws\.|_xll\.)+/i, "").toUpperCase();
}

// ---------------------------------------------------------------------------
// Printing, with relative references moved

const quoteSheet = (sheet: string) =>
  /^[A-Za-z_][\w.]*$/.test(sheet) && !/^[A-Za-z]{1,3}\d+$/.test(sheet)
    ? sheet
    : `'${sheet.replace(/'/g, "''")}'`;

const columnLetters = (index: number) => {
  let name = "";
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) {
    name = String.fromCharCode(65 + ((n - 1) % 26)) + name;
  }
  return name;
};

/** A corner moved by a cell's offset from the formula's own cell; null when it falls off the sheet. */
export function moveCorner(corner: Corner, rows: number, columns: number): Corner | null {
  const row = corner.rowFixed ? corner.row : corner.row + rows;
  const column = corner.columnFixed ? corner.column : corner.column + columns;
  if (row < 1 || row > MAX_ROW || column < 0 || column > MAX_COLUMN) return null;
  return { ...corner, row, column };
}

const PRINT_POWER: Record<string, number> = { ...BINARY_POWER };

/** A formula's text, its relative references moved by `rows` and `columns`. */
export function printFormula(node: Node, rows = 0, columns = 0): string {
  const print = (each: Node, minPower = 0): string => {
    switch (each.kind) {
      case "number":
        return String(each.value);
      case "string":
        return `"${each.value.replace(/"/g, '""')}"`;
      case "boolean":
        return each.value ? "TRUE" : "FALSE";
      case "error":
        return each.code;
      case "missing":
        return "";
      case "array":
        return `{${each.rows.map((row) => row.map((value) => print(value)).join(",")).join(";")}}`;
      case "ref": {
        const prefix = each.sheet === null ? "" : `${quoteSheet(each.sheet)}!`;
        const from = moveCorner(each.from, rows, columns);
        const to = each.to ? moveCorner(each.to, rows, columns) : null;
        if (!from || (each.to && !to)) return `${prefix}#REF!`;
        const cell = (corner: Corner) =>
          `${corner.columnFixed ? "$" : ""}${columnLetters(corner.column)}${corner.rowFixed ? "$" : ""}${corner.row}`;
        if (each.whole === "columns" && to) {
          return `${prefix}${from.columnFixed ? "$" : ""}${columnLetters(from.column)}:${to.columnFixed ? "$" : ""}${columnLetters(to.column)}`;
        }
        if (each.whole === "rows" && to) {
          return `${prefix}${from.rowFixed ? "$" : ""}${from.row}:${to.rowFixed ? "$" : ""}${to.row}`;
        }
        return `${prefix}${cell(from)}${to ? `:${cell(to)}` : ""}`;
      }
      case "name":
        return each.sheet === null ? each.name : `${quoteSheet(each.sheet)}!${each.name}`;
      case "table": {
        const escapeColumn = (column: string) => column.replace(/(['#[\]])/g, "'$1");
        const columnsPart =
          each.first === null
            ? []
            : [
                each.last && each.last !== each.first
                  ? `[${escapeColumn(each.first)}]:[${escapeColumn(each.last)}]`
                  : `[${escapeColumn(each.first)}]`,
              ];
        const parts = [...each.items.map((item) => `[${item}]`), ...columnsPart];
        if (each.items.length === 0 && columnsPart.length === 1 && each.first === each.last) {
          return `${each.table}[${escapeColumn(each.first ?? "")}]`;
        }
        return `${each.table}[${parts.join(",")}]`;
      }
      case "call":
        return `${each.written}(${each.args.map((arg) => print(arg)).join(",")})`;
      case "unary":
        return `${each.op}${print(each.operand, PREFIX_POWER)}`;
      case "percent":
        return `${print(each.operand, PERCENT_POWER)}%`;
      case "range":
        return `${print(each.left, RANGE_POWER)}:${print(each.right, RANGE_POWER + 1)}`;
      case "binary": {
        const power = PRINT_POWER[each.op] as number;
        const text = `${print(each.left, power)}${each.op}${print(each.right, power + 1)}`;
        return power < minPower ? `(${text})` : text;
      }
      case "unsupported":
        return each.text;
    }
  };
  return print(node);
}
