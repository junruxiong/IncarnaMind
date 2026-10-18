/**
 * A workbook as the calculator needs it (spike: excel-edit): every sheet's
 * cells with their stored values and formulas, shared formulas tied to the
 * cell that holds their text, the defined names and the tables. Read from
 * the package with the readers the preview uses; nothing is written here
 * (./xlsxCopy writes the copy).
 */
import { parseCellRef } from "../../../shared/units";
import { ExtractionError } from "./errors";
import { builtInFormat } from "./numbers";
import { relationshipsOf } from "./xlsxDrawings";
import { type ErrorCode, type Node, parseFormula } from "./xlsxFormula";
import {
  child,
  descendants,
  elements,
  is,
  parseXml,
  streamElements,
  textOf,
  type XmlElement,
} from "./xml";
import { openPackage, type ZipArchive } from "./zip";

const MAX_STREAMED_BYTES = 256 * 1024 * 1024;

export interface CalcError {
  error: ErrorCode;
}

/** A cell's value: a number, text, TRUE or FALSE, an error, or nothing (null). */
export type Scalar = number | string | boolean | CalcError | null;

export const isError = (value: unknown): value is CalcError =>
  typeof value === "object" && value !== null && "error" in value;

export interface FormulaCell {
  /** The formula as the cell would show it, without "=". */
  text: string;
  /** Parsed, with references as written in the cell that holds the text. */
  node: Node | null;
  /** The cell the text is written relative to: itself, or a shared formula's first cell. */
  origin: { row: number; column: number };
  /** A shared formula's group, and whether this cell holds its text. */
  shared?: { index: number; anchor: boolean };
  /** An array formula's range ("D2:D9"), on its first cell. */
  array?: string;
  /** Excel's what-if data table, which the calculator doesn't do. */
  dataTable?: boolean;
  /** Why it couldn't be parsed. */
  problem?: string;
}

export interface BookCell {
  row: number;
  /** From 0. */
  column: number;
  value: Scalar;
  /** The cell's style index (`s`). */
  style: number;
  formula?: FormulaCell;
  /** Inside an array formula's range, not its first cell: the first cell's key. */
  arrayPart?: number;
}

export interface BookTable {
  name: string;
  sheet: BookSheet;
  top: number;
  left: number;
  bottom: number;
  right: number;
  headerRows: number;
  totalsRows: number;
  columns: string[];
  filtered: boolean;
}

export interface BookSheet {
  name: string;
  /** Its place among the workbook's sheets, hidden ones included: what a local name's `localSheetId` counts. */
  index: number;
  part: string;
  hidden: boolean;
  /** By `cellKey`. */
  cells: Map<number, BookCell>;
  tables: BookTable[];
  /** Rows hidden, by a filter or by hand. */
  hiddenRows: Set<number>;
  /** Whether it has a filter, whose hidden rows SUBTOTAL leaves out whatever its function. */
  filtered: boolean;
}

export interface DefinedName {
  name: string;
  /** The sheet a local name belongs to, else null. */
  sheet: number | null;
  text: string;
  node: Node | null;
}

export interface Book {
  sheets: BookSheet[];
  names: DefinedName[];
  tables: Map<string, BookTable>;
  date1904: boolean;
  /**
   * What the file stores for its formulas are placeholders, not results: it asks Excel to
   * recalculate everything on opening (`calcPr fullCalcOnLoad`, as XlsxWriter does), or every
   * stored result is empty or 0 (ExcelJS, openpyxl), as no workbook Excel saved has.
   */
  placeholders: boolean;
  /** A cell's number format code, by style index. */
  formats: (string | undefined)[];
}

export const cellKey = (row: number, column: number) => row * 16_384 + column;
export const keyRow = (key: number) => Math.floor(key / 16_384);
export const keyColumn = (key: number) => key % 16_384;

const sheetKind = (type: string) => type.endsWith("/worksheet");

/** Reads a workbook's sheets, names and tables for calculating. */
export async function readBook(bytes: Uint8Array): Promise<Book> {
  const zip = openPackage(bytes);
  const workbookXml = await zip.readText("xl/workbook.xml");
  if (workbookXml === undefined) {
    throw new ExtractionError("unreadable", "Not an Excel workbook: it has no xl/workbook.xml.");
  }
  const workbook = parseXml(workbookXml);
  const date1904 = ["1", "true"].includes(child(workbook, "workbookPr")?.attrs.date1904 ?? "");
  const asksToRecalculate = ["1", "true"].includes(
    child(workbook, "calcPr")?.attrs.fullCalcOnLoad ?? "",
  );
  const rels = await relationshipsOf(zip, "xl/workbook.xml");

  const formats: (string | undefined)[] = [];
  const stylesXml = await zip.readText("xl/styles.xml");
  if (stylesXml) {
    const styles = parseXml(stylesXml);
    const custom = new Map<number, string>();
    for (const format of descendants(styles, "numFmt")) {
      custom.set(Number(format.attrs.numFmtId), format.attrs.formatCode ?? "");
    }
    const xfs = child(styles, "cellXfs");
    for (const xf of xfs ? elements(xfs) : []) {
      const id = Number(xf.attrs.numFmtId ?? 0);
      formats.push(custom.get(id) ?? builtInFormat(id));
    }
  }

  const shared: string[] = [];
  if (zip.has("xl/sharedStrings.xml")) {
    for await (const si of streamElements(
      zip.textChunks("xl/sharedStrings.xml", MAX_STREAMED_BYTES),
      ["si"],
    )) {
      shared.push(stringItem(si));
    }
  }

  const sheets: BookSheet[] = [];
  for (const [index, entry] of descendants(workbook, "sheet").entries()) {
    const rel = rels.get(entry.attrs["r:id"] ?? "");
    const sheet: BookSheet = {
      name: entry.attrs.name ?? `Sheet${index + 1}`,
      index,
      part: rel?.target ?? "",
      hidden: entry.attrs.state === "hidden" || entry.attrs.state === "veryHidden",
      cells: new Map(),
      tables: [],
      hiddenRows: new Set(),
      filtered: false,
    };
    sheets.push(sheet);
    if (!rel || !sheetKind(rel.type) || !zip.has(rel.target)) continue;
    await readSheetCells(zip, sheet, shared);
  }

  const tables = new Map<string, BookTable>();
  for (const sheet of sheets) {
    if (!sheet.part) continue;
    for (const rel of (await relationshipsOf(zip, sheet.part)).values()) {
      if (!rel.type.endsWith("/table") || !zip.has(rel.target)) continue;
      const xml = await zip.readText(rel.target);
      if (!xml) continue;
      const table = readTable(parseXml(xml), sheet);
      if (table) {
        if (table.filtered) sheet.filtered = true;
        sheet.tables.push(table);
        tables.set(table.name.toLowerCase(), table);
      }
    }
  }

  const names: DefinedName[] = [];
  for (const defined of descendants(workbook, "definedName")) {
    const name = defined.attrs.name;
    if (!name) continue;
    const text = textOf(defined);
    let node: Node | null = null;
    try {
      node = parseFormula(text);
    } catch {
      node = null;
    }
    const local = defined.attrs.localSheetId;
    names.push({ name, sheet: local === undefined ? null : Number(local), text, node });
  }

  // A table's own formulas may leave their table unnamed ("[@Qty]"): parse them knowing it.
  for (const table of tables.values()) {
    for (const cell of table.sheet.cells.values()) {
      const formula = cell.formula;
      if (!formula?.text.includes("[")) continue;
      if (cell.row < table.top || cell.row > table.bottom) continue;
      if (cell.column < table.left || cell.column > table.right) continue;
      const parsed = parse(formula.text, table.name);
      formula.node = parsed.node;
      if (parsed.problem) formula.problem = parsed.problem;
      else delete formula.problem;
    }
  }
  const results = sheets.flatMap((sheet) =>
    [...sheet.cells.values()].flatMap((cell) => (cell.formula ? [cell.value] : [])),
  );
  const placeholders =
    asksToRecalculate ||
    (results.length > 0 && results.every((value) => value === null || value === 0 || value === ""));
  return { sheets, names, tables, date1904, placeholders, formats };
}

const stringItem = (si: XmlElement) =>
  elements(si)
    .filter((each) => is(each, "t") || is(each, "r"))
    .map((each) => (is(each, "t") ? textOf(each) : textOf(child(each, "t") ?? each)))
    .join("");

async function readSheetCells(zip: ZipArchive, sheet: BookSheet, shared: readonly string[]) {
  const anchors = new Map<
    number,
    { row: number; column: number; text: string; node: Node | null; problem?: string }
  >();
  const followers: BookCell[] = [];
  const arrays: { key: number; ref: string }[] = [];
  let rowNumber = 0;
  for await (const row of streamElements(zip.textChunks(sheet.part, MAX_STREAMED_BYTES), [
    "row",
    "autoFilter",
  ])) {
    if (is(row, "autoFilter")) {
      if (descendants(row, "filterColumn").length > 0) sheet.filtered = true;
      continue;
    }
    rowNumber = row.attrs.r ? Number(row.attrs.r) : rowNumber + 1;
    if (row.attrs.hidden === "1" || row.attrs.hidden === "true") sheet.hiddenRows.add(rowNumber);
    let column = -1;
    for (const c of elements(row)) {
      if (!is(c, "c")) continue;
      const ref = c.attrs.r ? parseCellRef(c.attrs.r) : null;
      column = ref ? ref.column : column + 1;
      const cell: BookCell = {
        row: rowNumber,
        column,
        value: storedValue(c, shared),
        style: Number(c.attrs.s ?? 0),
      };
      const f = child(c, "f");
      if (f) {
        const text = textOf(f);
        const kind = f.attrs.t ?? "normal";
        const formula: FormulaCell = { text, node: null, origin: { row: rowNumber, column } };
        if (kind === "shared" && f.attrs.si !== undefined) {
          const index = Number(f.attrs.si);
          const anchor = f.attrs.ref !== undefined && text !== "";
          formula.shared = { index, anchor };
          if (anchor) {
            const parsed = parse(text);
            formula.node = parsed.node;
            if (parsed.problem) formula.problem = parsed.problem;
            anchors.set(index, { row: rowNumber, column, text, ...parsed });
          } else followers.push(cell);
        } else if (kind === "dataTable") {
          formula.dataTable = true;
        } else {
          if (kind === "array" && f.attrs.ref) {
            formula.array = f.attrs.ref;
            arrays.push({ key: cellKey(rowNumber, column), ref: f.attrs.ref });
          }
          const parsed = parse(text);
          formula.node = parsed.node;
          if (parsed.problem) formula.problem = parsed.problem;
        }
        cell.formula = formula;
      }
      if (cell.value !== null || cell.formula || cell.style > 0) {
        sheet.cells.set(cellKey(rowNumber, column), cell);
      }
    }
  }
  for (const cell of followers) {
    const formula = cell.formula as FormulaCell;
    const anchor = anchors.get(formula.shared?.index ?? -1);
    if (!anchor) {
      formula.problem = "shared formula without its first cell";
      continue;
    }
    formula.origin = { row: anchor.row, column: anchor.column };
    formula.node = anchor.node;
    formula.text = anchor.text;
    if (anchor.problem) formula.problem = anchor.problem;
  }
  for (const { key, ref } of arrays) {
    const [from, to] = ref.split(":");
    const a = parseCellRef(from ?? "");
    const b = parseCellRef(to ?? from ?? "");
    if (!a || !b) continue;
    for (let row = a.row; row <= b.row; row++) {
      for (let column = a.column; column <= b.column; column++) {
        const at = cellKey(row, column);
        if (at === key) continue;
        const cell = sheet.cells.get(at) ?? { row, column, value: null, style: 0 };
        cell.arrayPart = key;
        sheet.cells.set(at, cell);
      }
    }
  }
}

function parse(text: string, table: string | null = null): { node: Node | null; problem?: string } {
  try {
    return { node: parseFormula(text, table) };
  } catch (error) {
    return { node: null, problem: error instanceof Error ? error.message : String(error) };
  }
}

const ERROR_CODES = new Set([
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
]);

/** A cell's stored value: its last result, for a formula. */
function storedValue(c: XmlElement, shared: readonly string[]): Scalar {
  const type = c.attrs.t ?? "n";
  const v = child(c, "v");
  const raw = v ? textOf(v) : "";
  switch (type) {
    case "s":
      return v ? (shared[Number(raw)] ?? "") : null;
    case "inlineStr":
      return stringItem(child(c, "is") ?? c);
    case "str":
      return v ? raw : null;
    case "b":
      return v ? raw === "1" || raw === "true" : null;
    case "e":
      return ERROR_CODES.has(raw) ? { error: raw as ErrorCode } : { error: "#VALUE!" };
    case "d":
      return v ? raw : null;
    default: {
      if (!v || raw === "") return null;
      const number = Number(raw);
      return Number.isFinite(number) ? number : raw;
    }
  }
}

function readTable(xml: XmlElement, sheet: BookSheet): BookTable | null {
  const root = xml;
  const ref = root.attrs.ref;
  const name = root.attrs.displayName ?? root.attrs.name;
  if (!ref || !name) return null;
  const [from, to] = ref.split(":");
  const a = parseCellRef(from ?? "");
  const b = parseCellRef(to ?? from ?? "");
  if (!a || !b) return null;
  // Names escape characters XML can't hold as _xHHHH_ (a line break is _x000a_).
  const columns = descendants(root, "tableColumn").map((column) =>
    (column.attrs.name ?? "").replace(/_x([0-9A-Fa-f]{4})_/g, (_, hex: string) =>
      String.fromCharCode(Number.parseInt(hex, 16)),
    ),
  );
  return {
    name,
    sheet,
    top: a.row,
    left: a.column,
    bottom: b.row,
    right: b.column,
    headerRows: Number(root.attrs.headerRowCount ?? 1),
    totalsRows: Number(root.attrs.totalsRowCount ?? 0),
    columns,
    filtered: descendants(root, "filterColumn").length > 0,
  };
}
