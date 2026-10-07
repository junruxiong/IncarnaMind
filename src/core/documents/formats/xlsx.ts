/**
 * Excel (.xlsx) → sheets of cells, and Units of blocks of rows (see ./grid).
 *
 * Values are shown as Excel shows them: formulas give their last cached
 * result, numbers follow their cell's number format ("£350,200", "4.2%"),
 * dates become ISO dates, booleans TRUE and FALSE. Hidden sheets are left
 * out. Sheets are read as a stream of rows, so a huge sheet costs no more
 * than the rows read: past `MAX_ROWS` rows or `MAX_CELLS` cells in all,
 * reading stops and the workbook says it was cut short. Ported from the
 * office-formats spike, with streaming and number formats added.
 */
import { parseCellRef, type TextUnit } from "../../../shared/units";
import { ExtractionError } from "./errors";
import { type Cell, gridUnits, MAX_CELLS, MAX_ROWS, type Sheet, type Workbook } from "./grid";
import { builtInFormat, formatNumber, generalNumber, isDateFormat } from "./numbers";
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
import { openPackage, resolvePart } from "./zip";

/** The largest a sheet or the shared strings may inflate to while being read. */
const MAX_STREAMED_BYTES = 256 * 1024 * 1024;

/** A rich-text string item's text: plain <t>, or runs <r><t>; phonetic hints (<rPh>) aren't text. */
const stringItem = (si: XmlElement) =>
  elements(si)
    .filter((each) => is(each, "t") || is(each, "r"))
    .map((each) => (is(each, "t") ? textOf(each) : textOf(child(each, "t") ?? each)))
    .join("");

export interface Limits {
  rows: number;
  cells: number;
}

export async function readWorkbook(
  bytes: Uint8Array,
  limits: Limits = { rows: MAX_ROWS, cells: MAX_CELLS },
): Promise<Workbook> {
  const zip = openPackage(bytes);
  const workbookXml = await zip.readText("xl/workbook.xml");
  if (workbookXml === undefined) {
    throw new ExtractionError("unreadable", "Not an Excel workbook: it has no xl/workbook.xml.");
  }
  const workbook = parseXml(workbookXml);
  const date1904 = ["1", "true"].includes(child(workbook, "workbookPr")?.attrs.date1904 ?? "");

  const relsXml = await zip.readText("xl/_rels/workbook.xml.rels");
  const targets = new Map<string, string>();
  for (const rel of relsXml ? elements(parseXml(relsXml)) : []) {
    if (rel.attrs.Id && rel.attrs.Target) {
      targets.set(rel.attrs.Id, resolvePart("xl/workbook.xml", rel.attrs.Target));
    }
  }

  const shared: string[] = [];
  if (zip.has("xl/sharedStrings.xml")) {
    const chunks = zip.textChunks("xl/sharedStrings.xml", MAX_STREAMED_BYTES);
    for await (const si of streamElements(chunks, ["si"])) shared.push(stringItem(si));
  }

  // The number format of each cell style (by its index in cellXfs).
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

  const sheets: Sheet[] = [];
  let rowsLeft = limits.rows;
  let cellsLeft = limits.cells;
  let truncated = false;
  for (const entry of descendants(workbook, "sheet")) {
    if (entry.attrs.state === "hidden" || entry.attrs.state === "veryHidden") continue;
    const part = targets.get(entry.attrs["r:id"] ?? "");
    if (!part || !zip.has(part)) continue;
    const sheet: Sheet = {
      name: entry.attrs.name ?? `Sheet${sheets.length + 1}`,
      cells: [],
      merges: [],
      rowCount: 0,
      columnCount: 0,
      columnWidths: [],
    };
    sheets.push(sheet);
    if (truncated) continue;
    let rowNumber = 0;
    const chunks = zip.textChunks(part, MAX_STREAMED_BYTES);
    for await (const element of streamElements(chunks, ["col", "row", "mergeCell"])) {
      if (is(element, "col")) {
        const width = Number(element.attrs.width);
        const min = Number(element.attrs.min);
        const max = Math.min(Number(element.attrs.max), min + 255);
        if (Number.isFinite(width) && min >= 1) {
          for (let column = min; column <= max; column++) {
            sheet.columnWidths[column - 1] = Math.round(width * 7 + 5);
          }
        }
        continue;
      }
      if (is(element, "mergeCell")) {
        if (element.attrs.ref) sheet.merges.push(element.attrs.ref);
        continue;
      }
      rowNumber = element.attrs.r ? Number(element.attrs.r) : rowNumber + 1;
      const cells = elements(element).filter((each) => is(each, "c"));
      if (cells.length === 0) continue;
      if (rowsLeft <= 0 || cellsLeft < cells.length) {
        truncated = true;
        break;
      }
      rowsLeft--;
      let column = -1;
      for (const c of cells) {
        const ref = c.attrs.r ? parseCellRef(c.attrs.r) : null;
        column = ref ? ref.column : column + 1;
        const cell = readCell(c, shared, formats, date1904);
        if (!cell) continue;
        cellsLeft--;
        sheet.cells.push({ row: rowNumber, column, ...cell });
        sheet.rowCount = Math.max(sheet.rowCount, rowNumber);
        sheet.columnCount = Math.max(sheet.columnCount, column + 1);
      }
    }
  }
  // Merged ranges stretch the grid even where they hold nothing.
  for (const sheet of sheets) {
    for (const merge of sheet.merges) {
      const end = parseCellRef(merge.split(":")[1] ?? "");
      if (!end) continue;
      sheet.rowCount = Math.max(sheet.rowCount, end.row);
      sheet.columnCount = Math.max(sheet.columnCount, end.column + 1);
    }
  }
  return { sheets, truncated };
}

/** A cell's value as shown, or null for an empty one. */
function readCell(
  c: XmlElement,
  shared: readonly string[],
  formats: readonly (string | undefined)[],
  date1904: boolean,
): Pick<Cell, "value" | "numeric"> | null {
  const type = c.attrs.t ?? "n";
  const v = child(c, "v");
  const raw = v ? textOf(v) : "";
  let value: string;
  let numeric = false;
  if (type === "s") value = shared[Number(raw)] ?? "";
  else if (type === "inlineStr") value = stringItem(child(c, "is") ?? c);
  else if (type === "b") value = raw === "1" ? "TRUE" : "FALSE";
  else if (!v) return null;
  else if (type === "n") {
    const number = Number(raw);
    if (!Number.isFinite(number)) value = raw;
    else {
      const format = formats[Number(c.attrs.s ?? 0)];
      value = format ? formatNumber(number, format, date1904) : generalNumber(number);
      numeric = !(format && isDateFormat(format));
    }
    numeric = numeric || /^\d{4}-\d\d-\d\d/.test(value);
  } else value = raw; // str (a formula's text), e (an error), d (an ISO date)
  value = value.trim();
  return value === "" ? null : { value, numeric };
}

export interface XlsxResult {
  units: TextUnit[];
  workbook: Workbook;
}

export async function extractXlsx(bytes: Uint8Array): Promise<XlsxResult> {
  const workbook = await readWorkbook(bytes);
  return { units: gridUnits(workbook.sheets, true), workbook };
}
