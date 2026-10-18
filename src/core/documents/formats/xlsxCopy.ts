/**
 * An edited copy of a workbook (spike: excel-edit), written the way a Word
 * file goes back into itself (ADR-0011, #105): every part the edit doesn't
 * reach is copied as it was, compressed bytes and all, so charts, pivot
 * tables, conditional formats, validation, names and whatever this version
 * doesn't know stay exactly as the file had them. In the sheets that change,
 * only the changed cells' XML is replaced, spliced in by the offsets the XML
 * reader records; their style (`s`) stays.
 *
 * The calculation chain is dropped (Excel rebuilds it), and the workbook asks
 * Excel to recalculate on opening, in case a cell the calculator couldn't
 * recalculate is stale. The original file is never written.
 */
import { parseCellRef } from "../../../shared/units";
import { type CalcError, isError, type Scalar } from "./xlsxBook";
import type { CellWrite } from "./xlsxCalc";
import { child, elements, is, parseXml, type XmlElement } from "./xml";
import { type Deflate, openPackage, writeZip, type ZipWriteEntry } from "./zip";

const escapeXml = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const columnName = (index: number) => {
  let name = "";
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) {
    name = String.fromCharCode(65 + ((n - 1) % 26)) + name;
  }
  return name;
};

interface Splice {
  start: number;
  end: number;
  text: string;
}

/** The copy's bytes: the workbook with the writes made. */
export async function writeCopy(
  bytes: Uint8Array,
  writes: readonly CellWrite[],
  deflate: Deflate,
): Promise<Uint8Array> {
  const zip = openPackage(bytes);
  const replaced = new Map<string, string>();
  const byPart = new Map<string, CellWrite[]>();
  for (const write of writes) {
    const list = byPart.get(write.part) ?? [];
    list.push(write);
    byPart.set(write.part, list);
  }
  for (const [part, list] of byPart) {
    const xml = await zip.readText(part);
    if (xml === undefined) throw new Error(`The workbook has no ${part}.`);
    replaced.set(part, patchSheet(xml, list));
  }
  const dropped = new Set<string>();
  if (writes.length > 0) {
    const workbook = await zip.readText("xl/workbook.xml");
    if (workbook !== undefined) replaced.set("xl/workbook.xml", recalculateOnOpening(workbook));
    if (zip.has("xl/calcChain.xml")) {
      dropped.add("xl/calcChain.xml");
      const rels = await zip.readText("xl/_rels/workbook.xml.rels");
      if (rels !== undefined) {
        replaced.set(
          "xl/_rels/workbook.xml.rels",
          rels.replace(/<(?:\w+:)?Relationship\b[^>]*Type="[^"]*\/calcChain"[^>]*\/>/g, ""),
        );
      }
      const types = await zip.readText("[Content_Types].xml");
      if (types !== undefined) {
        replaced.set(
          "[Content_Types].xml",
          types.replace(/<(?:\w+:)?Override\b[^>]*PartName="\/xl\/calcChain\.xml"[^>]*\/>/g, ""),
        );
      }
    }
  }
  const entries: ZipWriteEntry[] = [];
  for (const name of zip.names()) {
    if (dropped.has(name)) continue;
    const text = replaced.get(name);
    if (text !== undefined) entries.push({ name, data: text });
    else {
      const raw = zip.raw(name);
      if (raw) entries.push({ name, raw });
    }
  }
  return writeZip(entries, deflate);
}

/** workbook.xml with `fullCalcOnLoad` set on its calculation properties. */
function recalculateOnOpening(xml: string): string {
  const workbook = parseXml(xml);
  const calcPr = child(workbook, "calcPr");
  const prefix = workbook.name.includes(":") ? `${workbook.name.split(":")[0]}:` : "";
  if (calcPr && calcPr.start !== undefined && calcPr.openEnd !== undefined) {
    const tag = xml.slice(calcPr.start, calcPr.openEnd);
    const updated = /\bfullCalcOnLoad="[^"]*"/.test(tag)
      ? tag.replace(/\bfullCalcOnLoad="[^"]*"/, 'fullCalcOnLoad="1"')
      : tag.replace(/(\/?>)$/, ' fullCalcOnLoad="1"$1');
    return xml.slice(0, calcPr.start) + updated + xml.slice(calcPr.openEnd);
  }
  // CT_Workbook's order: calcPr follows definedNames (and sheets, functionGroups, externalReferences).
  const before = ["definedNames", "externalReferences", "functionGroups", "sheets"]
    .map((name) => child(workbook, name))
    .find((each) => each?.end !== undefined);
  if (!before?.end) return xml;
  return `${xml.slice(0, before.end)}<${prefix}calcPr fullCalcOnLoad="1"/>${xml.slice(before.end)}`;
}

/** A sheet's XML with the writes made to its cells, everything else as it was. */
export function patchSheet(xml: string, writes: readonly CellWrite[]): string {
  const worksheet = parseXml(xml);
  const sheetData = child(worksheet, "sheetData");
  if (!sheetData || sheetData.start === undefined || sheetData.end === undefined) {
    throw new Error("The sheet has no cells to write into.");
  }
  const prefix = sheetData.name.includes(":") ? `${sheetData.name.split(":")[0]}:` : "";
  const rows = new Map<number, { element: XmlElement; cells: Map<number, XmlElement> }>();
  let rowNumber = 0;
  for (const row of elements(sheetData)) {
    if (!is(row, "row")) continue;
    rowNumber = row.attrs.r ? Number(row.attrs.r) : rowNumber + 1;
    const cells = new Map<number, XmlElement>();
    let column = -1;
    for (const c of elements(row)) {
      if (!is(c, "c")) continue;
      const ref = c.attrs.r ? parseCellRef(c.attrs.r) : null;
      column = ref ? ref.column : column + 1;
      cells.set(column, c);
    }
    rows.set(rowNumber, { element: row, cells });
  }

  const splices: Splice[] = [];
  const newCells = new Map<number, CellWrite[]>();
  for (const write of writes) {
    const row = rows.get(write.row);
    const existing = row?.cells.get(write.column);
    if (existing && existing.start !== undefined && existing.end !== undefined) {
      splices.push({
        start: existing.start,
        end: existing.end,
        text: cellXml(prefix, write, existing, xml),
      });
    } else {
      const list = newCells.get(write.row) ?? [];
      list.push(write);
      newCells.set(write.row, list);
    }
  }

  const sortedRows = [...rows.keys()].sort((a, b) => a - b);
  for (const [number, list] of newCells) {
    list.sort((a, b) => a.column - b.column);
    const row = rows.get(number);
    if (
      row &&
      row.element.start !== undefined &&
      row.element.end !== undefined &&
      row.element.openEnd !== undefined
    ) {
      const element = row.element;
      const tag = xml
        .slice(element.start as number, element.openEnd as number)
        .replace(/\s+spans="[^"]*"/, "");
      const selfClosing = tag.endsWith("/>");
      const existing = [...row.cells.entries()].sort((a, b) => a[0] - b[0]);
      if (selfClosing) {
        const open = tag.replace(/\s*\/>$/, ">");
        const cells = list.map((write) => cellXml(prefix, write, undefined, xml)).join("");
        splices.push({
          start: element.start as number,
          end: element.end as number,
          text: `${open}${cells}</${element.name}>`,
        });
        continue;
      }
      splices.push({ start: element.start as number, end: element.openEnd as number, text: tag });
      // Each new cell goes before the first cell to its right, or at the row's end.
      const groups = new Map<number, string>();
      for (const write of list) {
        const after = existing.find(([column]) => column > write.column)?.[1];
        const at = after?.start ?? (element.end as number) - `</${element.name}>`.length;
        groups.set(at, (groups.get(at) ?? "") + cellXml(prefix, write, undefined, xml));
      }
      for (const [at, text] of groups) splices.push({ start: at, end: at, text });
      continue;
    }
    const cells = list.map((write) => cellXml(prefix, write, undefined, xml)).join("");
    const rowXml = `<${prefix}row r="${number}">${cells}</${prefix}row>`;
    const next = sortedRows.find((each) => each > number);
    const nextElement = next !== undefined ? rows.get(next)?.element : undefined;
    if (nextElement?.start !== undefined)
      splices.push({ start: nextElement.start, end: nextElement.start, text: rowXml });
    else if (xml.slice(sheetData.start, sheetData.end).endsWith("/>")) {
      splices.push({
        start: sheetData.start,
        end: sheetData.end,
        text: `<${sheetData.name}>${rowXml}</${sheetData.name}>`,
      });
    } else {
      const at = sheetData.end - `</${sheetData.name}>`.length;
      splices.push({ start: at, end: at, text: rowXml });
    }
  }

  const dimension = child(worksheet, "dimension");
  if (dimension?.start !== undefined && dimension.end !== undefined && newCells.size > 0) {
    const ref = widen(dimension.attrs.ref ?? "A1", writes);
    splices.push({
      start: dimension.start,
      end: dimension.end,
      text: `<${dimension.name} ref="${ref}"/>`,
    });
  }

  splices.sort((a, b) => b.start - a.start || b.end - a.end);
  let out = xml;
  for (const splice of splices)
    out = out.slice(0, splice.start) + splice.text + out.slice(splice.end);
  return out;
}

/** The used range, grown to the new cells. */
function widen(ref: string, writes: readonly CellWrite[]): string {
  const [from, to] = ref.split(":");
  const a = parseCellRef(from ?? "A1") ?? { row: 1, column: 0 };
  const b = parseCellRef(to ?? from ?? "A1") ?? a;
  let top = a.row;
  let left = a.column;
  let bottom = b.row;
  let right = b.column;
  for (const write of writes) {
    top = Math.min(top, write.row);
    left = Math.min(left, write.column);
    bottom = Math.max(bottom, write.row);
    right = Math.max(right, write.column);
  }
  const start = `${columnName(left)}${top}`;
  const end = `${columnName(right)}${bottom}`;
  return start === end ? start : `${start}:${end}`;
}

const errorText = (value: CalcError) => value.error;

/** A number as the file stores it: the shortest text that reads back as the same double. */
const numberXml = (value: number) => String(value);

/**
 * A cell's new XML: its own attributes kept (its style above all) but its
 * type, and with a new formula, its cell metadata; its formula kept when the
 * write keeps it, so a shared formula stays shared.
 */
function cellXml(
  prefix: string,
  write: CellWrite,
  existing: XmlElement | undefined,
  xml: string,
): string {
  const attrs: Record<string, string> = { ...(existing?.attrs ?? {}) };
  delete attrs.t;
  delete attrs.vm;
  if (write.formula !== undefined) delete attrs.cm;
  attrs.r = `${columnName(write.column)}${write.row}`;
  let formula = "";
  if (write.formula === undefined) {
    const f = existing ? child(existing, "f") : undefined;
    if (f?.start !== undefined && f.end !== undefined) formula = xml.slice(f.start, f.end);
  } else if (write.formula !== null) {
    formula = `<${prefix}f>${escapeXml(write.formula)}</${prefix}f>`;
  }
  const hasFormula = formula !== "";
  const { type, content } = valueXml(prefix, write.value, hasFormula);
  if (type) attrs.t = type;
  const extensions = existing ? child(existing, "extLst") : undefined;
  const ext =
    extensions?.start !== undefined && extensions.end !== undefined
      ? xml.slice(extensions.start, extensions.end)
      : "";
  // `r` first, as Excel writes it.
  const ordered = [["r", attrs.r], ...Object.entries(attrs).filter(([name]) => name !== "r")];
  const attributes = ordered
    .map(([name, value]) => ` ${name}="${escapeXml(value as string)}"`)
    .join("");
  const body = `${formula}${content}${ext}`;
  return body ? `<${prefix}c${attributes}>${body}</${prefix}c>` : `<${prefix}c${attributes}/>`;
}

function valueXml(
  prefix: string,
  value: Scalar,
  hasFormula: boolean,
): { type: string | null; content: string } {
  if (value === null) return { type: null, content: "" };
  if (typeof value === "number")
    return { type: null, content: `<${prefix}v>${numberXml(value)}</${prefix}v>` };
  if (typeof value === "boolean")
    return { type: "b", content: `<${prefix}v>${value ? 1 : 0}</${prefix}v>` };
  if (isError(value)) return { type: "e", content: `<${prefix}v>${errorText(value)}</${prefix}v>` };
  if (hasFormula) return { type: "str", content: `<${prefix}v>${escapeXml(value)}</${prefix}v>` };
  const space = /^\s|\s$|\n/.test(value) ? ' xml:space="preserve"' : "";
  return {
    type: "inlineStr",
    content: `<${prefix}is><${prefix}t${space}>${escapeXml(value)}</${prefix}t></${prefix}is>`,
  };
}
