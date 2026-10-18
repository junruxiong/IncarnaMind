/**
 * The workbook calculator (spike: excel-edit): it evaluates Excel formulas
 * over a workbook read by ./xlsxBook, and after an edit recalculates only the
 * cells that depend on it, in dependency order, as Excel's recalculation
 * does. Values follow Excel's rules (blank is 0 or "", text compares without
 * case, errors pass through); functions are in ./xlsxFunctions.
 *
 * A cell it can't recalculate (a function it doesn't have, a 3-D or external
 * reference, a circular one, a what-if data table) keeps the value the file
 * stored and is listed, so the copy asks Excel to recalculate on opening and
 * the editor can say which cells may be stale.
 *
 * Pure: no Node or DOM imports, so the core and the renderer share it.
 */
import { columnName, parseCellRef } from "../../../shared/units";
import {
  type Book,
  type BookCell,
  type BookSheet,
  type BookTable,
  cellKey,
  type DefinedName,
  type FormulaCell,
  isError,
  keyColumn,
  keyRow,
  readBook,
  type Scalar,
} from "./xlsxBook";
import {
  type Corner,
  MAX_COLUMN,
  MAX_ROW,
  moveCorner,
  type Node,
  parseFormula,
  printFormula,
} from "./xlsxFormula";
import { FUNCTIONS, VOLATILE } from "./xlsxFunctions";
import {
  CannotCalculate,
  err,
  isArray,
  isRange,
  type RangeValue,
  sameValue,
  scalarOp,
  toNumber,
  typedValue,
  type Value,
} from "./xlsxValues";

/** A global id for a cell: its sheet's place and its key. */
const cellId = (sheet: number, key: number) => sheet * 2 ** 35 + key;
const idSheet = (id: number) => Math.floor(id / 2 ** 35);
const idKey = (id: number) => id % 2 ** 35;

interface RangeEntry {
  box: Box;
  dependents: Set<number>;
  /** The formula cells inside it, once asked for. */
  formulas: number[] | null;
}

interface Box {
  sheet: number;
  top: number;
  left: number;
  bottom: number;
  right: number;
}

/** Where a formula is being evaluated. */
export interface Context {
  calc: Calculator;
  sheet: BookSheet;
  row: number;
  column: number;
  /** The cell the formula's references are written relative to. */
  origin: { row: number; column: number };
  /** An array formula's: operations on ranges give arrays, not one cell of them. */
  array: boolean;
}

export interface CellChange {
  sheet: string;
  row: number;
  column: number;
  value: Scalar;
}

export interface Stale {
  sheet: string;
  row: number;
  column: number;
  reason: string;
}

export interface Recalculated {
  /** Every cell whose value is now different, the edited ones included. */
  changed: CellChange[];
  /** Cells that depend on the edit but couldn't be recalculated: they keep their old values. */
  stale: Stale[];
}

/** An edit as the User typed it, for writing the copy (./xlsxCopy). */
export interface CellEdit {
  sheet: string;
  row: number;
  column: number;
  /** As typed: "=SUM(B2:B9)" is a formula, anything else a value. */
  input: string;
}

/** What the copy must write for a cell. */
export interface CellWrite {
  part: string;
  row: number;
  column: number;
  /** The formula to write, without "="; null writes a plain value; undefined keeps the cell's own formula. */
  formula?: string | null;
  value: Scalar;
}

const samePosition = (a: { row: number; column: number }, b: { row: number; column: number }) =>
  a.row === b.row && a.column === b.column;

export class Calculator {
  readonly book: Book;
  private readonly sheetsByName = new Map<string, BookSheet>();
  /** Row → sorted columns, per sheet, for iterating ranges over the cells there are. */
  private readonly rowIndex = new Map<number, { rows: number[]; columns: Map<number, number[]> }>();
  /** Dependents of single cells, by the cell. */
  private readonly cellDependents = new Map<number, Set<number>>();
  /** Each distinct range formulas read, with its dependents, by sheet: many formulas share one table. */
  private readonly ranges = new Map<number, Map<string, RangeEntry>>();
  /** What each formula reads: single cells, and the keys of the ranges. */
  private readonly precedents = new Map<number, { cells: number[]; ranges: RangeEntry[] }>();
  /** Every formula cell's key, by sheet, for finding those inside a range. */
  private formulaKeys: Map<number, number[]> | null = null;
  private readonly bounds = new Map<number, { rows: number; columns: number }>();
  private readonly volatile = new Set<number>();
  /** Cells whose formulas can't be calculated, with why. */
  readonly cannot = new Map<number, string>();
  /** Cells the User edited, with what they typed. */
  readonly edits = new Map<number, string>();
  /** Cells whose stored value changed: recalculated, or edited. */
  readonly touched = new Set<number>();
  /** Shared formula groups whose first cell was edited: their other cells get their own formulas. */
  readonly unshared = new Set<string>();
  /** The stored results were refreshed: once, before the first edit, in a file whose results are placeholders. */
  private refreshed = false;

  constructor(book: Book) {
    this.book = book;
    for (const sheet of book.sheets) this.sheetsByName.set(sheet.name.toLowerCase(), sheet);
    for (const sheet of book.sheets) {
      for (const [key, cell] of sheet.cells) {
        if (cell.formula) this.link(sheet, key, cell);
      }
    }
  }

  static async open(bytes: Uint8Array): Promise<Calculator> {
    return new Calculator(await readBook(bytes));
  }

  sheetNamed(name: string): BookSheet | undefined {
    return this.sheetsByName.get(name.toLowerCase());
  }

  // -------------------------------------------------------------------------
  // The dependency graph

  private link(sheet: BookSheet, key: number, cell: BookCell): void {
    const formula = cell.formula as FormulaCell;
    const id = cellId(sheet.index, key);
    const reason = whyNot(formula);
    if (reason) {
      this.cannot.set(id, reason);
      return;
    }
    const boxes: Box[] = [];
    const context: Context = {
      calc: this,
      sheet,
      row: cell.row,
      column: cell.column,
      origin: formula.origin,
      array: formula.array !== undefined,
    };
    try {
      this.collect(formula.node as Node, context, boxes, new Set());
    } catch (error) {
      this.cannot.set(id, error instanceof Error ? error.message : String(error));
      return;
    }
    const reads = { cells: [] as number[], ranges: [] as RangeEntry[] };
    for (const box of boxes) {
      if (box.top === box.bottom && box.left === box.right) {
        const precedent = cellId(box.sheet, cellKey(box.top, box.left));
        reads.cells.push(precedent);
        let set = this.cellDependents.get(precedent);
        if (!set) {
          set = new Set();
          this.cellDependents.set(precedent, set);
        }
        set.add(id);
        continue;
      }
      let onSheet = this.ranges.get(box.sheet);
      if (!onSheet) {
        onSheet = new Map();
        this.ranges.set(box.sheet, onSheet);
      }
      const name = `${box.top}:${box.left}:${box.bottom}:${box.right}`;
      let entry = onSheet.get(name);
      if (!entry) {
        entry = { box, dependents: new Set(), formulas: null };
        onSheet.set(name, entry);
      }
      entry.dependents.add(id);
      reads.ranges.push(entry);
    }
    this.precedents.set(id, reads);
  }

  private unlink(id: number): void {
    const reads = this.precedents.get(id);
    for (const precedent of reads?.cells ?? []) this.cellDependents.get(precedent)?.delete(id);
    for (const entry of reads?.ranges ?? []) entry.dependents.delete(id);
    this.precedents.delete(id);
    this.volatile.delete(id);
    this.cannot.delete(id);
  }

  /** The formula cells inside a range, worked out once. */
  private formulasIn(entry: RangeEntry): number[] {
    if (entry.formulas) return entry.formulas;
    if (!this.formulaKeys) {
      this.formulaKeys = new Map();
      for (const sheet of this.book.sheets) {
        const keys: number[] = [];
        for (const [key, cell] of sheet.cells) if (cell.formula) keys.push(key);
        this.formulaKeys.set(sheet.index, keys);
      }
    }
    const { box } = entry;
    entry.formulas = (this.formulaKeys.get(box.sheet) ?? [])
      .filter((key) => {
        const row = keyRow(key);
        const column = keyColumn(key);
        return row >= box.top && row <= box.bottom && column >= box.left && column <= box.right;
      })
      .map((key) => cellId(box.sheet, key));
    return entry.formulas;
  }

  /** After an edit adds or removes a formula, the ranges' lists of formulas are worked out again. */
  private formulasChanged(): void {
    this.formulaKeys = null;
    for (const onSheet of this.ranges.values())
      for (const entry of onSheet.values()) entry.formulas = null;
  }

  /** The boxes a formula reads, found from its tree (INDEX's and OFFSET's whole ranges included). */
  private collect(node: Node, context: Context, boxes: Box[], names: Set<string>): void {
    switch (node.kind) {
      case "ref": {
        const range = this.resolveRef(node, context);
        if (isRange(range)) boxes.push(boxOf(range));
        return;
      }
      case "table": {
        const range = this.resolveTable(node, context);
        if (isRange(range)) boxes.push(boxOf(range));
        return;
      }
      case "name": {
        const defined = this.findName(node.name, node.sheet, context.sheet);
        if (!defined) return;
        const key = `${defined.sheet}:${defined.name.toLowerCase()}`;
        if (names.has(key) || !defined.node) return;
        names.add(key);
        this.collect(
          defined.node,
          { ...context, origin: { row: context.row, column: context.column } },
          boxes,
          names,
        );
        return;
      }
      case "call": {
        if (VOLATILE.has(node.name)) {
          this.volatile.add(cellId(context.sheet.index, cellKey(context.row, context.column)));
        }
        if (!FUNCTIONS.has(node.name)) throw new CannotCalculate(`function ${node.name}`);
        for (const arg of node.args) this.collect(arg, context, boxes, names);
        return;
      }
      case "unary":
      case "percent":
        this.collect(node.operand, context, boxes, names);
        return;
      case "binary":
      case "range":
        this.collect(node.left, context, boxes, names);
        this.collect(node.right, context, boxes, names);
        return;
      case "unsupported":
        throw new CannotCalculate(node.reason);
      default:
        return;
    }
  }

  /** The cells whose values depend on these, directly or not, in the order to recalculate them. */
  private dependentsInOrder(starts: Iterable<number>, withVolatile = true): number[] {
    return this.order(this.dirtyFrom(starts, withVolatile));
  }

  /** The cells whose values depend on these, directly or not. */
  private dirtyFrom(starts: Iterable<number>, withVolatile: boolean): Set<number> {
    const dirty = new Set<number>();
    const queue: number[] = [];
    const add = (id: number) => {
      if (!dirty.has(id)) {
        dirty.add(id);
        queue.push(id);
      }
    };
    const visit = (id: number) => {
      for (const dependent of this.cellDependents.get(id) ?? []) add(dependent);
      const onSheet = this.ranges.get(idSheet(id));
      if (!onSheet) return;
      const key = idKey(id);
      const row = keyRow(key);
      const column = keyColumn(key);
      for (const { box, dependents } of onSheet.values()) {
        if (row >= box.top && row <= box.bottom && column >= box.left && column <= box.right) {
          for (const dependent of dependents) add(dependent);
        }
      }
    };
    for (const id of starts) visit(id);
    if (withVolatile) for (const id of this.volatile) add(id);
    for (let next = 0; next < queue.length; next++) visit(queue[next] as number);
    return dirty;
  }

  /** The cells in an order where each comes after the dirty cells it reads (iterative, for long chains). */
  private order(dirty: Set<number>): number[] {
    const done = new Set<number>();
    const onPath = new Set<number>();
    const ordered: number[] = [];
    for (const start of dirty) {
      if (done.has(start)) continue;
      const stack: { id: number; next: number[] | null }[] = [{ id: start, next: null }];
      while (stack.length > 0) {
        const top = stack[stack.length - 1] as { id: number; next: number[] | null };
        if (top.next === null) {
          onPath.add(top.id);
          top.next = this.dirtyPrecedents(top.id, dirty);
        }
        const following = top.next.pop();
        if (following === undefined) {
          stack.pop();
          onPath.delete(top.id);
          done.add(top.id);
          ordered.push(top.id);
          continue;
        }
        if (done.has(following)) continue;
        if (onPath.has(following)) {
          this.cannot.set(following, "circular reference");
          continue;
        }
        stack.push({ id: following, next: null });
      }
    }
    return ordered;
  }

  private dirtyPrecedents(id: number, dirty: Set<number>): number[] {
    const reads = this.precedents.get(id);
    if (!reads) return [];
    const found = reads.cells.filter((precedent) => dirty.has(precedent));
    for (const entry of reads.ranges) {
      for (const precedent of this.formulasIn(entry))
        if (dirty.has(precedent)) found.push(precedent);
    }
    return found;
  }

  // -------------------------------------------------------------------------
  // Cells

  private indexOf(sheet: BookSheet) {
    let index = this.rowIndex.get(sheet.index);
    if (!index) {
      const columns = new Map<number, number[]>();
      for (const key of sheet.cells.keys()) {
        const row = keyRow(key);
        let list = columns.get(row);
        if (!list) {
          list = [];
          columns.set(row, list);
        }
        list.push(keyColumn(key));
      }
      for (const list of columns.values()) list.sort((a, b) => a - b);
      index = { rows: [...columns.keys()].sort((a, b) => a - b), columns };
      this.rowIndex.set(sheet.index, index);
    }
    return index;
  }

  /** Calls `each` with the key of every stored cell inside the box, row by row. */
  eachKey(sheet: BookSheet, box: Omit<Box, "sheet">, each: (key: number) => void): void {
    const index = this.indexOf(sheet);
    let at = lowerBound(index.rows, box.top);
    for (; at < index.rows.length; at++) {
      const row = index.rows[at] as number;
      if (row > box.bottom) break;
      const columns = index.columns.get(row) as number[];
      for (let c = lowerBound(columns, box.left); c < columns.length; c++) {
        const column = columns[c] as number;
        if (column > box.right) break;
        each(cellKey(row, column));
      }
    }
  }

  /** The last row and column with a stored cell, to bound whole-column ranges. */
  usedBounds(sheet: BookSheet): { rows: number; columns: number } {
    const known = this.bounds.get(sheet.index);
    if (known) return known;
    const index = this.indexOf(sheet);
    let columns = 0;
    for (const list of index.columns.values())
      columns = Math.max(columns, (list[list.length - 1] ?? 0) + 1);
    const bounds = { rows: index.rows[index.rows.length - 1] ?? 0, columns };
    this.bounds.set(sheet.index, bounds);
    return bounds;
  }

  private addToIndex(sheet: BookSheet, row: number, column: number): void {
    this.bounds.delete(sheet.index);
    const index = this.rowIndex.get(sheet.index);
    if (!index) return;
    let list = index.columns.get(row);
    if (!list) {
      list = [];
      index.columns.set(row, list);
      index.rows.splice(lowerBound(index.rows, row), 0, row);
    }
    const at = lowerBound(list, column);
    if (list[at] !== column) list.splice(at, 0, column);
  }

  cellValue(sheet: BookSheet, row: number, column: number): Scalar {
    return sheet.cells.get(cellKey(row, column))?.value ?? null;
  }

  /** The formula a cell shows in the formula bar, with "=", or null for a plain value. */
  formulaText(sheetName: string, row: number, column: number): string | null {
    const sheet = this.sheetNamed(sheetName);
    const cell = sheet?.cells.get(cellKey(row, column));
    const formula = cell?.formula;
    if (!cell || !formula) return null;
    if (formula.node && !samePosition(formula.origin, cell)) {
      return `=${printFormula(formula.node, cell.row - formula.origin.row, cell.column - formula.origin.column)}`;
    }
    return `=${formula.text}`;
  }

  // -------------------------------------------------------------------------
  // Editing and recalculating

  /**
   * Sets a cell to what the User typed, then recalculates what depends on it.
   * Throws for a cell inside an array formula, which Excel doesn't let one change alone.
   */
  edit(sheetName: string, ref: string, input: string): Recalculated {
    const sheet = this.sheetNamed(sheetName);
    const at = parseCellRef(ref);
    if (!sheet || !at) throw new Error(`No cell ${sheetName}!${ref}`);
    // A file with placeholders for results (0, or none) would hand them on, from the cells an
    // edit doesn't reach to the ones it does: work every formula out first.
    const refreshed = this.book.placeholders && !this.refreshed ? this.recalculateAll() : null;
    this.refreshed = true;
    const key = cellKey(at.row, at.column);
    const id = cellId(sheet.index, key);
    const existing = sheet.cells.get(key);
    if (
      existing?.arrayPart !== undefined ||
      (existing?.formula?.array && !singleCell(existing.formula.array))
    ) {
      throw new Error(
        "This cell is part of an array formula, which can only be changed as a whole.",
      );
    }
    const cell: BookCell = existing ?? { row: at.row, column: at.column, value: null, style: 0 };
    if (!existing) {
      sheet.cells.set(key, cell);
      this.addToIndex(sheet, at.row, at.column);
    }
    if (cell.formula?.shared?.anchor)
      this.unshared.add(`${sheet.index}:${cell.formula.shared.index}`);
    this.unlink(id);
    const before = cell.value;
    const trimmed = input.trim();
    if (trimmed.startsWith("=") && trimmed.length > 1) {
      const text = trimmed.slice(1);
      let node: Node | null = null;
      let problem: string | undefined;
      try {
        node = parseFormula(text);
      } catch (error) {
        problem = error instanceof Error ? error.message : String(error);
      }
      cell.formula = {
        text,
        node,
        origin: { row: at.row, column: at.column },
        ...(problem ? { problem } : {}),
      };
      this.link(sheet, key, cell);
      try {
        cell.value = this.evaluateCell(sheet, cell);
      } catch (error) {
        if (!(error instanceof CannotCalculate)) throw error;
        this.cannot.set(id, error.message);
        cell.value = null;
      }
    } else {
      delete cell.formula;
      cell.value = typedValue(input);
    }
    this.formulasChanged();
    this.edits.set(id, input);
    this.touched.add(id);
    const changed: CellChange[] = [];
    if (!sameValue(before, cell.value))
      changed.push({ sheet: sheet.name, row: at.row, column: at.column, value: cell.value });
    const result = this.recalculate([id]);
    return {
      changed: [...(refreshed?.changed ?? []), ...changed, ...result.changed],
      stale: [...(refreshed?.stale ?? []), ...result.stale],
    };
  }

  /** Recalculates every formula (for checking the calculator against a file's stored results). */
  recalculateAll(): Recalculated {
    const all = new Set<number>();
    for (const sheet of this.book.sheets) {
      for (const [key, cell] of sheet.cells) if (cell.formula) all.add(cellId(sheet.index, key));
    }
    return this.run(this.order(all));
  }

  private recalculate(starts: number[]): Recalculated {
    return this.run(this.dependentsInOrder(starts).filter((id) => !starts.includes(id)));
  }

  private run(order: number[]): Recalculated {
    const changed: CellChange[] = [];
    const stale: Stale[] = [];
    for (const id of order) {
      const sheet = this.book.sheets[idSheet(id)] as BookSheet;
      const cell = sheet.cells.get(idKey(id));
      if (!cell?.formula) continue;
      const reason = this.cannot.get(id);
      if (reason) {
        stale.push({ sheet: sheet.name, row: cell.row, column: cell.column, reason });
        continue;
      }
      let value: Scalar;
      try {
        value = this.evaluateCell(sheet, cell);
      } catch (error) {
        if (!(error instanceof CannotCalculate)) throw error;
        this.cannot.set(id, error.message);
        stale.push({
          sheet: sheet.name,
          row: cell.row,
          column: cell.column,
          reason: error.message,
        });
        continue;
      }
      if (cell.formula.array && !singleCell(cell.formula.array)) {
        // The other cells of an array formula took their values in evaluateCell.
        changed.push({ sheet: sheet.name, row: cell.row, column: cell.column, value });
        continue;
      }
      if (!sameValue(cell.value, value)) {
        cell.value = value;
        this.touched.add(id);
        changed.push({ sheet: sheet.name, row: cell.row, column: cell.column, value });
      }
    }
    return { changed, stale };
  }

  /** A formula cell's value, worked out now. */
  evaluateCell(sheet: BookSheet, cell: BookCell): Scalar {
    const formula = cell.formula as FormulaCell;
    const reason = whyNot(formula);
    if (reason) throw new CannotCalculate(reason);
    const context: Context = {
      calc: this,
      sheet,
      row: cell.row,
      column: cell.column,
      origin: formula.origin,
      array: formula.array !== undefined,
    };
    const result = this.evaluate(formula.node as Node, context);
    if (formula.array && !singleCell(formula.array))
      return this.spreadArray(sheet, cell, formula.array, result);
    return finalValue(this.toScalar(result, context));
  }

  /** An array formula's result spread over its range; its first cell's value returned. */
  private spreadArray(sheet: BookSheet, cell: BookCell, ref: string, result: Value): Scalar {
    const [from, to] = ref.split(":");
    const a = parseCellRef(from ?? "");
    const b = parseCellRef(to ?? from ?? "");
    if (!a || !b) throw new CannotCalculate("array formula range");
    const rows = this.toRows(result);
    const height = rows.length;
    const width = rows[0]?.length ?? 0;
    for (let row = a.row; row <= b.row; row++) {
      for (let column = a.column; column <= b.column; column++) {
        const r = height === 1 ? 0 : row - a.row;
        const c = width === 1 ? 0 : column - a.column;
        const value = finalValue(rows[r]?.[c] ?? err("#N/A"));
        const target = sheet.cells.get(cellKey(row, column));
        if (!target || target === cell) continue;
        if (!sameValue(target.value, value)) {
          target.value = value;
          this.touched.add(cellId(sheet.index, cellKey(row, column)));
        }
      }
    }
    return finalValue(rows[0]?.[0] ?? err("#N/A"));
  }

  /** What the copy must write: the edits, and every cell whose stored value changed. */
  writes(): CellWrite[] {
    const writes: CellWrite[] = [];
    for (const id of this.touched) {
      const sheet = this.book.sheets[idSheet(id)] as BookSheet;
      const cell = sheet.cells.get(idKey(id));
      if (!cell) continue;
      const edited = this.edits.has(id);
      const write: CellWrite = {
        part: sheet.part,
        row: cell.row,
        column: cell.column,
        value: cell.value,
      };
      if (edited) write.formula = cell.formula ? cell.formula.text : null;
      writes.push(write);
    }
    // A shared formula whose first cell was edited: each other cell of it gets its own text.
    for (const group of this.unshared) {
      const [sheetIndex, index] = group.split(":").map(Number);
      const sheet = this.book.sheets[sheetIndex as number] as BookSheet;
      for (const [key, cell] of sheet.cells) {
        const formula = cell.formula;
        if (!formula?.shared || formula.shared.index !== index || formula.shared.anchor) continue;
        if (this.edits.has(cellId(sheet.index, key))) continue;
        const text = this.formulaText(sheet.name, cell.row, cell.column)?.slice(1) ?? formula.text;
        const existing = writes.find(
          (each) =>
            each.part === sheet.part && each.row === cell.row && each.column === cell.column,
        );
        if (existing) existing.formula = text;
        else
          writes.push({
            part: sheet.part,
            row: cell.row,
            column: cell.column,
            formula: text,
            value: cell.value,
          });
      }
    }
    return writes;
  }

  /** How many formula cells an edit of this cell would recalculate. */
  dependentCount(sheetName: string, row: number, column: number): number {
    const sheet = this.sheetNamed(sheetName);
    if (!sheet) return 0;
    const id = cellId(sheet.index, cellKey(row, column));
    const dirty = this.dirtyFrom([id], false);
    dirty.delete(id);
    return dirty.size;
  }

  /** The cells an edit of this cell recalculates, as "Sheet!A1". */
  dependentRefs(
    sheetName: string,
    row: number,
    column: number,
  ): { sheet: string; row: number; column: number }[] {
    const sheet = this.sheetNamed(sheetName);
    if (!sheet) return [];
    const id = cellId(sheet.index, cellKey(row, column));
    return this.dependentsInOrder([id], false)
      .filter((each) => each !== id)
      .map((each) => {
        const key = idKey(each);
        return {
          sheet: (this.book.sheets[idSheet(each)] as BookSheet).name,
          row: keyRow(key),
          column: keyColumn(key),
        };
      });
  }

  /** The cells that depend on a volatile function (TODAY, NOW, RAND, OFFSET, INDIRECT), as "Sheet!A1". */
  volatileCells(): Set<string> {
    const dirty = this.dirtyFrom([], true);
    const refs = new Set<string>();
    for (const id of dirty) {
      const key = idKey(id);
      const sheet = this.book.sheets[idSheet(id)] as BookSheet;
      refs.add(`${sheet.name}!${columnName(keyColumn(key))}${keyRow(key)}`);
    }
    return refs;
  }

  /** Whether any formula in the workbook can't be recalculated by this calculator. */
  get incomplete(): boolean {
    return this.cannot.size > 0;
  }

  // -------------------------------------------------------------------------
  // Evaluating

  evaluate(node: Node, context: Context): Value {
    switch (node.kind) {
      case "number":
        return node.value;
      case "string":
        return node.value;
      case "boolean":
        return node.value;
      case "error":
        return err(node.code);
      case "missing":
        return null;
      case "array":
        return {
          kind: "array",
          rows: node.rows.map((row) =>
            row.map((each) => this.toScalar(this.evaluate(each, context), context)),
          ),
        };
      case "ref":
        return this.resolveRef(node, context);
      case "table":
        return this.resolveTable(node, context);
      case "name": {
        const defined = this.findName(node.name, node.sheet, context.sheet);
        if (!defined) return err("#NAME?");
        if (!defined.node) throw new CannotCalculate(`name ${defined.name}`);
        return this.evaluate(defined.node, {
          ...context,
          origin: { row: context.row, column: context.column },
        });
      }
      case "call": {
        const fn = FUNCTIONS.get(node.name);
        if (!fn) throw new CannotCalculate(`function ${node.name}`);
        return fn(node.args, context);
      }
      case "unary": {
        const value = this.evaluate(node.operand, context);
        if (node.op === "+") return value;
        return this.lift1(value, context, (each) => {
          const number = toNumber(each);
          return isError(number) ? number : -number;
        });
      }
      case "percent":
        return this.lift1(this.evaluate(node.operand, context), context, (each) => {
          const number = toNumber(each);
          return isError(number) ? number : number / 100;
        });
      case "binary":
        return this.binary(
          node.op,
          this.evaluate(node.left, context),
          this.evaluate(node.right, context),
          context,
        );
      case "range": {
        const left = this.evaluate(node.left, context);
        const right = this.evaluate(node.right, context);
        if (!isRange(left) || !isRange(right) || left.sheet !== right.sheet) return err("#VALUE!");
        return {
          kind: "range",
          sheet: left.sheet,
          top: Math.min(left.top, right.top),
          left: Math.min(left.left, right.left),
          bottom: Math.max(left.bottom, right.bottom),
          right: Math.max(left.right, right.right),
        };
      }
      case "unsupported":
        throw new CannotCalculate(node.reason);
    }
  }

  resolveRef(node: Extract<Node, { kind: "ref" }>, context: Context): Value {
    const sheet = node.sheet === null ? context.sheet : this.sheetNamed(node.sheet);
    if (!sheet) return err("#REF!");
    const rows = context.row - context.origin.row;
    const columns = context.column - context.origin.column;
    const from = moveCorner(node.from, rows, columns);
    const to = node.to ? moveCorner(node.to, rows, columns) : from;
    if (!from || !to) return err("#REF!");
    return rangeOf(sheet, from, to);
  }

  resolveTable(node: Extract<Node, { kind: "table" }>, context: Context): Value {
    const table = this.book.tables.get(node.table.toLowerCase());
    if (!table) return err("#REF!");
    const firstData = table.top + table.headerRows;
    const lastData = table.bottom - table.totalsRows;
    let top = firstData;
    let bottom = lastData;
    const items = node.items.map((item) => item.toLowerCase());
    if (items.includes("#all")) {
      top = table.top;
      bottom = table.bottom;
    } else if (items.includes("#this row")) {
      top = context.row;
      bottom = context.row;
    } else {
      const headers = items.includes("#headers");
      const totals = items.includes("#totals");
      const data = items.includes("#data") || (!headers && !totals);
      top = headers ? table.top : data ? firstData : lastData + 1;
      bottom = totals ? table.bottom : data ? lastData : table.top + table.headerRows - 1;
    }
    let left = table.left;
    let right = table.right;
    if (node.first !== null) {
      const first = columnOf(table, node.first);
      const last = columnOf(table, node.last ?? node.first);
      if (first < 0 || last < 0) return err("#REF!");
      left = table.left + Math.min(first, last);
      right = table.left + Math.max(first, last);
    }
    if (bottom < top) return err("#REF!");
    return { kind: "range", sheet: table.sheet, top, left, bottom, right };
  }

  findName(name: string, sheetName: string | null, current: BookSheet): DefinedName | undefined {
    const lower = name.toLowerCase();
    const scope = sheetName === null ? current : this.sheetNamed(sheetName);
    const local = this.book.names.find(
      (each) => each.sheet === scope?.index && each.name.toLowerCase() === lower,
    );
    if (local || sheetName !== null) return local;
    return this.book.names.find((each) => each.sheet === null && each.name.toLowerCase() === lower);
  }

  // -------------------------------------------------------------------------
  // Values

  /** A value as one cell's: a range by implicit intersection with the formula's cell, an array's first. */
  toScalar(value: Value, context: Context): Scalar {
    if (isArray(value)) return value.rows[0]?.[0] ?? null;
    if (!isRange(value)) return value;
    if (value.top === value.bottom && value.left === value.right) {
      return this.cellValue(value.sheet, value.top, value.left);
    }
    if (context.array) return this.cellValue(value.sheet, value.top, value.left);
    if (value.left === value.right && context.row >= value.top && context.row <= value.bottom) {
      return this.cellValue(value.sheet, context.row, value.left);
    }
    if (
      value.top === value.bottom &&
      context.column >= value.left &&
      context.column <= value.right
    ) {
      return this.cellValue(value.sheet, value.top, context.column);
    }
    return err("#VALUE!");
  }

  /** A value as rows of cells. Whole columns and rows stop at the last stored cell. */
  toRows(value: Value): Scalar[][] {
    if (isArray(value)) return value.rows;
    if (!isRange(value)) return [[value]];
    const used = this.usedBounds(value.sheet);
    const bottom = Math.min(value.bottom, Math.max(used.rows, value.top));
    const right = Math.min(value.right, Math.max(used.columns - 1, value.left));
    const rows: Scalar[][] = [];
    for (let row = value.top; row <= bottom; row++) {
      const line: Scalar[] = [];
      for (let column = value.left; column <= right; column++)
        line.push(this.cellValue(value.sheet, row, column));
      rows.push(line);
    }
    return rows;
  }

  /** Every value of a range or array, in reading order; stored cells only for a range (blanks skipped). */
  eachValue(value: Value, each: (value: Scalar, fromRange: boolean) => void): void {
    if (isRange(value)) {
      this.eachKey(value.sheet, value, (key) =>
        each(value.sheet.cells.get(key)?.value ?? null, true),
      );
      return;
    }
    if (isArray(value)) {
      for (const row of value.rows) for (const cell of row) each(cell, false);
      return;
    }
    each(value, false);
  }

  private lift1(value: Value, context: Context, op: (value: Scalar) => Scalar): Value {
    if (isRange(value) && !(value.top === value.bottom && value.left === value.right)) {
      if (!context.array) return op(this.toScalar(value, context));
      return { kind: "array", rows: this.toRows(value).map((row) => row.map(op)) };
    }
    if (isArray(value)) return { kind: "array", rows: value.rows.map((row) => row.map(op)) };
    return op(this.toScalar(value, context));
  }

  binary(op: string, left: Value, right: Value, context: Context): Value {
    const many = (value: Value) =>
      isArray(value) ||
      (isRange(value) && !(value.top === value.bottom && value.left === value.right));
    if ((many(left) || many(right)) && context.array) {
      const a = this.toRows(left);
      const b = this.toRows(right);
      const height = Math.max(a.length, b.length);
      const width = Math.max(a[0]?.length ?? 0, b[0]?.length ?? 0);
      const rows: Scalar[][] = [];
      for (let r = 0; r < height; r++) {
        const line: Scalar[] = [];
        for (let c = 0; c < width; c++) {
          const x = pick(a, r, c);
          const y = pick(b, r, c);
          line.push(x === undefined || y === undefined ? err("#N/A") : scalarOp(op, x, y));
        }
        rows.push(line);
      }
      return { kind: "array", rows };
    }
    return scalarOp(op, this.toScalar(left, context), this.toScalar(right, context));
  }
}

const pick = (rows: Scalar[][], r: number, c: number): Scalar | undefined => {
  const row = rows.length === 1 ? rows[0] : rows[r];
  if (!row) return undefined;
  return row.length === 1 ? row[0] : row[c];
};

function columnOf(table: BookTable, name: string): number {
  const plain = (text: string) => text.replace(/\r\n?/g, "\n").toLowerCase();
  const wanted = plain(name);
  return table.columns.findIndex((column) => plain(column) === wanted);
}

function rangeOf(sheet: BookSheet, from: Corner, to: Corner): RangeValue {
  return {
    kind: "range",
    sheet,
    top: Math.min(from.row, to.row),
    left: Math.min(from.column, to.column),
    bottom: Math.min(MAX_ROW, Math.max(from.row, to.row)),
    right: Math.min(MAX_COLUMN, Math.max(from.column, to.column)),
  };
}

const boxOf = (range: RangeValue): Box => ({
  sheet: range.sheet.index,
  top: range.top,
  left: range.left,
  bottom: range.bottom,
  right: range.right,
});

function lowerBound(list: readonly number[], value: number): number {
  let low = 0;
  let high = list.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if ((list[middle] as number) < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

const singleCell = (ref: string) => {
  const [from, to] = ref.split(":");
  return to === undefined || from === to;
};

function whyNot(formula: FormulaCell): string | null {
  if (formula.dataTable) return "what-if data table";
  if (formula.problem) return formula.problem;
  if (!formula.node) return "unreadable formula";
  return null;
}

/** A result as a cell stores it: an empty reference is 0, as in Excel. */
function finalValue(value: Scalar): Scalar {
  if (value === null) return 0;
  if (typeof value === "number" && !Number.isFinite(value)) return err("#NUM!");
  return value;
}
