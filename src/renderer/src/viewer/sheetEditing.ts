/**
 * Editing a workbook in the sheet viewer (spike: excel-edit): the calculator
 * built from the file's bytes when the User starts editing, each edit made
 * and what depends on it recalculated, and the cells shown again as their
 * number formats show them. Saving sends the edits to the core, which makes
 * them again on the original file and saves a copy (`CoreApi.saveSheetCopy`).
 */
import { useCallback, useRef, useState } from "react";
import type { SavedSheetCopy, SheetEdit } from "../../../core/api";
import type { Cell, Sheet, Workbook } from "../../../core/documents/formats/grid";
import { displayNumber, displayText } from "../../../core/documents/formats/numberDisplay";
import { formatNumber, generalNumber, isDateFormat } from "../../../core/documents/formats/numbers";
import { cellIndex } from "../../../core/documents/formats/sheetLayout";
import { isError, type Scalar } from "../../../core/documents/formats/xlsxBook";
import { Calculator, type CellChange } from "../../../core/documents/formats/xlsxCalc";
import { columnName } from "../../../shared/units";
import { core } from "../core";

export interface SelectedCell {
  row: number;
  column: number;
}

export interface SheetEditing {
  on: boolean;
  starting: boolean;
  /** Bumped by every edit, so the grid draws again. */
  revision: number;
  edits: SheetEdit[];
  /** Cells that depend on an edit but couldn't be recalculated here. */
  stale: number;
  problem: string | null;
  saving: boolean;
  saved: SavedSheetCopy | null;
  start(): Promise<void>;
  stop(): void;
  /** What the formula bar shows for a cell: its formula, else its value as typed. */
  inputOf(sheet: string, cell: SelectedCell): string;
  commit(sheet: string, cell: SelectedCell, input: string): void;
  undo(): void;
  save(): Promise<void>;
}

/** A cell's value as the preview shows it, in its own number format. */
function showCell(
  workbook: Workbook,
  sheet: Sheet,
  row: number,
  column: number,
  value: Scalar,
  date1904: boolean,
): void {
  const look = sheet.layout;
  const at = cellIndex(row, column);
  const style = workbook.layout?.styles[look?.styles.get(at) ?? 0];
  const format = style?.numberFormat;
  let index = sheet.cells.findIndex((cell) => cell.row === row && cell.column === column);
  const cell: Cell =
    index >= 0 ? (sheet.cells[index] as Cell) : { row, column, value: "", numeric: false };
  delete cell.raw;
  look?.shown.delete(at);
  look?.centred.delete(at);
  look?.runs.delete(at);
  if (typeof value === "number") {
    const code = format?.code;
    cell.value = code ? formatNumber(value, code, date1904) : generalNumber(value);
    cell.numeric = !(code && isDateFormat(code));
    const width = look?.columns.get(column)?.width ?? look?.defaultColumnWidth ?? 64;
    const shown = displayNumber(value, code, {
      date1904,
      ...(format?.builtIn !== undefined ? { builtIn: format.builtIn } : {}),
      generalWidth: Math.min(11, Math.max(1, Math.floor((width - 5) / 7))),
    });
    if (shown.text !== cell.value || shown.color || shown.fill !== undefined)
      look?.shown.set(at, shown);
  } else if (typeof value === "boolean" || isError(value)) {
    cell.value = typeof value === "boolean" ? (value ? "TRUE" : "FALSE") : value.error;
    cell.numeric = false;
    look?.centred.add(at);
  } else {
    cell.value = value ?? "";
    cell.numeric = false;
    const shown = displayText(cell.value, format?.code);
    if (shown.text !== cell.value || shown.color) look?.shown.set(at, shown);
  }
  if (index < 0) {
    sheet.cells.push(cell);
    index = sheet.cells.length - 1;
  }
  sheet.rowCount = Math.max(sheet.rowCount, row);
  sheet.columnCount = Math.max(sheet.columnCount, column + 1);
}

const valueAsTyped = (value: Scalar): string =>
  value === null
    ? ""
    : typeof value === "boolean"
      ? value
        ? "TRUE"
        : "FALSE"
      : isError(value)
        ? value.error
        : String(value);

export function useSheetEditing(
  documentId: string,
  workbook: Workbook,
  bytes: Uint8Array | null,
): SheetEditing {
  const calc = useRef<Calculator | null>(null);
  const [on, setOn] = useState(false);
  const [starting, setStarting] = useState(false);
  const [revision, setRevision] = useState(0);
  const [edits, setEdits] = useState<SheetEdit[]>([]);
  const undoStack = useRef<SheetEdit[]>([]);
  const [stale, setStale] = useState(0);
  const [problem, setProblem] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<SavedSheetCopy | null>(null);

  const apply = useCallback(
    (changed: CellChange[]) => {
      const date1904 = calc.current?.book.date1904 ?? false;
      for (const change of changed) {
        const sheet = workbook.sheets.find((each) => each.name === change.sheet);
        if (sheet) showCell(workbook, sheet, change.row, change.column, change.value, date1904);
      }
      setRevision((n) => n + 1);
    },
    [workbook],
  );

  const start = useCallback(async () => {
    if (!bytes) return;
    setStarting(true);
    setProblem(null);
    try {
      calc.current ??= await Calculator.open(bytes);
      setOn(true);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    } finally {
      setStarting(false);
    }
  }, [bytes]);

  const inputOf = useCallback((sheet: string, cell: SelectedCell) => {
    const engine = calc.current;
    if (!engine) return "";
    return (
      engine.formulaText(sheet, cell.row, cell.column) ??
      valueAsTyped(
        engine.sheetNamed(sheet)?.cells.get(cell.row * 16_384 + cell.column)?.value ?? null,
      )
    );
  }, []);

  const commit = useCallback(
    (sheet: string, cell: SelectedCell, input: string) => {
      const engine = calc.current;
      if (!engine) return;
      const before = inputOf(sheet, cell);
      if (before === input) return;
      const ref = `${columnName(cell.column)}${cell.row}`;
      try {
        const result = engine.edit(sheet, ref, input);
        setProblem(null);
        setStale(result.stale.length);
        apply(result.changed);
        undoStack.current.push({ sheet, ref, input: before });
        setEdits((list) => [...list, { sheet, ref, input }]);
        setSaved(null);
      } catch (error) {
        setProblem(error instanceof Error ? error.message : String(error));
      }
    },
    [apply, inputOf],
  );

  const undo = useCallback(() => {
    const engine = calc.current;
    const last = undoStack.current.pop();
    if (!engine || !last) return;
    const result = engine.edit(last.sheet, last.ref, last.input);
    apply(result.changed);
    setEdits((list) => list.slice(0, -1));
    setSaved(null);
  }, [apply]);

  const save = useCallback(async () => {
    setSaving(true);
    setProblem(null);
    try {
      setSaved(await core.saveSheetCopy({ documentId, edits }));
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  }, [documentId, edits]);

  return {
    on,
    starting,
    revision,
    edits,
    stale,
    problem,
    saving,
    saved,
    start,
    stop: () => setOn(false),
    inputOf,
    commit,
    undo,
    save,
  };
}
