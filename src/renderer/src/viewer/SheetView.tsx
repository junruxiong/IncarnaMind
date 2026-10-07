import {
  type CSSProperties,
  type ReactNode,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { Document } from "../../../core/api";
import { extractCsv } from "../../../core/documents/formats/csv";
import {
  gridUnits,
  MAX_ROWS,
  type Sheet,
  type Workbook,
} from "../../../core/documents/formats/grid";
import { readWorkbook } from "../../../core/documents/formats/xlsx";
import { anchorsCovered, quoteInUnits } from "../../../shared/locations";
import { columnName, parseCellRef, type TextUnit } from "../../../shared/units";
import { useT } from "../i18n";
import type { ViewerTarget } from "../store";
import { FileStates } from "./FileStates";
import { CitationMark, quoteMarkOf, quoteTone } from "./quoteMark";
import { useDocumentFile } from "./useDocumentFile";
import { ViewerHeader } from "./ViewerHeader";

/** A row's height and the default column width, in CSS pixels; rows above and below the view drawn. */
const ROW_HEIGHT = 24;
const COLUMN_WIDTH = 96;
const ROW_HEADER_WIDTH = 48;
const OVERSCAN = 12;
/** The most columns drawn. */
const MAX_COLUMNS = 256;

interface Loaded {
  workbook: Workbook;
  units: TextUnit[];
}

const readXlsx = async (bytes: Uint8Array): Promise<Loaded> => {
  const workbook = await readWorkbook(bytes);
  return { workbook, units: gridUnits(workbook.sheets, true) };
};

const readCsv = (bytes: Uint8Array): Loaded => extractCsv(bytes);

/**
 * An Excel workbook or a CSV file (ADR-0011): its sheets as a grid, read with
 * the code that indexed them, with sheet tabs at the bottom as in Excel.
 * Opened at a Citation, it shows the cited rows' sheet, washes the cells the
 * quote covers, scrolls them into view and sets the Citation's mark beside
 * them. Only the rows that are in view are drawn.
 */
export function SheetView({ document, target }: { document: Document; target: ViewerTarget }) {
  const loaded = useDocumentFile(document.id, document.kind === "csv" ? readCsv : readXlsx);
  return (
    <FileStates
      loaded={loaded}
      quote={target.quote}
      ready={(value) => <Sheets loaded={value} name={document.name} target={target} />}
    />
  );
}

/** A cell by sheet and reference: "Revenue!B12". */
const cellKey = (sheet: string, ref: string) => `${sheet}!${ref}`;

/** The cells a quote covers, and the sheet they are on; null if it isn't found. */
function findCells(
  units: readonly TextUnit[],
  quote: string,
  from: number,
  to: number,
): { sheet: string; cells: Set<string> } | null {
  const cited = units.filter((unit) => unit.page >= from && unit.page <= to);
  const tries = [cited, ...units.filter((unit) => !cited.includes(unit)).map((unit) => [unit])];
  for (const tried of tries) {
    if (tried.length === 0) continue;
    const ranges = quoteInUnits(tried, quote);
    if (!ranges) continue;
    const covered = anchorsCovered(tried, ranges);
    if (covered.length === 0) continue;
    const sheet = covered[0]?.unit.label?.sheet ?? "";
    return {
      sheet,
      cells: new Set(covered.map(({ unit, target }) => cellKey(unit.label?.sheet ?? "", target))),
    };
  }
  return null;
}

function Sheets({ loaded, name, target }: { loaded: Loaded; name: string; target: ViewerTarget }) {
  const t = useT();
  const { workbook, units } = loaded;
  const sheets = workbook.sheets;
  const from = target.pageFrom ?? 1;
  const to = target.pageTo ?? from;
  const found = useMemo(
    () => (target.quote ? findCells(units, target.quote, from, to) : null),
    [units, target.quote, from, to],
  );
  const citedUnit = units.find((unit) => unit.page === from);
  const citedSheet = found?.sheet ?? citedUnit?.label?.sheet ?? null;
  const [active, setActive] = useState(() =>
    Math.max(0, citedSheet === null ? 0 : sheets.findIndex((sheet) => sheet.name === citedSheet)),
  );
  const appliedRequest = useRef<number | null>(null);
  const [scrollTo, setScrollTo] = useState<{ row: number; request: number } | null>(null);

  // Each open request shows the cited sheet and its rows.
  useLayoutEffect(() => {
    if (appliedRequest.current === target.request) return;
    appliedRequest.current = target.request;
    if (target.pageFrom === undefined && !found) return;
    const index = sheets.findIndex((sheet) => sheet.name === (citedSheet ?? ""));
    if (index >= 0) setActive(index);
    const rows = found
      ? [...found.cells].map((key) => parseCellRef(key.slice(key.lastIndexOf("!") + 1))?.row ?? 1)
      : [citedUnit?.label?.from ?? 1];
    setScrollTo({ row: Math.min(...rows), request: target.request });
  }, [target.request, target.pageFrom, found, citedSheet, citedUnit, sheets]);

  const sheet = sheets[active];
  const mark = useMemo(() => quoteMarkOf(target.citation), [target.citation]);
  return (
    <>
      <ViewerHeader />
      <div
        data-testid="viewer-sheet"
        data-quote-tone={quoteTone(target.citation)}
        className="viewer-sheet flex min-h-0 flex-1 flex-col"
      >
        {sheet && sheet.cells.length > 0 ? (
          <Grid
            key={active}
            sheet={sheet}
            highlighted={found?.sheet === sheet.name ? found.cells : null}
            mark={found?.sheet === sheet.name ? mark : null}
            scrollTo={sheet.name === (citedSheet ?? "") ? scrollTo : null}
          />
        ) : (
          <p data-testid="viewer-sheet-empty" className="viewer-sheet-empty">
            {t("viewer.sheet.empty")}
          </p>
        )}
        {workbook.truncated && (
          <p data-testid="viewer-sheet-truncated" className="viewer-sheet-notice">
            {t("viewer.sheet.truncated", { rows: MAX_ROWS.toLocaleString() })}
          </p>
        )}
        <div
          role="tablist"
          aria-label={t("viewer.sheet.tabs")}
          data-testid="viewer-sheet-tabs"
          className="viewer-sheet-tabs"
        >
          {sheets.map((each, index) => (
            <button
              key={each.name || index}
              type="button"
              role="tab"
              aria-selected={index === active}
              data-testid="viewer-sheet-tab"
              onClick={() => setActive(index)}
              className="viewer-sheet-tab"
            >
              {each.name || name || t("viewer.sheet.csv")}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

interface MarkPlace {
  top: number;
  left: number;
}

/** Rows not drawn, above or below the view: one empty row of their height. */
function Spacer({ rows, columns }: { rows: number; columns: number }) {
  return (
    <tr>
      <td colSpan={columns} className="viewer-grid-spacer" style={{ height: rows * ROW_HEIGHT }} />
    </tr>
  );
}

function Grid({
  sheet,
  highlighted,
  mark,
  scrollTo,
}: {
  sheet: Sheet;
  highlighted: ReadonlySet<string> | null;
  mark: ReturnType<typeof quoteMarkOf>;
  scrollTo: { row: number; request: number } | null;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const table = useRef<HTMLTableElement>(null);
  const [view, setView] = useState({ top: 0, height: 600, left: 0, width: 600 });
  const [place, setPlace] = useState<MarkPlace | null>(null);
  const [markWidth, setMarkWidth] = useState(40);
  const applied = useRef<number | null>(null);
  const rowCount = Math.max(sheet.rowCount, 1);
  const columnCount = Math.min(Math.max(sheet.columnCount, 1), MAX_COLUMNS);
  const cells = useMemo(
    () => new Map(sheet.cells.map((cell) => [`${columnName(cell.column)}${cell.row}`, cell])),
    [sheet],
  );
  const highlightedRows = useMemo(() => {
    const rows = new Set<number>();
    for (const key of highlighted ?? []) {
      const row = parseCellRef(key.slice(key.lastIndexOf("!") + 1))?.row;
      if (row !== undefined) rows.add(row);
    }
    return rows;
  }, [highlighted]);
  const merges = useMemo(
    () =>
      sheet.merges.flatMap((merge) => {
        const [start, end] = merge.split(":");
        const a = parseCellRef(start ?? "");
        const b = parseCellRef(end ?? start ?? "");
        return a && b ? [{ top: a.row, left: a.column, bottom: b.row, right: b.column }] : [];
      }),
    [sheet],
  );
  const widths = Array.from(
    { length: columnCount },
    (_, column) => sheet.columnWidths[column] ?? COLUMN_WIDTH,
  );

  const measure = useCallback(() => {
    const element = scroller.current;
    if (!element) return;
    setView((previous) =>
      previous.top === element.scrollTop &&
      previous.height === element.clientHeight &&
      previous.left === element.scrollLeft &&
      previous.width === element.clientWidth
        ? previous
        : {
            top: element.scrollTop,
            height: element.clientHeight,
            left: element.scrollLeft,
            width: element.clientWidth,
          },
    );
  }, []);
  useLayoutEffect(() => {
    measure();
    const element = scroller.current;
    if (!element) return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [measure]);

  // Scroll the cited rows into view, a third of the way down.
  useLayoutEffect(() => {
    const element = scroller.current;
    if (!element || !scrollTo || applied.current === scrollTo.request) return;
    applied.current = scrollTo.request;
    element.scrollTop = Math.max(0, (scrollTo.row - 1) * ROW_HEIGHT - element.clientHeight / 3);
    measure();
  }, [scrollTo, measure]);

  const first = Math.max(1, Math.floor(view.top / ROW_HEIGHT) + 1 - OVERSCAN);
  const last = Math.min(rowCount, Math.ceil((view.top + view.height) / ROW_HEIGHT) + OVERSCAN);

  // The Citation's mark, to the right of the quote's cells in its first row; while
  // those run on past the view, at the view's right edge, over the washed row.
  // biome-ignore lint/correctness/useExhaustiveDependencies: placed again as the view moves
  useLayoutEffect(() => {
    const grid = table.current;
    const box = grid?.parentElement;
    if (!grid || !box || !mark || !highlighted) {
      setPlace(null);
      return;
    }
    const marked = [...grid.querySelectorAll<HTMLElement>("td[data-quote-highlight]")];
    const top = marked[0];
    if (!top) {
      setPlace(null);
      return;
    }
    const row = top.dataset.row;
    const right = Math.max(
      ...marked
        .filter((cell) => cell.dataset.row === row)
        .map((cell) => cell.getBoundingClientRect().right),
    );
    const origin = box.getBoundingClientRect();
    const rect = top.getBoundingClientRect();
    const visibleRight = view.left + view.width - markWidth - 8;
    setPlace({
      top: rect.top - origin.top + rect.height / 2 - 9,
      left: Math.max(0, Math.min(right - origin.left + 8, visibleRight)),
    });
  }, [mark, highlighted, first, last, view.left, view.width, markWidth]);

  const rows: ReactNode[] = [];
  for (let row = first; row <= last; row++) {
    const tds: ReactNode[] = [];
    for (let column = 0; column < columnCount; column++) {
      const merge = merges.find(
        (each) =>
          row >= each.top && row <= each.bottom && column >= each.left && column <= each.right,
      );
      const anchorRow = merge ? Math.max(merge.top, first) : row;
      if (merge && (column !== merge.left || row !== anchorRow)) continue;
      const ref = `${columnName(merge ? merge.left : column)}${merge ? merge.top : row}`;
      const cell = cells.get(ref);
      const isHighlighted = highlighted?.has(cellKey(sheet.name, ref)) ?? false;
      tds.push(
        <td
          key={column}
          data-ref={ref}
          data-row={row}
          data-quote-highlight={isHighlighted ? "" : undefined}
          rowSpan={merge ? Math.min(merge.bottom, last) - anchorRow + 1 : undefined}
          colSpan={merge ? Math.min(merge.right, columnCount - 1) - merge.left + 1 : undefined}
          className={`${cell?.numeric ? "viewer-cell--number" : ""} ${
            isHighlighted ? "viewer-cell--quote" : ""
          }`}
        >
          {merge && row !== merge.top ? "" : (cell?.value ?? "")}
        </td>,
      );
    }
    rows.push(
      <tr key={row} style={{ height: ROW_HEIGHT }}>
        <th
          scope="row"
          className={highlightedRows.has(row) ? "viewer-row--cited" : undefined}
          data-row-header={row}
        >
          {row}
        </th>
        {tds}
      </tr>,
    );
  }

  const style = {
    "--grid-width": `${ROW_HEADER_WIDTH + widths.reduce((sum, width) => sum + width, 0)}px`,
  } as CSSProperties;
  return (
    <div
      ref={scroller}
      data-testid="viewer-grid"
      onScroll={measure}
      className="viewer-grid-scroll min-h-0 flex-1 overflow-auto"
    >
      <div className="relative w-max" style={style}>
        <table ref={table} className="viewer-grid">
          <colgroup>
            <col style={{ width: ROW_HEADER_WIDTH }} />
            {widths.map((width, column) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: columns are positions
              <col key={column} style={{ width }} />
            ))}
          </colgroup>
          <thead>
            <tr>
              <td className="viewer-grid-corner" />
              {widths.map((_, column) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: columns are positions
                <th key={column} scope="col">
                  {columnName(column)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {first > 1 && <Spacer rows={first - 1} columns={columnCount + 1} />}
            {rows}
            {last < rowCount && <Spacer rows={rowCount - last} columns={columnCount + 1} />}
          </tbody>
        </table>
        {mark && place && (
          <CitationMark
            mark={mark}
            onWidth={setMarkWidth}
            style={{ top: place.top, left: place.left }}
          />
        )}
      </div>
    </div>
  );
}
