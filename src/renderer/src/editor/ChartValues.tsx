import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import type { CitationCheck } from "../../../core/api";
import { type ChartAttributes, type ChartValue, drawnSeries } from "../../../shared/charts";
import { ChartCheck } from "./ChartCheck";
import { type PointEvents, type PointRef, samePoint } from "./ChartPlot";

/**
 * "Show as table" (#436): every value the chart draws, with its check, as a
 * table, the way to read the chart without a pointer. It is one Tab stop:
 * the arrow keys move between values, each of which shows its card (its
 * source, and Open) on focus, as the chart's points do.
 */
export function ChartValues({
  chart,
  seriesName,
  labelHeading,
  active,
  events,
  checkOf,
}: {
  chart: ChartAttributes;
  seriesName(index: number): string;
  labelHeading: string;
  active: PointRef | null;
  events: PointEvents;
  /** A value's check (null: it has no Citation), and what it means, for its mark. */
  checkOf(value: ChartValue): { check: CitationCheck | null; label: string };
}) {
  const drawn = drawnSeries(chart);
  const rows = chart.data.rows;
  /** Each value shown, in reading order: what the arrow keys move between. */
  const cells: PointRef[] = [];
  rows.forEach((row, index) => {
    for (const series of drawn) if (row.values[series]) cells.push({ row: index, series });
  });
  const [current, setCurrent] = useState(0);
  const at = Math.min(current, Math.max(0, cells.length - 1));
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const moved = useRef(false);
  useEffect(() => {
    if (!moved.current) return;
    moved.current = false;
    const cell = cells[at];
    if (cell) buttons.current.get(`${cell.row}:${cell.series}`)?.focus();
  });

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, point: PointRef) => {
    const index = cells.findIndex((cell) => samePoint(cell, point));
    let next: number | null = null;
    if (event.key === "Home") next = 0;
    else if (event.key === "End") next = cells.length - 1;
    else if (event.key === "ArrowRight") next = Math.min(cells.length - 1, index + 1);
    else if (event.key === "ArrowLeft") next = Math.max(0, index - 1);
    else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const step = event.key === "ArrowDown" ? 1 : -1;
      const found = cells.findIndex(
        (cell) => cell.series === point.series && cell.row === point.row + step,
      );
      next = found >= 0 ? found : index;
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      events.activate(point, event.currentTarget);
      return;
    }
    if (next === null) return;
    event.preventDefault();
    moved.current = true;
    setCurrent(next);
  };

  return (
    <div className="chart-values" data-chart-ui="">
      <table data-testid="chart-table">
        <thead>
          <tr>
            <th scope="col">{labelHeading}</th>
            {drawn.map((series) => (
              <th key={series} scope="col">
                {seriesName(series)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: rows have no identity but their place.
            <tr key={index}>
              <th scope="row">{row.label}</th>
              {drawn.map((series) => {
                const value = row.values[series];
                if (!value) return <td key={series} />;
                const point = { row: index, series };
                const key = `${index}:${series}`;
                const mark = checkOf(value);
                const position = cells.findIndex((cell) => samePoint(cell, point));
                return (
                  <td key={series}>
                    <button
                      type="button"
                      ref={(element) => {
                        if (element) buttons.current.set(key, element);
                        else buttons.current.delete(key);
                      }}
                      className="chart-value"
                      data-testid="chart-value"
                      data-point={key}
                      data-active={samePoint(point, active) || undefined}
                      tabIndex={position === at ? 0 : -1}
                      aria-label={events.label(point)}
                      aria-haspopup="dialog"
                      onPointerEnter={(event) =>
                        events.peek(point, event.currentTarget, "pointer", true)
                      }
                      onPointerLeave={(event) =>
                        events.peek(point, event.currentTarget, "pointer", false)
                      }
                      onFocus={(event) => {
                        setCurrent(position);
                        events.peek(point, event.currentTarget, "focus", true);
                      }}
                      onBlur={(event) => events.peek(point, event.currentTarget, "focus", false)}
                      onClick={(event) => events.activate(point, event.currentTarget)}
                      onKeyDown={(event) => onKeyDown(event, point)}
                    >
                      <span>{value.shown}</span>
                      <ChartCheck check={mark.check} label={mark.label} size="sm" />
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
