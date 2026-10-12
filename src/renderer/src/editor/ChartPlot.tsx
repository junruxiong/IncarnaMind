import { type KeyboardEvent, useLayoutEffect, useMemo, useRef, useState } from "react";
import { type ChartLayout, type LaidOutPoint, layoutChart } from "../../../shared/chartLayout";
import type { ChartAttributes } from "../../../shared/charts";

export interface PointRef {
  row: number;
  series: number;
}

export const samePoint = (a: PointRef | null, b: PointRef | null) =>
  a !== null && b !== null && a.row === b.row && a.series === b.series;

export interface PointEvents {
  peek(point: PointRef, anchor: Element, source: "pointer" | "focus", on: boolean): void;
  activate(point: PointRef, anchor: Element): void;
  label(point: PointRef): string;
}

let canvas: CanvasRenderingContext2D | null | undefined;
function measureText(text: string): number {
  if (canvas === undefined) {
    canvas = document.createElement("canvas").getContext("2d");
    if (canvas) {
      const family = getComputedStyle(document.body).fontFamily || "system-ui, sans-serif";
      canvas.font = `12px ${family}`;
    }
  }
  return canvas ? canvas.measureText(text).width : text.length * 7.2;
}

const MOVES = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]);

function nextPoint(
  points: readonly LaidOutPoint[],
  from: number,
  key: string,
  byRow: boolean,
): number | null {
  const here = points[from];
  if (!here || !MOVES.has(key)) return null;
  if (key === "Home") return 0;
  if (key === "End") return points.length - 1;
  const forward = key === "ArrowRight" || key === "ArrowDown";
  const across = key === "ArrowLeft" || key === "ArrowRight";
  if (!byRow || across) {
    if (!byRow) return Math.max(0, Math.min(points.length - 1, from + (forward ? 1 : -1)));
    const same = points
      .map((point, index) => ({ point, index }))
      .filter(({ point }) => point.series === here.series);
    const at = same.findIndex(({ index }) => index === from);
    return same[at + (forward ? 1 : -1)]?.index ?? from;
  }
  // Up and down: the series drawn above or below at this row, or the nearest row it has.
  const series = [...new Set(points.map((point) => point.series))];
  const target = series[series.indexOf(here.series) + (forward ? 1 : -1)];
  if (target === undefined) return from;
  let best: number | null = null;
  points.forEach((point, index) => {
    if (point.series !== target) return;
    const nearer =
      best === null ||
      Math.abs(point.row - here.row) < Math.abs((points[best]?.row ?? 0) - here.row);
    if (nearer) best = index;
  });
  return best ?? from;
}

const fillClass = (point: LaidOutPoint) => `chart-fill-${point.ink}`;

/**
 * Drawn at the width its column has, so its labels stay 12 px. Its values
 * are one Tab stop: the arrows move along a series and across series.
 */
export function ChartPlot({
  chart,
  language,
  seriesName,
  active,
  events,
  label,
}: {
  chart: ChartAttributes;
  language: string;
  seriesName(index: number): string;
  active: PointRef | null;
  events: PointEvents;
  label: string;
}) {
  const box = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const element = box.current;
    if (!element) return;
    const measure = () => setWidth(Math.floor(element.clientWidth));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const layout: ChartLayout | null = useMemo(
    () =>
      width > 0 ? layoutChart(chart, { width, language, seriesName, measure: measureText }) : null,
    [chart, width, language, seriesName],
  );
  const points = layout?.points ?? [];

  const [current, setCurrent] = useState(0);
  const at = Math.min(current, Math.max(0, points.length - 1));
  const marks = useRef(new Map<string, SVGElement>());

  const onKeyDown = (event: KeyboardEvent<SVGElement>, index: number) => {
    const point = points[index];
    if (!point) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      events.activate(point, event.currentTarget);
      return;
    }
    const byRow = chart.kind === "bar" || chart.kind === "line";
    const next = nextPoint(points, index, event.key, byRow);
    if (next === null) return;
    event.preventDefault();
    // At once, not after the next render: a second key pressed before it goes from here.
    const target = points[next];
    if (target) marks.current.get(`${target.row}:${target.series}`)?.focus();
    setCurrent(next);
  };

  // Drawn after every slice, so its neighbours don't cover the ring.
  const ringed = points.find((point) => samePoint(point, active))?.mark;

  const size = layout && {
    width: layout.width,
    height: layout.height,
    viewBox: `0 0 ${layout.width} ${layout.height}`,
  };
  return (
    <figure ref={box} className="chart-plot" data-chart-ui="">
      {layout && size && (
        <>
          {/* The axes, the lines and the key, under the values: what their own names already say. */}
          <svg {...size} aria-hidden="true" className="chart-drawing">
            {layout.grid.map((line) => (
              <line
                key={`grid-${line.y}`}
                className={line.base ? "chart-base" : "chart-grid"}
                x1={line.x1}
                x2={line.x2}
                y1={line.y}
                y2={line.y}
              />
            ))}
            {layout.texts.map((text, index) => (
              <text
                // biome-ignore lint/suspicious/noArrayIndexKey: labels have no identity but their place.
                key={`text-${index}`}
                className={text.role === "axis" ? "chart-axis" : "chart-label"}
                x={text.x}
                y={text.y}
                textAnchor={text.anchor}
              >
                {text.text}
              </text>
            ))}
            {layout.lines.flatMap((line) =>
              line.runs.map((run, index) => (
                <polyline
                  // biome-ignore lint/suspicious/noArrayIndexKey: a line's runs have no identity but their place.
                  key={`line-${line.series}-${index}`}
                  className={`chart-line chart-stroke-${line.ink}`}
                  points={run.map(([x, y]) => `${x},${y}`).join(" ")}
                />
              )),
            )}
            {layout.legendMarks.map((mark) => (
              <g key={`key-${mark.x}-${mark.y}`}>
                <rect
                  className={`chart-fill-${mark.ink}`}
                  x={mark.x}
                  y={mark.y}
                  width={12}
                  height={12}
                  rx={2}
                />
                <text className="chart-label" x={mark.text.x} y={mark.text.y}>
                  {mark.text.text}
                </text>
              </g>
            ))}
          </svg>
          {/* The values, over the drawing: each its own button with its name. */}
          <svg {...size} className="chart-marks" data-testid="chart-plot" data-kind={layout.kind}>
            <title>{label}</title>
            {points.map((point, index) => {
              const key = `${point.row}:${point.series}`;
              const common = {
                ref: (element: SVGElement | null) => {
                  if (element) marks.current.set(key, element);
                  else marks.current.delete(key);
                },
                className: `chart-point ${fillClass(point)}`,
                "data-testid": "chart-point",
                "data-point": key,
                "data-active": samePoint(point, active) || undefined,
                role: "button",
                tabIndex: index === at ? 0 : -1,
                "aria-label": events.label(point),
                "aria-haspopup": "dialog" as const,
                onPointerEnter: (event: { currentTarget: Element }) =>
                  events.peek(point, event.currentTarget, "pointer", true),
                onPointerLeave: (event: { currentTarget: Element }) =>
                  events.peek(point, event.currentTarget, "pointer", false),
                onFocus: (event: { currentTarget: Element }) => {
                  setCurrent(index);
                  events.peek(point, event.currentTarget, "focus", true);
                },
                onBlur: (event: { currentTarget: Element }) =>
                  events.peek(point, event.currentTarget, "focus", false),
                onClick: (event: { currentTarget: Element }) =>
                  events.activate(point, event.currentTarget),
                onKeyDown: (event: KeyboardEvent<SVGElement>) => onKeyDown(event, index),
              };
              const { mark } = point;
              if (mark.shape === "bar") {
                return (
                  <rect
                    key={key}
                    {...common}
                    x={mark.x}
                    y={mark.y}
                    width={mark.width}
                    height={mark.height}
                    rx={mark.width >= 6 ? 2 : 0}
                  />
                );
              }
              if (mark.shape === "dot") {
                return <circle key={key} {...common} cx={mark.cx} cy={mark.cy} r={mark.r} />;
              }
              return <path key={key} {...common} d={mark.path} />;
            })}
            {ringed?.shape === "slice" && <path className="chart-ring" d={ringed.path} />}
          </svg>
        </>
      )}
    </figure>
  );
}
