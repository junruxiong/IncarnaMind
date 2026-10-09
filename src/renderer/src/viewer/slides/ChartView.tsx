import { Children, type ReactNode } from "react";
import { formatNumber } from "../../../../core/documents/formats/numbers";
import type { ChartData } from "../../../../core/documents/formats/pptxDrawing";
import { niceScale } from "./chartScale";

/** Chart text: Office's default 10 pt, in CSS pixels, and its title's 14 pt. */
const TEXT = 13.3;
const TITLE = 18.6;
const FONT = "'Calibri', 'Carlito', 'Helvetica Neue', Arial, sans-serif";
const GRID = "#D9D9D9";
const AXIS = "#BFBFBF";

/** Roughly how wide a label is, for the space it takes: a chart isn't measured. */
const textWidth = (text: string, size = TEXT) => text.length * size * 0.55;

const plain = (value: number) => {
  const rounded = Math.round(value * 1e6) / 1e6;
  return Math.abs(rounded) >= 1000 ? rounded.toLocaleString("en-US") : String(rounded);
};

/** How a chart's text and lines look: the file's own sizes, colours and number formats, else Office's. */
interface ChartStyle {
  size: number;
  colour: string;
  labelSize: number;
  labelColour: string;
  grid: string;
  axis: string;
  /** A data label's text, and a value axis tick's. */
  label(value: number): string;
  tick(value: number): string;
}

function chartStyle(chart: ChartData): ChartStyle {
  const size = chart.textSize ?? TEXT;
  const colour = chart.textColour ?? "#595959";
  const formatted = (code: string | null) => (value: number) =>
    code ? formatNumber(value, code) : plain(value);
  return {
    size,
    colour,
    labelSize: chart.labelSize ?? size,
    labelColour: chart.labelColour ?? colour,
    grid: chart.gridColour ?? GRID,
    axis: chart.axisColour ?? AXIS,
    label: formatted(chart.labelFormat),
    tick: formatted(chart.axisFormat),
  };
}

/**
 * A chart drawn from its cached values, as Office draws it by default:
 * columns, bars, lines and areas on a value axis with gridlines, pies and
 * doughnuts in their slices' colours, a title and a legend. Text is SVG text.
 */
export function ChartView({
  chart,
  width,
  height,
}: {
  chart: ChartData;
  width: number;
  height: number;
}) {
  const style = chartStyle(chart);
  const { colour, size } = style;
  const pad = 8;
  let top = pad;
  let bottom = height - pad;
  let left = pad;
  let right = width - pad;
  const title = chart.title ? (
    <text x={width / 2} y={top + TITLE} textAnchor="middle" fontSize={TITLE} fill={colour}>
      {chart.title}
    </text>
  ) : null;
  if (title) top += TITLE * 1.6;

  const round = chart.kind === "pie" || chart.kind === "doughnut";
  const legendEntries = round
    ? chart.categories.map((name, at) => ({
        name,
        colour: chart.series[0]?.pointColours[at] ?? chart.series[0]?.colour ?? "#4472C4",
      }))
    : chart.series.map((series) => ({ name: series.name, colour: series.colour }));
  let legend: ReactNode = null;
  if (chart.legend && legendEntries.length > 0) {
    const swatch = size * 0.7;
    if (chart.legend === "bottom" || chart.legend === "top") {
      const widths = legendEntries.map((entry) => swatch + 6 + textWidth(entry.name, size) + 16);
      const total = widths.reduce((sum, each) => sum + each, 0);
      const y = chart.legend === "bottom" ? bottom - size : top + size * 0.2;
      let x = Math.max(left, width / 2 - total / 2);
      legend = legendEntries.map((entry, at) => {
        const at0 = x;
        x += widths[at] ?? 0;
        return (
          // biome-ignore lint/suspicious/noArrayIndexKey: entries never reorder
          <g key={at}>
            <rect x={at0} y={y} width={swatch} height={swatch} fill={entry.colour} />
            <text x={at0 + swatch + 6} y={y + swatch} fontSize={size} fill={colour}>
              {entry.name}
            </text>
          </g>
        );
      });
      if (chart.legend === "bottom") bottom -= size * 2;
      else top += size * 1.8;
    } else {
      const widest = Math.max(...legendEntries.map((entry) => textWidth(entry.name, size)));
      const blockWidth = swatch + 6 + widest;
      const x = chart.legend === "right" ? right - blockWidth : left;
      const startY = (top + bottom) / 2 - (legendEntries.length * size * 1.5) / 2;
      legend = legendEntries.map((entry, at) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: entries never reorder
        <g key={at}>
          <rect
            x={x}
            y={startY + at * size * 1.5}
            width={swatch}
            height={swatch}
            fill={entry.colour}
          />
          <text
            x={x + swatch + 6}
            y={startY + at * size * 1.5 + swatch}
            fontSize={size}
            fill={colour}
          >
            {entry.name}
          </text>
        </g>
      ));
      if (chart.legend === "right") right -= blockWidth + 12;
      else left += blockWidth + 12;
    }
  }

  const plot = round
    ? roundChart(chart, { left, top, right, bottom }, style)
    : axisChart(chart, { left, top, right, bottom }, style);
  return (
    <svg
      className="slide-chart"
      width={Math.max(width, 1)}
      height={Math.max(height, 1)}
      fontFamily={FONT}
      role="img"
      aria-label={chart.title ?? undefined}
    >
      {title}
      {plot}
      {legend}
    </svg>
  );
}

interface Area {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

function roundChart(chart: ChartData, area: Area, style: ChartStyle) {
  const series = chart.series[0];
  if (!series) return null;
  const values = series.values.map((value) => Math.max(0, value ?? 0));
  const total = values.reduce((sum, each) => sum + each, 0);
  if (total <= 0) return null;
  const cx = (area.left + area.right) / 2;
  const cy = (area.top + area.bottom) / 2;
  const radius = Math.max(4, Math.min(area.right - area.left, area.bottom - area.top) / 2 - 4);
  const hole = chart.kind === "doughnut" ? radius * 0.5 : 0;
  let angle = -Math.PI / 2;
  return values.map((value, at) => {
    const sweep = (value / total) * Math.PI * 2;
    const start = angle;
    angle += sweep;
    const end = angle;
    const point = (r: number, a: number) => `${cx + r * Math.cos(a)} ${cy + r * Math.sin(a)}`;
    const large = sweep > Math.PI ? 1 : 0;
    const d =
      sweep >= Math.PI * 2 - 1e-6
        ? `M${point(radius, 0)} A${radius} ${radius} 0 1 1 ${point(radius, Math.PI)} A${radius} ${radius} 0 1 1 ${point(radius, 0)} Z`
        : hole > 0
          ? `M${point(radius, start)} A${radius} ${radius} 0 ${large} 1 ${point(radius, end)} L${point(hole, end)} A${hole} ${hole} 0 ${large} 0 ${point(hole, start)} Z`
          : `M${cx} ${cy} L${point(radius, start)} A${radius} ${radius} 0 ${large} 1 ${point(radius, end)} Z`;
    const mid = (start + end) / 2;
    const labelRadius = hole > 0 ? (radius + hole) / 2 : radius * 0.65;
    return (
      // biome-ignore lint/suspicious/noArrayIndexKey: slices never reorder
      <g key={at}>
        <path
          d={d}
          fill={series.pointColours[at] ?? series.colour}
          stroke="#FFFFFF"
          strokeWidth={1.5}
        />
        {chart.dataLabels && value > 0 && (
          <text
            x={cx + labelRadius * Math.cos(mid)}
            y={cy + labelRadius * Math.sin(mid) + style.labelSize / 3}
            textAnchor="middle"
            fontSize={style.labelSize}
            fill={style.labelColour}
          >
            {style.label(series.values[at] ?? 0)}
          </text>
        )}
      </g>
    );
  });
}

function axisChart(chart: ChartData, area: Area, style: ChartStyle) {
  const horizontal = chart.kind === "bar";
  const stacked = chart.grouping === "stacked" || chart.grouping === "percentStacked";
  const percent = chart.grouping === "percentStacked";
  const count = chart.categories.length;
  if (count === 0) return null;
  const totals = chart.categories.map((_, at) =>
    chart.series.reduce((sum, series) => sum + Math.abs(series.values[at] ?? 0), 0),
  );
  const valueAt = (series: number, at: number) => {
    const value = chart.series[series]?.values[at] ?? 0;
    return percent ? (totals[at] ? (value / (totals[at] as number)) * 100 : 0) : value;
  };
  let lowest = 0;
  let highest = 0;
  for (let at = 0; at < count; at++) {
    if (stacked) {
      let positive = 0;
      let negative = 0;
      chart.series.forEach((_, s) => {
        const value = valueAt(s, at);
        if (value >= 0) positive += value;
        else negative += value;
      });
      highest = Math.max(highest, positive);
      lowest = Math.min(lowest, negative);
    } else {
      chart.series.forEach((_, s) => {
        const value = valueAt(s, at);
        highest = Math.max(highest, value);
        lowest = Math.min(lowest, value);
      });
    }
  }
  const scale = niceScale(lowest, highest);
  const ticks: number[] = [];
  for (let value = scale.min; value <= scale.max + scale.step / 2; value += scale.step)
    ticks.push(value);
  const tickLabels = ticks.map((value) => (percent ? `${style.tick(value)}%` : style.tick(value)));

  // Room for the axes' labels.
  const valueLabelWidth = Math.max(...tickLabels.map((label) => textWidth(label, style.size))) + 8;
  const categoryLabelWidth =
    Math.max(...chart.categories.map((label) => textWidth(label, style.size))) + 8;
  const plot = {
    left: area.left + (horizontal ? categoryLabelWidth : valueLabelWidth),
    right: area.right - 4,
    top: area.top + 4,
    bottom: area.bottom - style.size * 1.6,
  };
  if (plot.right - plot.left < 10 || plot.bottom - plot.top < 10) return null;
  const valueSpan = scale.max - scale.min || 1;
  // Where a value is along the value axis, and where a category's band starts.
  const valuePos = (value: number) =>
    horizontal
      ? plot.left + ((value - scale.min) / valueSpan) * (plot.right - plot.left)
      : plot.bottom - ((value - scale.min) / valueSpan) * (plot.bottom - plot.top);
  const bandSize = (horizontal ? plot.bottom - plot.top : plot.right - plot.left) / count;
  // Office draws a bar chart's first category at the bottom.
  const bandStart = (at: number) =>
    horizontal ? plot.bottom - (at + 1) * bandSize : plot.left + at * bandSize;
  const zero = valuePos(Math.max(scale.min, Math.min(scale.max, 0)));

  const parts: ReactNode[] = [];
  // Gridlines and value labels.
  ticks.forEach((value, at) => {
    const position = valuePos(value);
    if (chart.gridlines || value === 0) {
      parts.push(
        horizontal ? (
          <line
            x1={position}
            x2={position}
            y1={plot.top}
            y2={plot.bottom}
            stroke={style.grid}
            strokeWidth={1}
          />
        ) : (
          <line
            x1={plot.left}
            x2={plot.right}
            y1={position}
            y2={position}
            stroke={style.grid}
            strokeWidth={1}
          />
        ),
      );
    }
    parts.push(
      horizontal ? (
        <text
          x={position}
          y={plot.bottom + style.size * 1.3}
          textAnchor="middle"
          fontSize={style.size}
          fill={style.colour}
        >
          {tickLabels[at]}
        </text>
      ) : (
        <text
          x={plot.left - 6}
          y={position + style.size / 3}
          textAnchor="end"
          fontSize={style.size}
          fill={style.colour}
        >
          {tickLabels[at]}
        </text>
      ),
    );
  });
  // Category labels and the category axis.
  chart.categories.forEach((label, at) => {
    const middle = bandStart(at) + bandSize / 2;
    parts.push(
      horizontal ? (
        <text
          x={plot.left - 6}
          y={middle + style.size / 3}
          textAnchor="end"
          fontSize={style.size}
          fill={style.colour}
        >
          {label}
        </text>
      ) : (
        <text
          x={middle}
          y={plot.bottom + style.size * 1.3}
          textAnchor="middle"
          fontSize={style.size}
          fill={style.colour}
        >
          {label}
        </text>
      ),
    );
  });
  parts.push(
    horizontal ? (
      <line
        x1={zero}
        x2={zero}
        y1={plot.top}
        y2={plot.bottom}
        stroke={style.axis}
        strokeWidth={1}
      />
    ) : (
      <line
        x1={plot.left}
        x2={plot.right}
        y1={zero}
        y2={zero}
        stroke={style.axis}
        strokeWidth={1}
      />
    ),
  );

  if (chart.kind === "column" || chart.kind === "bar") {
    const seriesCount = chart.series.length;
    // A gap of 150% of a bar between clusters, as Office does by default.
    const barSize = stacked ? bandSize / 2.5 : bandSize / (seriesCount + 1.5);
    const stacks = chart.categories.map(() => ({ positive: 0, negative: 0 }));
    chart.series.forEach((series, s) => {
      chart.categories.forEach((_, at) => {
        const value = valueAt(s, at);
        let from = 0;
        let to = value;
        if (stacked) {
          const stack = stacks[at] as { positive: number; negative: number };
          from = value >= 0 ? stack.positive : stack.negative;
          to = from + value;
          if (value >= 0) stack.positive = to;
          else stack.negative = to;
        }
        const offset = stacked
          ? (bandSize - barSize) / 2
          : (bandSize - barSize * seriesCount) / 2 +
            (horizontal ? seriesCount - 1 - s : s) * barSize;
        const a = valuePos(from);
        const b = valuePos(to);
        const fill = series.pointColours[at] ?? series.colour;
        const along = bandStart(at) + offset;
        const rect = horizontal
          ? { x: Math.min(a, b), y: along, width: Math.abs(b - a), height: barSize }
          : { x: along, y: Math.min(a, b), width: barSize, height: Math.abs(b - a) };
        parts.push(<rect {...rect} fill={fill} />);
        if (chart.dataLabels && series.values[at] !== null) {
          parts.push(
            horizontal ? (
              <text
                x={Math.max(a, b) + 4}
                y={along + barSize / 2 + style.labelSize / 3}
                fontSize={style.labelSize}
                fill={style.labelColour}
              >
                {style.label(series.values[at] ?? 0)}
              </text>
            ) : (
              <text
                x={along + barSize / 2}
                y={Math.min(a, b) - 4}
                textAnchor="middle"
                fontSize={style.labelSize}
                fill={style.labelColour}
              >
                {style.label(series.values[at] ?? 0)}
              </text>
            ),
          );
        }
      });
    });
  } else {
    // Lines and areas: a point in the middle of each category's band.
    const running = chart.categories.map(() => 0);
    chart.series.forEach((series, s) => {
      const points = chart.categories.map((_, at) => {
        const value = valueAt(s, at);
        const level = stacked ? (running[at] as number) + value : value;
        if (stacked) running[at] = level;
        return [bandStart(at) + bandSize / 2, valuePos(level)] as const;
      });
      const line = points.map(([x, y], at) => `${at === 0 ? "M" : "L"}${x} ${y}`).join(" ");
      if (chart.kind === "area") {
        const first = points[0];
        const last = points[points.length - 1];
        if (first && last) {
          parts.push(
            <path
              d={`${line} L${last[0]} ${zero} L${first[0]} ${zero} Z`}
              fill={series.colour}
              fillOpacity={0.85}
            />,
          );
        }
      } else {
        parts.push(
          <path
            d={line}
            fill="none"
            stroke={series.colour}
            strokeWidth={3}
            strokeLinejoin="round"
          />,
        );
        points.forEach(([x, y]) => {
          parts.push(<circle cx={x} cy={y} r={3.5} fill={series.colour} />);
        });
      }
      if (chart.dataLabels) {
        points.forEach(([x, y], at) => {
          parts.push(
            <text
              x={x}
              y={y - 8}
              textAnchor="middle"
              fontSize={style.labelSize}
              fill={style.labelColour}
            >
              {style.label(series.values[at] ?? 0)}
            </text>,
          );
        });
      }
    });
  }
  // Keyed by their place: a chart is drawn once and never reorders.
  return <g>{Children.toArray(parts)}</g>;
}
