/** A chart as a standalone SVG document, on white, for an export's picture (#436). */
import { CHART_FONT_SIZE, type ChartInk, type ChartLayout } from "./chartLayout";

/** The light theme's values of the tokens the editor draws charts with (styles.css, `.chart-*`): a page is white. */
export const LIGHT_CHART_COLOURS = {
  sheet: "#FFFFFF",
  rule: "#E3E5E9",
  ruleStrong: "#D5D8DD",
  axis: "#656B74",
  label: "#4A4F57",
  prior: "#B9BEC6",
  now: "#4A4F57",
  categories: [
    "#3DB8C9",
    "#9B6CF0",
    "#F7A03C",
    "#F5C836",
    "#1B5962",
    "#5921BD",
    "#734714",
    "#624F11",
    "#9AA0A8",
  ],
} as const;

/** The app's font (DESIGN.md, Typography): the picture is drawn on this computer, with its fonts. */
const FONT_FAMILY =
  '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif';

const escapeXml = (text: string) =>
  text.replace(/[&<>"]/g, (char) =>
    char === "&" ? "&amp;" : char === "<" ? "&lt;" : char === ">" ? "&gt;" : "&quot;",
  );

const round = (value: number) => Number(value.toFixed(1));

export function inkColour(ink: ChartInk): string {
  if (ink === "prior") return LIGHT_CHART_COLOURS.prior;
  if (ink === "now") return LIGHT_CHART_COLOURS.now;
  const index = Number(ink.slice("category".length)) || 0;
  const { categories } = LIGHT_CHART_COLOURS;
  return categories[index % categories.length] ?? LIGHT_CHART_COLOURS.now;
}

export function chartSvg(layout: ChartLayout, scale = 1): string {
  const colours = LIGHT_CHART_COLOURS;
  const parts: string[] = [
    `<rect x="0" y="0" width="${layout.width}" height="${layout.height}" fill="${colours.sheet}"/>`,
  ];
  for (const line of layout.grid) {
    parts.push(
      `<line x1="${round(line.x1)}" x2="${round(line.x2)}" y1="${round(line.y)}" y2="${round(line.y)}" stroke="${line.base ? colours.ruleStrong : colours.rule}" stroke-width="1"/>`,
    );
  }
  for (const line of layout.lines) {
    for (const run of line.runs) {
      const points = run.map(([x, y]) => `${round(x)},${round(y)}`).join(" ");
      parts.push(
        `<polyline points="${points}" fill="none" stroke="${inkColour(line.ink)}" stroke-width="2" stroke-linejoin="round"/>`,
      );
    }
  }
  for (const point of layout.points) {
    const fill = inkColour(point.ink);
    const { mark } = point;
    if (mark.shape === "bar") {
      parts.push(
        `<rect x="${round(mark.x)}" y="${round(mark.y)}" width="${round(mark.width)}" height="${round(mark.height)}" rx="${mark.width >= 6 ? 2 : 0}" fill="${fill}"/>`,
      );
    } else if (mark.shape === "dot") {
      parts.push(
        `<circle cx="${round(mark.cx)}" cy="${round(mark.cy)}" r="${mark.r}" fill="${fill}"/>`,
      );
    } else {
      parts.push(
        `<path d="${mark.path}" fill="${fill}" stroke="${colours.sheet}" stroke-width="2"/>`,
      );
    }
  }
  for (const mark of layout.legendMarks) {
    parts.push(
      `<rect x="${round(mark.x)}" y="${round(mark.y)}" width="12" height="12" rx="2" fill="${inkColour(mark.ink)}"/>`,
    );
  }
  const texts = [...layout.texts, ...layout.legendMarks.map((mark) => mark.text)];
  for (const text of texts) {
    parts.push(
      `<text x="${round(text.x)}" y="${round(text.y)}" text-anchor="${text.anchor}" fill="${text.role === "axis" ? colours.axis : colours.label}">${escapeXml(text.text)}</text>`,
    );
  }
  const width = Math.round(layout.width * scale);
  const height = Math.round(layout.height * scale);
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${layout.width} ${layout.height}" font-family='${FONT_FAMILY}' font-size="${CHART_FONT_SIZE}">` +
    `${parts.join("")}</svg>`
  );
}
