/**
 * A shape's geometry as SVG paths, at its size: PowerPoint's common preset
 * shapes (rectangles, rounded and snipped ones, ellipses, triangles, arrows,
 * chevrons, stars, callouts, flowchart shapes, connectors…) after the
 * formulas of ECMA-376's presetShapeDefinitions.xml, simplified, and custom
 * geometry. A preset this doesn't know is drawn as its rectangle.
 */
import type { CustomPath, Geometry } from "../../../../core/documents/formats/pptxDrawing";

export interface ShapePath {
  d: string;
  fill: boolean;
  stroke: boolean;
}

const n = (value: number) => (Math.round(value * 100) / 100).toString();
const polygon = (points: readonly (readonly [number, number])[]): string =>
  `${points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${n(x)} ${n(y)}`).join(" ")} Z`;

function ellipsePath(cx: number, cy: number, rx: number, ry: number): string {
  return `M${n(cx - rx)} ${n(cy)} A${n(rx)} ${n(ry)} 0 1 0 ${n(cx + rx)} ${n(cy)} A${n(rx)} ${n(ry)} 0 1 0 ${n(cx - rx)} ${n(cy)} Z`;
}

/** A rectangle with each corner rounded (r > 0) or snipped (r < 0). */
function cornerRect(w: number, h: number, corners: [number, number, number, number]): string {
  const limit = Math.min(w, h) / 2;
  const [tl, tr, br, bl] = corners.map((value) => Math.max(-limit, Math.min(limit, value))) as [
    number,
    number,
    number,
    number,
  ];
  const corner = (r: number, x: number, y: number) =>
    r > 0 ? `A${n(r)} ${n(r)} 0 0 1 ${n(x)} ${n(y)}` : `L${n(x)} ${n(y)}`;
  const a = Math.abs;
  return [
    `M${n(a(tl))} 0`,
    `L${n(w - a(tr))} 0`,
    corner(tr, w, a(tr)),
    `L${n(w)} ${n(h - a(br))}`,
    corner(br, w - a(br), h),
    `L${n(a(bl))} ${n(h)}`,
    corner(bl, 0, h - a(bl)),
    `L0 ${n(a(tl))}`,
    corner(tl, a(tl), 0),
    "Z",
  ].join(" ");
}

/** A star of `points` points whose inner radius is `inner` of the outer, filling the box. */
function star(points: number, inner: number, w: number, h: number): string {
  const raw: [number, number][] = [];
  for (let index = 0; index < points * 2; index++) {
    const angle = -Math.PI / 2 + (index * Math.PI) / points;
    const radius = index % 2 === 0 ? 1 : inner;
    raw.push([Math.cos(angle) * radius, Math.sin(angle) * radius]);
  }
  return polygon(fitToBox(raw, w, h));
}

/** A regular polygon of `sides` sides, a vertex at the top, filling the box. */
function regular(sides: number, w: number, h: number): string {
  const raw: [number, number][] = [];
  for (let index = 0; index < sides; index++) {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / sides;
    raw.push([Math.cos(angle), Math.sin(angle)]);
  }
  return polygon(fitToBox(raw, w, h));
}

function fitToBox(points: [number, number][], w: number, h: number): [number, number][] {
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const spanX = Math.max(...xs) - minX || 1;
  const spanY = Math.max(...ys) - minY || 1;
  return points.map(([x, y]) => [((x - minX) / spanX) * w, ((y - minY) / spanY) * h]);
}

/** A callout's body with a pointer to (tipX, tipY), from the edge nearest it. */
function calloutRect(w: number, h: number, tipX: number, tipY: number, radius: number): string {
  const dx = (tipX - w / 2) / w;
  const dy = (tipY - h / 2) / h;
  const inside = tipX >= 0 && tipX <= w && tipY >= 0 && tipY <= h;
  if (inside) return cornerRect(w, h, [radius, radius, radius, radius]);
  const half = Math.min(w, h) / 10;
  const clampX = (x: number) => Math.max(radius + half, Math.min(w - radius - half, x));
  const clampY = (y: number) => Math.max(radius + half, Math.min(h - radius - half, y));
  const pts: string[] = [`M${n(radius)} 0`];
  const arc = (x: number, y: number) =>
    radius > 0 ? `A${n(radius)} ${n(radius)} 0 0 1 ${n(x)} ${n(y)}` : `L${n(x)} ${n(y)}`;
  const vertical = Math.abs(dy) >= Math.abs(dx);
  if (vertical && dy < 0) {
    const x = clampX(tipX);
    pts.push(`L${n(x - half)} 0 L${n(tipX)} ${n(tipY)} L${n(x + half)} 0`);
  }
  pts.push(`L${n(w - radius)} 0`, arc(w, radius));
  if (!vertical && dx > 0) {
    const y = clampY(tipY);
    pts.push(`L${n(w)} ${n(y - half)} L${n(tipX)} ${n(tipY)} L${n(w)} ${n(y + half)}`);
  }
  pts.push(`L${n(w)} ${n(h - radius)}`, arc(w - radius, h));
  if (vertical && dy > 0) {
    const x = clampX(tipX);
    pts.push(`L${n(x + half)} ${n(h)} L${n(tipX)} ${n(tipY)} L${n(x - half)} ${n(h)}`);
  }
  pts.push(`L${n(radius)} ${n(h)}`, arc(0, h - radius));
  if (!vertical && dx < 0) {
    const y = clampY(tipY);
    pts.push(`L0 ${n(y + half)} L${n(tipX)} ${n(tipY)} L0 ${n(y - half)}`);
  }
  pts.push(`L0 ${n(radius)}`, arc(radius, 0), "Z");
  return pts.join(" ");
}

const filled = (d: string): ShapePath[] => [{ d, fill: true, stroke: true }];
const stroked = (d: string): ShapePath[] => [{ d, fill: false, stroke: true }];

/** A preset geometry's paths at w × h, with its adjust values (in 1/100,000ths). */
export function presetPaths(
  name: string,
  w: number,
  h: number,
  adjust: Readonly<Record<string, number>>,
): ShapePath[] {
  const ss = Math.min(w, h);
  const adj = (key: string, fallback: number) => (adjust[key] ?? fallback) / 100000;
  const a = (fallback: number) => adj("adj", fallback);
  switch (name) {
    case "rect":
    case "flowChartProcess":
    case "actionButtonBlank":
      return filled(
        polygon([
          [0, 0],
          [w, 0],
          [w, h],
          [0, h],
        ]),
      );
    case "roundRect":
    case "flowChartAlternateProcess": {
      const r = ss * a(16667);
      return filled(cornerRect(w, h, [r, r, r, r]));
    }
    case "round1Rect": {
      const r = ss * a(16667);
      return filled(cornerRect(w, h, [0, r, 0, 0]));
    }
    case "round2SameRect": {
      const top = ss * adj("adj1", 16667);
      const bottom = ss * adj("adj2", 0);
      return filled(cornerRect(w, h, [top, top, bottom, bottom]));
    }
    case "round2DiagRect": {
      const one = ss * adj("adj1", 16667);
      const two = ss * adj("adj2", 0);
      return filled(cornerRect(w, h, [one, two, one, two]));
    }
    case "snip1Rect": {
      const r = ss * a(16667);
      return filled(cornerRect(w, h, [0, -r, 0, 0]));
    }
    case "snip2SameRect": {
      const top = ss * adj("adj1", 16667);
      const bottom = ss * adj("adj2", 0);
      return filled(cornerRect(w, h, [-top, -top, -bottom, -bottom]));
    }
    case "snip2DiagRect": {
      const one = ss * adj("adj1", 0);
      const two = ss * adj("adj2", 16667);
      return filled(cornerRect(w, h, [-one, -two, -one, -two]));
    }
    case "snipRoundRect": {
      const round = ss * adj("adj1", 16667);
      const snip = ss * adj("adj2", 16667);
      return filled(cornerRect(w, h, [round, -snip, 0, 0]));
    }
    case "flowChartTerminator":
      return filled(cornerRect(w, h, [ss / 2, ss / 2, ss / 2, ss / 2]));
    case "ellipse":
    case "flowChartConnector":
    case "cloud":
    case "flowChartOr":
    case "flowChartSummingJunction":
      return filled(ellipsePath(w / 2, h / 2, w / 2, h / 2));
    case "donut": {
      const dr = ss * a(25000);
      return [
        {
          d: `${ellipsePath(w / 2, h / 2, w / 2, h / 2)} ${ellipsePath(w / 2, h / 2, Math.max(0, w / 2 - dr), Math.max(0, h / 2 - dr))}`,
          fill: true,
          stroke: true,
        },
      ];
    }
    case "frame": {
      const inset = ss * adj("adj1", 12500);
      return [
        {
          d: `${polygon([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
          ])} ${polygon([
            [inset, inset],
            [inset, h - inset],
            [w - inset, h - inset],
            [w - inset, inset],
          ])}`,
          fill: true,
          stroke: true,
        },
      ];
    }
    case "triangle":
    case "flowChartExtract": {
      const x = w * (name === "triangle" ? a(50000) : 0.5);
      return filled(
        polygon([
          [x, 0],
          [w, h],
          [0, h],
        ]),
      );
    }
    case "flowChartMerge":
      return filled(
        polygon([
          [0, 0],
          [w, 0],
          [w / 2, h],
        ]),
      );
    case "rtTriangle":
      return filled(
        polygon([
          [0, 0],
          [w, h],
          [0, h],
        ]),
      );
    case "diamond":
    case "flowChartDecision":
      return filled(
        polygon([
          [w / 2, 0],
          [w, h / 2],
          [w / 2, h],
          [0, h / 2],
        ]),
      );
    case "parallelogram":
    case "flowChartInputOutput": {
      const x = name === "parallelogram" ? Math.min(w, ss * a(25000)) : w / 5;
      return filled(
        polygon([
          [x, 0],
          [w, 0],
          [w - x, h],
          [0, h],
        ]),
      );
    }
    case "trapezoid":
    case "flowChartManualOperation": {
      const x = name === "trapezoid" ? Math.min(w / 2, ss * a(25000)) : w / 5;
      return name === "trapezoid"
        ? filled(
            polygon([
              [0, h],
              [x, 0],
              [w - x, 0],
              [w, h],
            ]),
          )
        : filled(
            polygon([
              [0, 0],
              [w, 0],
              [w - x, h],
              [x, h],
            ]),
          );
    }
    case "pentagon":
      return filled(regular(5, w, h));
    case "heptagon":
      return filled(regular(7, w, h));
    case "decagon":
      return filled(regular(10, w, h));
    case "dodecagon":
      return filled(regular(12, w, h));
    case "hexagon":
    case "flowChartPreparation": {
      const x = Math.min(w / 2, name === "hexagon" ? ss * a(25000) : w / 5);
      return filled(
        polygon([
          [0, h / 2],
          [x, 0],
          [w - x, 0],
          [w, h / 2],
          [w - x, h],
          [x, h],
        ]),
      );
    }
    case "octagon": {
      const x = Math.min(ss / 2, ss * a(29289));
      return filled(
        polygon([
          [x, 0],
          [w - x, 0],
          [w, x],
          [w, h - x],
          [w - x, h],
          [x, h],
          [0, h - x],
          [0, x],
        ]),
      );
    }
    case "plus":
    case "mathPlus": {
      const x = name === "plus" ? Math.min(ss / 2, ss * a(25000)) : ss * 0.35;
      const y = name === "plus" ? x : x;
      return filled(
        polygon([
          [0, y],
          [x, y],
          [x, 0],
          [w - x, 0],
          [w - x, y],
          [w, y],
          [w, h - y],
          [w - x, h - y],
          [w - x, h],
          [x, h],
          [x, h - y],
          [0, h - y],
        ]),
      );
    }
    case "mathMinus":
      return filled(
        polygon([
          [0, h * 0.38],
          [w, h * 0.38],
          [w, h * 0.62],
          [0, h * 0.62],
        ]),
      );
    case "homePlate":
    case "flowChartOffpageConnector": {
      const x = w - Math.min(w, ss * a(50000));
      return name === "homePlate"
        ? filled(
            polygon([
              [0, 0],
              [x, 0],
              [w, h / 2],
              [x, h],
              [0, h],
            ]),
          )
        : filled(
            polygon([
              [0, 0],
              [w, 0],
              [w, h * 0.8],
              [w / 2, h],
              [0, h * 0.8],
            ]),
          );
    }
    case "chevron": {
      const x = Math.min(w, ss * a(50000));
      return filled(
        polygon([
          [0, 0],
          [w - x, 0],
          [w, h / 2],
          [w - x, h],
          [0, h],
          [x, h / 2],
        ]),
      );
    }
    case "rightArrow":
    case "notchedRightArrow":
    case "stripedRightArrow":
    case "bentArrow":
    case "curvedRightArrow": {
      const dy = (h * adj("adj1", 50000)) / 2;
      const x = w - Math.min(w, ss * adj("adj2", 50000));
      const notch = name === "notchedRightArrow" ? (dy * x) / (h / 2) / 4 : 0;
      return filled(
        polygon([
          [0, h / 2 - dy],
          [x, h / 2 - dy],
          [x, 0],
          [w, h / 2],
          [x, h],
          [x, h / 2 + dy],
          [0, h / 2 + dy],
          [notch, h / 2],
        ]),
      );
    }
    case "leftArrow": {
      const dy = (h * adj("adj1", 50000)) / 2;
      const x = Math.min(w, ss * adj("adj2", 50000));
      return filled(
        polygon([
          [w, h / 2 - dy],
          [x, h / 2 - dy],
          [x, 0],
          [0, h / 2],
          [x, h],
          [x, h / 2 + dy],
          [w, h / 2 + dy],
        ]),
      );
    }
    case "upArrow": {
      const dx = (w * adj("adj1", 50000)) / 2;
      const y = Math.min(h, ss * adj("adj2", 50000));
      return filled(
        polygon([
          [w / 2 - dx, h],
          [w / 2 - dx, y],
          [0, y],
          [w / 2, 0],
          [w, y],
          [w / 2 + dx, y],
          [w / 2 + dx, h],
        ]),
      );
    }
    case "downArrow": {
      const dx = (w * adj("adj1", 50000)) / 2;
      const y = h - Math.min(h, ss * adj("adj2", 50000));
      return filled(
        polygon([
          [w / 2 - dx, 0],
          [w / 2 + dx, 0],
          [w / 2 + dx, y],
          [w, y],
          [w / 2, h],
          [0, y],
          [w / 2 - dx, y],
        ]),
      );
    }
    case "leftRightArrow": {
      const dy = (h * adj("adj1", 50000)) / 2;
      const x = Math.min(w / 2, ss * adj("adj2", 50000));
      return filled(
        polygon([
          [0, h / 2],
          [x, 0],
          [x, h / 2 - dy],
          [w - x, h / 2 - dy],
          [w - x, 0],
          [w, h / 2],
          [w - x, h],
          [w - x, h / 2 + dy],
          [x, h / 2 + dy],
          [x, h],
        ]),
      );
    }
    case "upDownArrow": {
      const dx = (w * adj("adj1", 50000)) / 2;
      const y = Math.min(h / 2, ss * adj("adj2", 50000));
      return filled(
        polygon([
          [w / 2, 0],
          [w, y],
          [w / 2 + dx, y],
          [w / 2 + dx, h - y],
          [w, h - y],
          [w / 2, h],
          [0, h - y],
          [w / 2 - dx, h - y],
          [w / 2 - dx, y],
          [0, y],
        ]),
      );
    }
    case "star4":
      return filled(star(4, a(12500) * 2, w, h));
    case "star5":
      return filled(star(5, Math.min(0.9, a(19098) * 2), w, h));
    case "star6":
      return filled(star(6, Math.min(0.9, a(28868) * 2), w, h));
    case "star7":
      return filled(star(7, 0.6, w, h));
    case "star8":
      return filled(star(8, Math.min(0.9, a(38250) * 2), w, h));
    case "star10":
      return filled(star(10, 0.8, w, h));
    case "star12":
      return filled(star(12, Math.min(0.9, a(37500) * 2), w, h));
    case "star16":
    case "star24":
    case "star32":
      return filled(star(Number(name.slice(4)), 0.85, w, h));
    case "wedgeRectCallout":
    case "wedgeRoundRectCallout": {
      const tipX = w / 2 + w * adj("adj1", -20833);
      const tipY = h / 2 + h * adj("adj2", 62500);
      return filled(
        calloutRect(w, h, tipX, tipY, name === "wedgeRoundRectCallout" ? ss * 0.16667 : 0),
      );
    }
    case "wedgeEllipseCallout":
    case "cloudCallout": {
      const tipX = w / 2 + w * adj("adj1", -20833);
      const tipY = h / 2 + h * adj("adj2", 62500);
      const angle = Math.atan2(tipY - h / 2, tipX - w / 2);
      const spread = 0.25;
      const base = (offset: number): [number, number] => [
        w / 2 + (w / 2) * Math.cos(angle + offset),
        h / 2 + (h / 2) * Math.sin(angle + offset),
      ];
      return [
        { d: polygon([base(-spread), [tipX, tipY], base(spread)]), fill: true, stroke: true },
        { d: ellipsePath(w / 2, h / 2, w / 2, h / 2), fill: true, stroke: true },
      ];
    }
    case "can":
    case "flowChartMagneticDisk": {
      const ry = Math.min(h / 2, (ss * a(25000)) / 2);
      return [
        {
          d: `M0 ${n(ry)} A${n(w / 2)} ${n(ry)} 0 0 0 ${n(w)} ${n(ry)} L${n(w)} ${n(h - ry)} A${n(w / 2)} ${n(ry)} 0 0 1 0 ${n(h - ry)} Z`,
          fill: true,
          stroke: true,
        },
        { d: ellipsePath(w / 2, ry, w / 2, ry), fill: true, stroke: true },
      ];
    }
    case "flowChartDocument": {
      const wave = h * 0.1;
      return filled(
        `M0 0 L${n(w)} 0 L${n(w)} ${n(h - wave)} C${n(w * 0.75)} ${n(h - wave * 3)} ${n(w * 0.5)} ${n(h + wave)} ${n(w * 0.25)} ${n(h - wave)} C${n(w * 0.12)} ${n(h - wave * 2)} 0 ${n(h - wave)} 0 ${n(h - wave)} Z`,
      );
    }
    case "flowChartPredefinedProcess":
      return [
        {
          d: polygon([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
          ]),
          fill: true,
          stroke: true,
        },
        {
          d: `M${n(w / 8)} 0 L${n(w / 8)} ${n(h)} M${n((w * 7) / 8)} 0 L${n((w * 7) / 8)} ${n(h)}`,
          fill: false,
          stroke: true,
        },
      ];
    case "flowChartManualInput":
      return filled(
        polygon([
          [0, h / 5],
          [w, 0],
          [w, h],
          [0, h],
        ]),
      );
    case "foldedCorner": {
      const fold = ss * a(16667);
      return [
        {
          d: polygon([
            [0, 0],
            [w, 0],
            [w, h - fold],
            [w - fold, h],
            [0, h],
          ]),
          fill: true,
          stroke: true,
        },
        {
          d: polygon([
            [w - fold, h],
            [w - fold * 0.8, h - fold * 0.8],
            [w, h - fold],
          ]),
          fill: true,
          stroke: true,
        },
      ];
    }
    case "heart":
      return filled(
        `M${n(w / 2)} ${n(h / 4)} C${n(w / 2)} 0 0 0 0 ${n(h / 4)} C0 ${n(h / 2)} ${n(w / 4)} ${n((h * 3) / 4)} ${n(w / 2)} ${n(h)} C${n((w * 3) / 4)} ${n((h * 3) / 4)} ${n(w)} ${n(h / 2)} ${n(w)} ${n(h / 4)} C${n(w)} 0 ${n(w / 2)} 0 ${n(w / 2)} ${n(h / 4)} Z`,
      );
    case "pie":
    case "arc":
    case "chord":
    case "blockArc": {
      const start = ((adjust.adj1 ?? (name === "pie" ? 0 : 16200000)) / 60000) * (Math.PI / 180);
      const end = ((adjust.adj2 ?? (name === "pie" ? 16200000 : 0)) / 60000) * (Math.PI / 180);
      const point = (angle: number) => [
        w / 2 + (w / 2) * Math.cos(angle),
        h / 2 + (h / 2) * Math.sin(angle),
      ];
      let sweep = end - start;
      if (sweep <= 0) sweep += Math.PI * 2;
      const [x1, y1] = point(start) as [number, number];
      const [x2, y2] = point(start + sweep) as [number, number];
      const large = sweep > Math.PI ? 1 : 0;
      const arcPath = `M${n(x1)} ${n(y1)} A${n(w / 2)} ${n(h / 2)} 0 ${large} 1 ${n(x2)} ${n(y2)}`;
      if (name === "arc") return stroked(arcPath);
      if (name === "chord") return filled(`${arcPath} Z`);
      return filled(`M${n(w / 2)} ${n(h / 2)} L${arcPath.slice(1)} Z`);
    }
    case "line":
    case "straightConnector1":
      return stroked(`M0 0 L${n(w)} ${n(h)}`);
    case "bentConnector2":
      return stroked(`M0 0 L${n(w)} 0 L${n(w)} ${n(h)}`);
    case "bentConnector3":
    case "bentConnector4":
    case "bentConnector5": {
      const x = w * adj("adj1", 50000);
      return stroked(`M0 0 L${n(x)} 0 L${n(x)} ${n(h)} L${n(w)} ${n(h)}`);
    }
    case "curvedConnector2":
      return stroked(`M0 0 C${n(w)} 0 ${n(w)} 0 ${n(w)} ${n(h)}`);
    case "curvedConnector3":
    case "curvedConnector4":
    case "curvedConnector5": {
      const x = w * adj("adj1", 50000);
      return stroked(`M0 0 C${n(x)} 0 ${n(x)} ${n(h)} ${n(w)} ${n(h)}`);
    }
    case "leftBracket":
      return stroked(
        `M${n(w)} 0 Q0 0 0 ${n(Math.min(h / 2, w))} L0 ${n(h - Math.min(h / 2, w))} Q0 ${n(h)} ${n(w)} ${n(h)}`,
      );
    case "rightBracket":
      return stroked(
        `M0 0 Q${n(w)} 0 ${n(w)} ${n(Math.min(h / 2, w))} L${n(w)} ${n(h - Math.min(h / 2, w))} Q${n(w)} ${n(h)} 0 ${n(h)}`,
      );
    case "leftBrace":
      return stroked(
        `M${n(w)} 0 Q${n(w / 2)} 0 ${n(w / 2)} ${n(h / 4)} L${n(w / 2)} ${n(h / 2 - w / 2)} Q${n(w / 2)} ${n(h / 2)} 0 ${n(h / 2)} Q${n(w / 2)} ${n(h / 2)} ${n(w / 2)} ${n(h / 2 + w / 2)} L${n(w / 2)} ${n((h * 3) / 4)} Q${n(w / 2)} ${n(h)} ${n(w)} ${n(h)}`,
      );
    case "rightBrace":
      return stroked(
        `M0 0 Q${n(w / 2)} 0 ${n(w / 2)} ${n(h / 4)} L${n(w / 2)} ${n(h / 2 - w / 2)} Q${n(w / 2)} ${n(h / 2)} ${n(w)} ${n(h / 2)} Q${n(w / 2)} ${n(h / 2)} ${n(w / 2)} ${n(h / 2 + w / 2)} L${n(w / 2)} ${n((h * 3) / 4)} Q${n(w / 2)} ${n(h)} 0 ${n(h)}`,
      );
    case "bracketPair": {
      const r = ss * a(16667);
      return stroked(
        `M${n(r)} 0 Q0 0 0 ${n(r)} L0 ${n(h - r)} Q0 ${n(h)} ${n(r)} ${n(h)} M${n(w - r)} 0 Q${n(w)} 0 ${n(w)} ${n(r)} L${n(w)} ${n(h - r)} Q${n(w)} ${n(h)} ${n(w - r)} ${n(h)}`,
      );
    }
  }
  return filled(
    polygon([
      [0, 0],
      [w, 0],
      [w, h],
      [0, h],
    ]),
  );
}

/** A custom geometry's paths, scaled from each path's own space to w × h. */
export function customPaths(paths: readonly CustomPath[], w: number, h: number): ShapePath[] {
  return paths.map((path) => {
    const sx = path.w > 0 ? w / path.w : 1;
    const sy = path.h > 0 ? h / path.h : 1;
    let x = 0;
    let y = 0;
    let startX = 0;
    let startY = 0;
    const parts: string[] = [];
    for (const command of path.commands) {
      switch (command.op) {
        case "M":
        case "L":
          x = command.x;
          y = command.y;
          if (command.op === "M") {
            startX = x;
            startY = y;
          }
          parts.push(`${command.op}${n(x * sx)} ${n(y * sy)}`);
          break;
        case "C": {
          const [x1, y1, x2, y2, x3, y3] = command.points;
          parts.push(
            `C${n(x1 * sx)} ${n(y1 * sy)} ${n(x2 * sx)} ${n(y2 * sy)} ${n(x3 * sx)} ${n(y3 * sy)}`,
          );
          x = x3;
          y = y3;
          break;
        }
        case "Q": {
          const [x1, y1, x2, y2] = command.points;
          parts.push(`Q${n(x1 * sx)} ${n(y1 * sy)} ${n(x2 * sx)} ${n(y2 * sy)}`);
          x = x2;
          y = y2;
          break;
        }
        case "A": {
          // The arc's ellipse passes through the current point at the start angle.
          const start = (command.start * Math.PI) / 180;
          const swing = (command.swing * Math.PI) / 180;
          const cx = x - command.wR * Math.cos(start);
          const cy = y - command.hR * Math.sin(start);
          const endX = cx + command.wR * Math.cos(start + swing);
          const endY = cy + command.hR * Math.sin(start + swing);
          if (Math.abs(command.swing) >= 359.99) {
            const midX = cx + command.wR * Math.cos(start + swing / 2);
            const midY = cy + command.hR * Math.sin(start + swing / 2);
            const sweepFlag = swing > 0 ? 1 : 0;
            parts.push(
              `A${n(command.wR * sx)} ${n(command.hR * sy)} 0 0 ${sweepFlag} ${n(midX * sx)} ${n(midY * sy)}`,
              `A${n(command.wR * sx)} ${n(command.hR * sy)} 0 0 ${sweepFlag} ${n(endX * sx)} ${n(endY * sy)}`,
            );
          } else {
            parts.push(
              `A${n(command.wR * sx)} ${n(command.hR * sy)} 0 ${Math.abs(command.swing) > 180 ? 1 : 0} ${swing > 0 ? 1 : 0} ${n(endX * sx)} ${n(endY * sy)}`,
            );
          }
          x = endX;
          y = endY;
          break;
        }
        case "Z":
          parts.push("Z");
          x = startX;
          y = startY;
          break;
      }
    }
    return { d: parts.join(" "), fill: path.fill, stroke: path.stroke };
  });
}

/** Where a shape's text goes inside its box: insets from each edge, CSS pixels. */
export interface TextInsets {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

const NONE: TextInsets = { left: 0, top: 0, right: 0, bottom: 0 };

/**
 * A preset's text rectangle, as insets from its box: text sits inside an
 * ellipse's inscribed rectangle, after a chevron's notch, in an arrow's
 * shaft (after presetShapeDefinitions.xml, simplified). Other shapes use
 * their box.
 */
export function textInsets(geometry: Geometry, w: number, h: number): TextInsets {
  if (geometry.kind !== "preset") return NONE;
  const { name, adjust } = geometry;
  const ss = Math.min(w, h);
  const adj = (key: string, fallback: number) => (adjust[key] ?? fallback) / 100000;
  const a = (fallback: number) => adj("adj", fallback);
  const ellipse = { left: w * 0.1464, top: h * 0.1464, right: w * 0.1464, bottom: h * 0.1464 };
  switch (name) {
    case "ellipse":
    case "flowChartConnector":
    case "cloud":
    case "cloudCallout":
    case "wedgeEllipseCallout":
    case "donut":
      return ellipse;
    case "roundRect":
    case "flowChartAlternateProcess":
    case "wedgeRoundRectCallout": {
      const inset = ss * (name === "roundRect" ? a(16667) : 0.16667) * 0.29289;
      return { left: inset, top: inset, right: inset, bottom: inset };
    }
    case "flowChartTerminator":
      return { left: w * 0.0471, top: h * 0.1464, right: w * 0.0471, bottom: h * 0.1464 };
    case "chevron": {
      const x = Math.min(w, ss * a(50000));
      return { left: x, top: 0, right: x, bottom: 0 };
    }
    case "homePlate": {
      const x = Math.min(w, ss * a(50000));
      return { left: 0, top: 0, right: x / 2, bottom: 0 };
    }
    case "triangle":
    case "flowChartExtract": {
      const apex = w * (name === "triangle" ? a(50000) : 0.5);
      return { left: apex / 2, top: h / 2, right: (w - apex) / 2, bottom: 0 };
    }
    case "rtTriangle":
      return { left: w / 12, top: (h * 7) / 12, right: (w * 5) / 12, bottom: h / 12 };
    case "diamond":
    case "flowChartDecision":
      return { left: w / 4, top: h / 4, right: w / 4, bottom: h / 4 };
    case "parallelogram":
    case "trapezoid":
    case "flowChartInputOutput": {
      const x = name === "flowChartInputOutput" ? w / 5 : Math.min(w / 2, ss * a(25000));
      return { left: x * 0.6, top: 0, right: x * 0.6, bottom: 0 };
    }
    case "hexagon":
    case "octagon": {
      const x = Math.min(w / 2, ss * a(name === "hexagon" ? 25000 : 29289));
      return {
        left: x / 2,
        top: name === "octagon" ? x / 2 : h * 0.1,
        right: x / 2,
        bottom: name === "octagon" ? x / 2 : h * 0.1,
      };
    }
    case "pentagon":
      return { left: w * 0.2, top: h * 0.3, right: w * 0.2, bottom: 0 };
    case "star4":
    case "star5":
    case "star6":
    case "star7":
    case "star8":
    case "star10":
    case "star12":
      return { left: w * 0.3, top: h * 0.35, right: w * 0.3, bottom: h * 0.25 };
    case "plus": {
      const x = Math.min(ss / 2, ss * a(25000));
      return { left: 0, top: x, right: 0, bottom: x };
    }
    case "rightArrow":
    case "notchedRightArrow":
    case "stripedRightArrow": {
      const dy = (h * adj("adj1", 50000)) / 2;
      const head = Math.min(w, ss * adj("adj2", 50000));
      return { left: 0, top: h / 2 - dy, right: head / 2, bottom: h / 2 - dy };
    }
    case "leftArrow": {
      const dy = (h * adj("adj1", 50000)) / 2;
      const head = Math.min(w, ss * adj("adj2", 50000));
      return { left: head / 2, top: h / 2 - dy, right: 0, bottom: h / 2 - dy };
    }
    case "leftRightArrow": {
      const dy = (h * adj("adj1", 50000)) / 2;
      const head = Math.min(w / 2, ss * adj("adj2", 50000));
      return { left: head / 2, top: h / 2 - dy, right: head / 2, bottom: h / 2 - dy };
    }
    case "upArrow":
    case "downArrow": {
      const dx = (w * adj("adj1", 50000)) / 2;
      const head = Math.min(h, ss * adj("adj2", 50000));
      return name === "upArrow"
        ? { left: w / 2 - dx, top: head / 2, right: w / 2 - dx, bottom: 0 }
        : { left: w / 2 - dx, top: 0, right: w / 2 - dx, bottom: head / 2 };
    }
    case "can":
    case "flowChartMagneticDisk": {
      const ry = Math.min(h / 2, (ss * a(25000)) / 2);
      return { left: 0, top: ry * 2, right: 0, bottom: ry };
    }
    case "flowChartDocument":
      return { left: 0, top: 0, right: 0, bottom: h * 0.17 };
    case "flowChartPredefinedProcess":
      return { left: w / 8, top: 0, right: w / 8, bottom: 0 };
    case "foldedCorner":
      return { left: 0, top: 0, right: 0, bottom: ss * a(16667) };
  }
  return NONE;
}

/** A geometry's paths at w × h. */
export function geometryPaths(geometry: Geometry, w: number, h: number): ShapePath[] {
  return geometry.kind === "preset"
    ? presetPaths(geometry.name, w, h, geometry.adjust)
    : customPaths(geometry.paths, w, h);
}

/** Whether a geometry is its box: drawn as a plain rectangle, a picture needs no clip. */
export const isRectangle = (geometry: Geometry): boolean =>
  geometry.kind === "preset" && (geometry.name === "rect" || geometry.name === "flowChartProcess");

/** Whether a geometry is a line, which has no inside to fill. */
export const isLine = (geometry: Geometry): boolean =>
  geometry.kind === "preset" &&
  (geometry.name === "line" || geometry.name.includes("Connector") || geometry.name === "arc");
