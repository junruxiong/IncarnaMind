/**
 * Finding the lines of text on a page's picture (#397): the detection model
 * gives each pixel how likely it is to be text, and this turns that map into
 * boxes as PaddleOCR's DB post-processing does (threshold 0.3, box score 0.6,
 * unclip ratio 1.5), without OpenCV. Each blob of likely pixels is a line;
 * its box is the smallest rectangle around it, turned as the line is, grown
 * by the unclip distance so that the whole of its letters is inside.
 *
 * Pure: the recognizer (./onnx) runs the model, this reads its output.
 */

export type Point = readonly [number, number];

/** A box's corners: top-left, top-right, bottom-right, bottom-left, in pixels. */
export type Quad = readonly [Point, Point, Point, Point];

/** The detection model's output: how likely each pixel is to be text, row by row. */
export interface ProbabilityMap {
  width: number;
  height: number;
  data: Float32Array;
}

/** PaddleOCR's settings for PP-OCRv5's detection model. */
export const DETECTION = {
  /** A pixel above this is text. */
  threshold: 0.3,
  /** A box whose pixels average less is dropped. */
  boxThreshold: 0.6,
  unclipRatio: 1.5,
  /** Boxes narrower than this, in the map's pixels, are dropped. */
  minSize: 3,
  maxCandidates: 1000,
} as const;

/** A rectangle turned to any angle: its centre, its two unit axes, and half its size along each. */
interface Rotated {
  centre: Point;
  along: Point;
  across: Point;
  halfLength: number;
  halfWidth: number;
}

/** The convex hull of points, counter-clockwise (Andrew's monotone chain). */
export function convexHull(points: readonly Point[]): Point[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length < 3) return sorted;
  const cross = (o: Point, a: Point, b: Point) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Point[] = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower.at(-2) as Point, lower.at(-1) as Point, point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }
  const upper: Point[] = [];
  for (let index = sorted.length - 1; index >= 0; index--) {
    const point = sorted[index] as Point;
    while (upper.length >= 2 && cross(upper.at(-2) as Point, upper.at(-1) as Point, point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

/** The smallest-area rectangle around a convex hull: one of its sides lies along a hull edge. */
export function minAreaRect(hull: readonly Point[]): Rotated {
  if (hull.length === 1) {
    const [only] = hull as [Point];
    return { centre: only, along: [1, 0], across: [0, 1], halfLength: 0, halfWidth: 0 };
  }
  let best: Rotated | null = null;
  let bestArea = Number.POSITIVE_INFINITY;
  for (let index = 0; index < hull.length; index++) {
    const a = hull[index] as Point;
    const b = hull[(index + 1) % hull.length] as Point;
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (length === 0) continue;
    const along: Point = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
    const across: Point = [-along[1], along[0]];
    let minAlong = Number.POSITIVE_INFINITY;
    let maxAlong = Number.NEGATIVE_INFINITY;
    let minAcross = Number.POSITIVE_INFINITY;
    let maxAcross = Number.NEGATIVE_INFINITY;
    for (const point of hull) {
      const u = point[0] * along[0] + point[1] * along[1];
      const v = point[0] * across[0] + point[1] * across[1];
      minAlong = Math.min(minAlong, u);
      maxAlong = Math.max(maxAlong, u);
      minAcross = Math.min(minAcross, v);
      maxAcross = Math.max(maxAcross, v);
    }
    const area = (maxAlong - minAlong) * (maxAcross - minAcross);
    if (area < bestArea) {
      bestArea = area;
      const u = (minAlong + maxAlong) / 2;
      const v = (minAcross + maxAcross) / 2;
      best = {
        centre: [u * along[0] + v * across[0], u * along[1] + v * across[1]],
        along,
        across,
        halfLength: (maxAlong - minAlong) / 2,
        halfWidth: (maxAcross - minAcross) / 2,
      };
    }
  }
  return (
    best ?? {
      centre: hull[0] as Point,
      along: [1, 0],
      across: [0, 1],
      halfLength: 0,
      halfWidth: 0,
    }
  );
}

/**
 * A rectangle's corners in reading order, as PaddleOCR's `get_mini_boxes`
 * orders them: of the two leftmost, the higher is top-left; of the two
 * rightmost, the higher is top-right.
 */
function corners(rect: Rotated): Quad {
  const { centre, along, across, halfLength, halfWidth } = rect;
  const at = (s: number, t: number): Point => [
    centre[0] + along[0] * halfLength * s + across[0] * halfWidth * t,
    centre[1] + along[1] * halfLength * s + across[1] * halfWidth * t,
  ];
  const points = [at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)].sort((a, b) => a[0] - b[0]);
  const [p0, p1, p2, p3] = points as [Point, Point, Point, Point];
  const [topLeft, bottomLeft] = p0[1] <= p1[1] ? [p0, p1] : [p1, p0];
  const [topRight, bottomRight] = p2[1] <= p3[1] ? [p2, p3] : [p3, p2];
  return [topLeft, topRight, bottomRight, bottomLeft];
}

/** The mean of the map inside a convex quad (PaddleOCR's "fast" box score). */
function quadScore(map: ProbabilityMap, quad: Quad): number {
  const xs = quad.map((point) => point[0]);
  const ys = quad.map((point) => point[1]);
  const left = Math.max(0, Math.floor(Math.min(...xs)));
  const right = Math.min(map.width - 1, Math.ceil(Math.max(...xs)));
  const top = Math.max(0, Math.floor(Math.min(...ys)));
  const bottom = Math.min(map.height - 1, Math.ceil(Math.max(...ys)));
  let sum = 0;
  let count = 0;
  for (let y = top; y <= bottom; y++) {
    for (let x = left; x <= right; x++) {
      if (!insideQuad(quad, x, y)) continue;
      sum += map.data[y * map.width + x] as number;
      count++;
    }
  }
  return count === 0 ? 0 : sum / count;
}

/** Whether (x, y) is inside or on a convex quad whose corners go round in order. */
function insideQuad(quad: Quad, x: number, y: number): boolean {
  let sign = 0;
  for (let index = 0; index < 4; index++) {
    const a = quad[index] as Point;
    const b = quad[(index + 1) % 4] as Point;
    const cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    if (cross === 0) continue;
    const side = cross > 0 ? 1 : -1;
    if (sign === 0) sign = side;
    else if (side !== sign) return false;
  }
  return true;
}

/** A found line: its box in the page picture's pixels, and how sure the model is it is text. */
export interface TextBox {
  quad: Quad;
  score: number;
}

/**
 * The lines of text in a probability map, as boxes in the picture's pixels:
 * the map is `scaleX` × `scaleY` the picture's size, and boxes are clipped to
 * `width` × `height`. In no particular order.
 */
export function textBoxes(
  map: ProbabilityMap,
  scaleX: number,
  scaleY: number,
  width: number,
  height: number,
): TextBox[] {
  const { width: mapWidth, height: mapHeight, data } = map;
  const labels = new Int32Array(mapWidth * mapHeight);
  const boxes: TextBox[] = [];
  const stack: number[] = [];
  let label = 0;
  for (let start = 0; start < data.length; start++) {
    if (labels[start] !== 0 || (data[start] as number) <= DETECTION.threshold) continue;
    label++;
    if (label > DETECTION.maxCandidates) break;
    // Each row's leftmost and rightmost pixel of the blob: the hull needs no others.
    const rows = new Map<number, [number, number]>();
    labels[start] = label;
    stack.push(start);
    while (stack.length > 0) {
      const at = stack.pop() as number;
      const x = at % mapWidth;
      const y = (at - x) / mapWidth;
      const row = rows.get(y);
      if (!row) rows.set(y, [x, x]);
      else {
        if (x < row[0]) row[0] = x;
        if (x > row[1]) row[1] = x;
      }
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= mapHeight) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= mapWidth || (dx === 0 && dy === 0)) continue;
          const next = ny * mapWidth + nx;
          if (labels[next] === 0 && (data[next] as number) > DETECTION.threshold) {
            labels[next] = label;
            stack.push(next);
          }
        }
      }
    }
    const points: Point[] = [];
    for (const [y, [left, right]] of rows) points.push([left, y], [right, y]);
    const rect = minAreaRect(convexHull(points));
    if (Math.min(rect.halfLength, rect.halfWidth) * 2 < DETECTION.minSize) continue;
    const score = quadScore(map, corners(rect));
    if (score < DETECTION.boxThreshold) continue;
    // Grown on every side by the distance PaddleOCR's unclip offsets the box by.
    const length = rect.halfLength * 2;
    const breadth = rect.halfWidth * 2;
    const distance = (length * breadth * DETECTION.unclipRatio) / (2 * (length + breadth));
    const grown: Rotated = {
      ...rect,
      halfLength: rect.halfLength + distance,
      halfWidth: rect.halfWidth + distance,
    };
    if (Math.min(grown.halfLength, grown.halfWidth) * 2 < DETECTION.minSize + 2) continue;
    const clip = (value: number, limit: number) => Math.min(Math.max(Math.round(value), 0), limit);
    const toPicture = ([x, y]: Point): Point => [clip(x / scaleX, width), clip(y / scaleY, height)];
    const [topLeft, topRight, bottomRight, bottomLeft] = corners(grown);
    boxes.push({
      quad: [
        toPicture(topLeft),
        toPicture(topRight),
        toPicture(bottomRight),
        toPicture(bottomLeft),
      ],
      score,
    });
  }
  return boxes;
}
