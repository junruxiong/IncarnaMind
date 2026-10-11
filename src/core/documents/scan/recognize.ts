/**
 * Reading each line found on a page's picture (#397), as PaddleOCR feeds its
 * recognition model and reads its output: the line is cut out along its box,
 * turned level, scaled to 48 pixels high, and the model's guess at each step
 * along it is collapsed into text (CTC greedy decoding), with how sure it was
 * of each character and where along the line that character is.
 *
 * Pure: the recognizer (./onnx) runs the model, this prepares and reads.
 */
import type { Point, Quad } from "./detect";

/** A picture in shades of grey, one byte a pixel, row by row. */
export interface GreyImage {
  width: number;
  height: number;
  pixels: Uint8Array;
}

/** The recognition model reads lines this many pixels high. */
export const LINE_HEIGHT = 48;
/** A batch's lines are padded to at least this width, as PaddleOCR pads them (`rec_image_shape`). */
export const MIN_LINE_WIDTH = 320;
/** No line is read wider than this: a longer one is squeezed. */
export const MAX_LINE_WIDTH = 3200;
/**
 * A line's box is first read at most this many pixels high, each pixel the
 * mean of the picture's under it, then scaled up to `LINE_HEIGHT`. PP-OCRv5's
 * mobile model reads sharp, large lines worse: from a 200 dpi page it drops
 * simple Chinese characters (一, 人, 日, 月, 方). On the scanned-page set,
 * boxes read 12 to 16 pixels high read best; 15 had the fewest errors
 * (eval/ocr/README.md).
 */
export const READ_HEIGHT = 15;

const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** A line's box as it is read: turned so that its text runs left to right, and its size. */
export interface LineFrame {
  quad: Quad;
  length: number;
  breadth: number;
}

/**
 * How a line's box is read: as it is, or, for a box at least 1.5 times as
 * tall as it is wide (a column of vertical text), turned a quarter left, as
 * PaddleOCR turns it (`np.rot90`).
 */
export function lineFrame(quad: Quad): LineFrame {
  const [topLeft, topRight, bottomRight, bottomLeft] = quad;
  const length = Math.max(distance(topLeft, topRight), distance(bottomLeft, bottomRight));
  const breadth = Math.max(distance(topLeft, bottomLeft), distance(topRight, bottomRight));
  if (breadth / Math.max(length, 1) >= 1.5) {
    return { quad: [topRight, bottomRight, bottomLeft, topLeft], length: breadth, breadth: length };
  }
  return { quad, length, breadth };
}

/** How wide a line is read: its width at `LINE_HEIGHT` pixels high, at most `MAX_LINE_WIDTH`. */
export const readWidth = (frame: LineFrame) =>
  Math.max(
    1,
    Math.min(MAX_LINE_WIDTH, Math.ceil((LINE_HEIGHT * frame.length) / Math.max(frame.breadth, 1))),
  );

/** The grey value at a point between pixels, read from its four neighbours; the edge repeats. */
function sample(image: GreyImage, x: number, y: number): number {
  const { width, height, pixels } = image;
  const cx = Math.min(Math.max(x - 0.5, 0), width - 1);
  const cy = Math.min(Math.max(y - 0.5, 0), height - 1);
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = cx - x0;
  const fy = cy - y0;
  const top =
    (pixels[y0 * width + x0] as number) * (1 - fx) + (pixels[y0 * width + x1] as number) * fx;
  const bottom =
    (pixels[y1 * width + x0] as number) * (1 - fx) + (pixels[y1 * width + x1] as number) * fx;
  return top * (1 - fy) + bottom * fy;
}

/**
 * Writes a line, cut out of `image` along its frame, into a batch's tensor:
 * channel-first, three identical channels of `stride` × `LINE_HEIGHT` values
 * from `offset`, normalised to −1…1 as PaddleOCR normalises (x / 255 − 0.5) / 0.5.
 * It fills `width` columns; the rest of the row stays 0, PaddleOCR's padding.
 * The line is read at `READ_HEIGHT` first (see there).
 */
export function writeLine(
  image: GreyImage,
  frame: LineFrame,
  width: number,
  tensor: Float32Array,
  offset: number,
  stride: number,
): void {
  const [topLeft, topRight, , bottomLeft] = frame.quad;
  const acrossX = topRight[0] - topLeft[0];
  const acrossY = topRight[1] - topLeft[1];
  const downX = bottomLeft[0] - topLeft[0];
  const downY = bottomLeft[1] - topLeft[1];
  const low = Math.max(1, Math.round(Math.min(frame.breadth, READ_HEIGHT)));
  const lowWidth = Math.max(1, Math.round((width * low) / LINE_HEIGHT));
  const samplesAcross = Math.max(1, Math.min(8, Math.ceil(frame.length / lowWidth)));
  const samplesDown = Math.max(1, Math.min(8, Math.ceil(frame.breadth / low)));
  const small = new Float32Array(low * lowWidth);
  for (let v = 0; v < low; v++) {
    for (let u = 0; u < lowWidth; u++) {
      let sum = 0;
      for (let sv = 0; sv < samplesDown; sv++) {
        const t = (v + (sv + 0.5) / samplesDown) / low;
        for (let su = 0; su < samplesAcross; su++) {
          const s = (u + (su + 0.5) / samplesAcross) / lowWidth;
          sum += sample(
            image,
            topLeft[0] + s * acrossX + t * downX,
            topLeft[1] + s * acrossY + t * downY,
          );
        }
      }
      small[v * lowWidth + u] = sum / (samplesAcross * samplesDown);
    }
  }
  const plane = stride * LINE_HEIGHT;
  for (let v = 0; v < LINE_HEIGHT; v++) {
    const sy = Math.min(Math.max(((v + 0.5) * low) / LINE_HEIGHT - 0.5, 0), low - 1);
    const y0 = Math.floor(sy);
    const y1 = Math.min(y0 + 1, low - 1);
    const fy = sy - y0;
    for (let u = 0; u < width; u++) {
      const sx = Math.min(Math.max(((u + 0.5) * lowWidth) / width - 0.5, 0), lowWidth - 1);
      const x0 = Math.floor(sx);
      const x1 = Math.min(x0 + 1, lowWidth - 1);
      const fx = sx - x0;
      const grey =
        ((small[y0 * lowWidth + x0] as number) * (1 - fx) +
          (small[y0 * lowWidth + x1] as number) * fx) *
          (1 - fy) +
        ((small[y1 * lowWidth + x0] as number) * (1 - fx) +
          (small[y1 * lowWidth + x1] as number) * fx) *
          fy;
      const value = (grey / 255 - 0.5) / 0.5;
      const at = offset + v * stride + u;
      tensor[at] = value;
      tensor[at + plane] = value;
      tensor[at + 2 * plane] = value;
    }
  }
}

/**
 * One character read: its text (an entry of the model's character list,
 * which may be an emoji of two UTF-16 code units), how sure the model was,
 * and where its middle is along the line, from 0 (its start) to 1.
 */
export interface ReadCharacter {
  text: string;
  confidence: number;
  at: number;
}

export interface DecodedLine {
  /** Its characters' text, joined. */
  text: string;
  /** The mean of its characters' confidences, as PaddleOCR scores a line. */
  confidence: number;
  characters: ReadCharacter[];
}

/**
 * Collapses the model's output for one line into text (CTC greedy decoding):
 * at each of `steps` steps the likeliest of `classes` classes; class 0 is
 * blank, class i is `characters[i - 1]`, and the last class is a space.
 * Repeats and blanks are dropped. `read` is how much of the padded width the
 * line itself took, from 0 to 1, to place each character along it.
 */
export function decodeLine(
  output: Float32Array,
  offset: number,
  steps: number,
  classes: number,
  characters: readonly string[],
  read: number,
): DecodedLine {
  const found: ReadCharacter[] = [];
  let previous = 0;
  for (let step = 0; step < steps; step++) {
    const base = offset + step * classes;
    let best = 0;
    let bestValue = output[base] as number;
    for (let index = 1; index < classes; index++) {
      const value = output[base + index] as number;
      if (value > bestValue) {
        bestValue = value;
        best = index;
      }
    }
    if (best !== 0 && best !== previous) {
      found.push({
        text: best === classes - 1 ? " " : (characters[best - 1] ?? ""),
        confidence: bestValue,
        at: Math.min(1, (step + 0.5) / steps / Math.max(read, 1e-6)),
      });
    }
    previous = best;
  }
  const confidence =
    found.length === 0 ? 0 : found.reduce((sum, each) => sum + each.confidence, 0) / found.length;
  return { text: found.map((each) => each.text).join(""), confidence, characters: found };
}
