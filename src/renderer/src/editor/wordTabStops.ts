import type { WordParagraphLook, WordTabStop } from "../../../core/api";

/** Word's default tab stops when the file doesn't set its own (`w:defaultTabStop`): 0.5 in. */
const DEFAULT_TAB_STOP = 36;
/** A tab right on a stop goes to the next one, as in Word. */
const EPSILON = 0.5;

/**
 * The stop a tab at `x` (points from the column's left edge) goes to: the
 * paragraph's own, then its hanging indent, which Word treats as a stop, then
 * the default stops, `every` points apart.
 */
export function tabStopAfter(
  x: number,
  look: WordParagraphLook | undefined,
  every: number = DEFAULT_TAB_STOP,
): WordTabStop {
  const stops = (look?.tabs ?? []).filter((stop) => stop.align !== "bar");
  if (look?.firstLine !== undefined && look.firstLine < 0) {
    stops.push({ at: look.left ?? 0, align: "left", leader: "none" });
  }
  const next = stops.filter((stop) => stop.at > x + EPSILON).sort((a, b) => a.at - b.at)[0];
  if (next) return next;
  return {
    at: (Math.floor((x + EPSILON) / every) + 1) * every,
    align: "left",
    leader: "none",
  };
}

/**
 * How wide a tab is drawn, in pixels: from `x` to its stop, less what the
 * stop aligns after it (all of the text up to the next tab for a right stop,
 * half of it for a centred one, the text before its decimal point for a
 * decimal one).
 */
export function tabWidth(
  stop: WordTabStop,
  {
    x,
    stopX,
    following,
    beforeDecimal,
  }: { x: number; stopX: number; following: number; beforeDecimal: number },
): number {
  const aligned =
    stop.align === "right"
      ? following
      : stop.align === "center"
        ? following / 2
        : stop.align === "decimal"
          ? beforeDecimal
          : 0;
  return Math.max(0, stopX - x - aligned);
}

/** Whether a paragraph's tab stops line text up in columns, which wrapping would break. */
export const alignsColumns = (look: WordParagraphLook) =>
  (look.tabs ?? []).some((stop) => stop.align !== "left" && stop.align !== "bar");
