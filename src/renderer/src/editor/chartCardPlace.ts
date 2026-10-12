import { flip, type Middleware, offset, shift } from "@floating-ui/dom";

const CARD_WIDTH = 300;
/** Narrower than this, a card beside a pie can't show its source: it goes over the pie instead. */
const NARROWEST = 140;
const GAP = 8;
const EDGE = 4;
const WIDTH_VARIABLE = "--chart-card-width";

export interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

function around(elements: Iterable<Element>): Box | null {
  let box: Box | null = null;
  for (const element of elements) {
    const { left, top, right, bottom } = element.getBoundingClientRect();
    box = box
      ? {
          left: Math.min(box.left, left),
          top: Math.min(box.top, top),
          right: Math.max(box.right, right),
          bottom: Math.max(box.bottom, bottom),
        }
      : { left, top, right, bottom };
  }
  return box;
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(value, high));

/**
 * Where a chart's value card goes (#436, #578): across, within the Mind's
 * column (`column`); up and down, between the chart's head and foot (`body`),
 * so it never covers "Bar ⌄", Show as table or the foot. Above the value when
 * there is room, else beside it with its top just under the head.
 *
 * A pie's card goes beside the pie, off its key and the slice in focus: on the
 * side away from the key, or past the key, narrowing to the room there. Only a
 * column too narrow for a card there puts it over the pie, on the half away
 * from the slice in focus.
 */
export function chartCardPlace(body: Element, column: Element | null, pie: boolean): Middleware[] {
  const frame = body.getBoundingClientRect();
  const across = column?.getBoundingClientRect() ?? { left: 0, right: window.innerWidth };
  const within: Box = {
    left: across.left,
    top: frame.top,
    right: across.right,
    bottom: frame.bottom,
  };
  if (pie) return [besidePie(body, within)];
  const boundary = {
    x: within.left,
    y: within.top,
    width: within.right - within.left,
    height: within.bottom - within.top,
  };
  return [
    offset(({ placement, rects }) =>
      placement.startsWith("top") || placement.startsWith("bottom")
        ? 4
        : { mainAxis: 8, alignmentAxis: frame.top + EDGE - rects.reference.y },
    ),
    flip({ boundary, fallbackPlacements: ["right-start", "left-start"] }),
    shift({ boundary, crossAxis: true, padding: EDGE }),
  ];
}

function besidePie(body: Element, within: Box): Middleware {
  return {
    name: "besidePie",
    fn({ rects, elements }) {
      const slices = around(body.querySelectorAll(".chart-marks .chart-point"));
      const key = around(body.querySelectorAll(".chart-key"));
      if (!slices || !key) return {};
      const { x, y, width, height } = rects.reference;
      const focus = { left: x, top: y, right: x + width, bottom: y + height };
      const place = pieCardPlace(rects.floating.height, focus, { slices, key }, within);
      // Its height follows its width (its source can wrap): measured again at the new width.
      if (elements.floating.style.getPropertyValue(WIDTH_VARIABLE) !== `${place.width}px`) {
        elements.floating.style.setProperty(WIDTH_VARIABLE, `${place.width}px`);
        return { reset: { rects: true } };
      }
      return { x: place.x, y: place.y };
    },
  };
}

/** Where a pie's card of this height goes for the slice in `focus` (see `chartCardPlace`), and its width. */
export function pieCardPlace(
  height: number,
  focus: Box,
  pie: { slices: Box; key: Box },
  within: Box,
): { x: number; y: number; width: number } {
  const { slices, key } = pie;
  const left = within.left + EDGE;
  const right = within.right - EDGE;
  const top = within.top + EDGE;
  const bottom = within.bottom - EDGE;
  const middle = (focus.top + focus.bottom) / 2;
  const sideY = clamp(middle - height / 2, top, bottom - height);
  const away = Math.floor(slices.left - GAP - left);
  const past = Math.floor(right - (key.right + GAP));
  const beside = (side: "away" | "past", width: number) => ({
    x: side === "away" ? slices.left - GAP - width : key.right + GAP,
    y: sideY,
    width,
  });
  if (away >= CARD_WIDTH) return beside("away", CARD_WIDTH);
  if (past >= CARD_WIDTH) return beside("past", CARD_WIDTH);
  if (Math.max(away, past) >= NARROWEST)
    return past >= away ? beside("past", past) : beside("away", away);
  const width = Math.floor(Math.min(CARD_WIDTH, key.left - GAP - left));
  const focusLow = middle > (slices.top + slices.bottom) / 2;
  return {
    x: clamp((slices.left + slices.right) / 2 - width / 2, left, key.left - GAP - width),
    y: focusLow ? top : bottom - height,
    width,
  };
}
