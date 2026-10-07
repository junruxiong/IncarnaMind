/**
 * A PDF's zoom levels, and zooming by wheel: a trackpad's pinch, or the
 * wheel with Ctrl (or ⌘ on macOS) held. Both go through the same steps as the
 * zoom buttons, within the same limits.
 */

/** The zoom levels the buttons, the wheel and a pinch go through: 1 shows a page at its printed size. */
export const ZOOM_STEPS: readonly number[] = [
  0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4,
];
export const MIN_ZOOM = ZOOM_STEPS[0] as number;
export const MAX_ZOOM = ZOOM_STEPS.at(-1) as number;

/**
 * The zoom `steps` steps from `current` (positive: in), within the limits. A
 * zoom between steps, as fitting to the width gives, goes to the next step
 * either way; one below the smallest step stays put going out.
 */
export function stepZoom(current: number, steps: number): number {
  let zoom = current;
  for (let step = 0; step < Math.abs(steps); step++) {
    zoom =
      steps > 0
        ? (ZOOM_STEPS.find((each) => each > zoom + 0.001) ?? MAX_ZOOM)
        : (ZOOM_STEPS.findLast((each) => each < zoom - 0.001) ?? Math.min(zoom, MIN_ZOOM));
  }
  return zoom;
}

/** What a zoom gesture reads of a wheel event. */
export interface ZoomWheel {
  deltaY: number;
  /** `WheelEvent.deltaMode`: 0 for pixels, 1 for lines, 2 for pages. */
  deltaMode: number;
  ctrlKey: boolean;
  metaKey: boolean;
  timeStamp: number;
}

const DELTA_PIXEL = 0;
/** A pause this long between wheel events, in milliseconds, starts a new gesture. */
const GESTURE_PAUSE = 200;
/** How far a pinch's deltas run for one step: about a tenth of the page's size. */
const PINCH_DELTA_PER_STEP = 10;
/** And a wheel's, in pixels: about a notch of a mouse wheel. */
const WHEEL_DELTA_PER_STEP = 100;

/**
 * Turns wheel events into zoom steps. Chromium sends a trackpad's pinch as
 * wheel events with Ctrl set though no key is down, so the gesture follows
 * the modifier keys (`keyChanged`) to tell a pinch, whose small deltas add up
 * to a step, from a mouse wheel with Ctrl held, which takes a step at its
 * first notch however small its delta.
 */
export function createWheelZoom({ mac }: { mac: boolean }) {
  /** Ctrl (or ⌘ on macOS) is held down: a wheel with it set is the User's, not a pinch. */
  let held = false;
  /** The part of a step the gesture has moved but not yet taken; its sign is its direction. */
  let unused = 0;
  let last = Number.NEGATIVE_INFINITY;

  return {
    /** Follows the modifier keys, from each key going down or up. */
    keyChanged(event: { ctrlKey: boolean; metaKey: boolean }): void {
      held = event.ctrlKey || (mac && event.metaKey);
    },
    /** The window lost the focus: no key is held any more. */
    blur(): void {
      held = false;
    },
    /** Whether a wheel event zooms rather than scrolls. */
    isZoom(event: Pick<ZoomWheel, "ctrlKey" | "metaKey">): boolean {
      return event.ctrlKey || (mac && event.metaKey);
    },
    /** How many steps a zooming wheel event takes: positive in, negative out, often 0. */
    steps(event: ZoomWheel): number {
      // Away from the User (scrolling up) zooms in.
      const direction = -Math.sign(event.deltaY);
      if (direction === 0) return 0;
      const fresh = event.timeStamp - last > GESTURE_PAUSE || Math.sign(unused) === -direction;
      last = event.timeStamp;
      if (fresh) unused = 0;
      // A wheel that scrolls by lines or pages: a step for each event.
      if (event.deltaMode !== DELTA_PIXEL) return direction;
      const pinch = event.ctrlKey && !held;
      if (!pinch && fresh) return direction;
      unused += -event.deltaY / (pinch ? PINCH_DELTA_PER_STEP : WHEEL_DELTA_PER_STEP);
      const whole = Math.trunc(unused) || 0;
      unused -= whole;
      return whole;
    },
  };
}

export type WheelZoom = ReturnType<typeof createWheelZoom>;
