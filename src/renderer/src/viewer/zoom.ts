/**
 * A PDF's zoom levels, and zooming by wheel: a trackpad's pinch follows the
 * fingers smoothly, as in Preview or Chrome, and the wheel with Ctrl (or ⌘ on
 * macOS) held goes through the same steps as the zoom buttons. All stay within
 * the same limits.
 */

/** The zoom levels the buttons and the wheel go through: 1 shows a page at its printed size. */
export const ZOOM_STEPS: readonly number[] = [
  0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4,
];
export const MIN_ZOOM = ZOOM_STEPS[0] as number;
export const MAX_ZOOM = ZOOM_STEPS.at(-1) as number;

/**
 * The zoom `steps` steps from `current` (positive: in), within the limits. A
 * zoom between steps, as fitting to the width or a pinch gives, goes to the
 * next step either way; one below the smallest step stays put going out.
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

/**
 * Chromium sends a pinch that scales by `s` as a wheel event with
 * `deltaY = -100 · ln s`, so these deltas add up as the scales multiply.
 */
const PINCH_DELTA_PER_E = 100;

/**
 * The zoom a pinch's wheel event takes `current` to: scaled as much as the
 * fingers moved, within the limits. Fitted below the smallest zoom (a narrow
 * viewer), pinching closed stays put, and opening goes on from there.
 */
export function pinchZoom(current: number, deltaY: number): number {
  const zoom = current * Math.exp(-deltaY / PINCH_DELTA_PER_E);
  return Math.min(Math.max(zoom, Math.min(current, MIN_ZOOM)), MAX_ZOOM);
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
/** A pause this long between a wheel's events, in milliseconds, starts a new turn of it. */
const GESTURE_PAUSE = 200;
/** How far a wheel's deltas run for one step, in pixels: about a notch of a mouse wheel. */
const WHEEL_DELTA_PER_STEP = 100;

/**
 * Turns wheel events into zooms. Chromium sends a trackpad's pinch as wheel
 * events with Ctrl set though no key is down, so the gesture follows the
 * modifier keys (`keyChanged`) to tell a pinch, which the zoom follows
 * continuously, from a mouse wheel with Ctrl held, which takes a step at its
 * first notch however small its delta, then a step per notch.
 */
export function createWheelZoom({ mac }: { mac: boolean }) {
  /** Ctrl (or ⌘ on macOS) is held down: a wheel with it set is the User's, not a pinch. */
  let held = false;
  /** The part of a step the wheel has moved but not yet taken; its sign is its direction. */
  let unused = 0;
  let last = Number.NEGATIVE_INFINITY;

  /** How many steps a wheel (not a pinch) takes with this event: positive in, negative out, often 0. */
  const wheelSteps = (event: ZoomWheel, direction: number): number => {
    const fresh = event.timeStamp - last > GESTURE_PAUSE || Math.sign(unused) === -direction;
    last = event.timeStamp;
    if (fresh) unused = 0;
    // A wheel that scrolls by lines or pages: a step for each event.
    if (event.deltaMode !== DELTA_PIXEL || fresh) return direction;
    unused += -event.deltaY / WHEEL_DELTA_PER_STEP;
    const whole = Math.trunc(unused) || 0;
    unused -= whole;
    return whole;
  };

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
    /** The zoom a zooming wheel event takes `current` to; `current` itself if it doesn't move it. */
    zoom(event: ZoomWheel, current: number): number {
      // Away from the User (scrolling up, or pinching open) zooms in.
      const direction = -Math.sign(event.deltaY);
      if (direction === 0) return current;
      const pinch = event.deltaMode === DELTA_PIXEL && event.ctrlKey && !held;
      if (pinch) return pinchZoom(current, event.deltaY);
      return stepZoom(current, wheelSteps(event, direction));
    },
  };
}
