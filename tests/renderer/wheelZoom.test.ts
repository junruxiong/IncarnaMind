import { describe, expect, test } from "vitest";
import {
  createWheelZoom,
  MAX_ZOOM,
  MIN_ZOOM,
  pinchZoom,
  stepZoom,
  ZOOM_STEPS,
  type ZoomWheel,
} from "../../src/renderer/src/viewer/zoom";

const PIXEL = 0;
const LINE = 1;

/** A wheel event, by default a trackpad's pinch (Chromium sends one as a wheel with Ctrl). */
const wheel = (deltaY: number, at: number, more: Partial<ZoomWheel> = {}): ZoomWheel => ({
  deltaY,
  deltaMode: PIXEL,
  ctrlKey: true,
  metaKey: false,
  timeStamp: at,
  ...more,
});

/** The deltaY Chromium sends for a pinch that scales by `scale`. */
const pinchDelta = (scale: number) => -100 * Math.log(scale);

/** Runs wheel events through a gesture from `from`, giving the zoom after each. */
function zoomsOf(gesture: ReturnType<typeof createWheelZoom>, from: number, events: ZoomWheel[]) {
  const zooms: number[] = [];
  let zoom = from;
  for (const event of events) {
    zoom = gesture.zoom(event, zoom);
    zooms.push(zoom);
  }
  return zooms;
}

describe("zoom steps", () => {
  test("go to the next step up or down, from a step or from a zoom between steps", () => {
    expect(stepZoom(1, 1)).toBe(1.1);
    expect(stepZoom(1, -1)).toBe(0.9);
    expect(stepZoom(1, 3)).toBe(1.5);
    expect(stepZoom(0.73, 1)).toBe(0.75);
    expect(stepZoom(0.73, -1)).toBe(0.67);
    // As a pinch leaves it, just off a step.
    expect(stepZoom(1.2337, 1)).toBe(1.25);
    expect(stepZoom(1.2337, -1)).toBe(1.1);
  });

  test("stay within the limits", () => {
    expect(stepZoom(MAX_ZOOM, 1)).toBe(MAX_ZOOM);
    expect(stepZoom(3, 5)).toBe(MAX_ZOOM);
    expect(stepZoom(MIN_ZOOM, -1)).toBe(MIN_ZOOM);
    expect(stepZoom(0.33, -4)).toBe(MIN_ZOOM);
    // Fitted below the smallest step (a narrow viewer): out stays put, in goes to the smallest step.
    expect(stepZoom(0.2, -1)).toBe(0.2);
    expect(stepZoom(0.2, 1)).toBe(MIN_ZOOM);
  });
});

describe("pinching", () => {
  test("scales the zoom as much as the fingers moved", () => {
    expect(pinchZoom(1, pinchDelta(2))).toBeCloseTo(2, 9);
    expect(pinchZoom(1.5, pinchDelta(0.5))).toBeCloseTo(0.75, 9);
    expect(pinchZoom(0.8, pinchDelta(1.05))).toBeCloseTo(0.84, 9);
  });

  test("stays within the limits; fitted below the smallest zoom, closing stays put", () => {
    expect(pinchZoom(3.5, pinchDelta(2))).toBe(MAX_ZOOM);
    expect(pinchZoom(0.3, pinchDelta(0.5))).toBe(MIN_ZOOM);
    expect(pinchZoom(0.2, pinchDelta(0.9))).toBe(0.2);
    expect(pinchZoom(0.2, pinchDelta(1.1))).toBeCloseTo(0.22, 9);
  });
});

describe("zooming by wheel", () => {
  test("only Ctrl, or ⌘ on macOS, makes a wheel zoom; without them it scrolls", () => {
    const mac = createWheelZoom({ mac: true });
    const other = createWheelZoom({ mac: false });
    expect(mac.isZoom(wheel(5, 0))).toBe(true);
    expect(mac.isZoom(wheel(5, 0, { ctrlKey: false, metaKey: true }))).toBe(true);
    expect(mac.isZoom(wheel(5, 0, { ctrlKey: false }))).toBe(false);
    expect(other.isZoom(wheel(5, 0, { ctrlKey: false, metaKey: true }))).toBe(false);
    expect(other.isZoom(wheel(5, 0))).toBe(true);
  });

  test("a pinch moves the zoom smoothly with every event, not in steps", () => {
    const gesture = createWheelZoom({ mac: true });
    // Pinching open: small deltas, each one zooming in a little.
    const opening = zoomsOf(
      gesture,
      1,
      Array.from({ length: 7 }, (_, index) => wheel(-3, index * 16)),
    );
    for (const [index, zoom] of opening.entries()) {
      expect(zoom).toBeGreaterThan(opening[index - 1] ?? 1);
    }
    expect(opening.at(-1)).toBeCloseTo(Math.exp(0.21), 9);
    expect(opening.filter((zoom) => ZOOM_STEPS.includes(zoom))).toEqual([]);
    // Closing as far again comes back to where it started.
    const closing = zoomsOf(
      gesture,
      opening.at(-1) ?? 1,
      Array.from({ length: 7 }, (_, index) => wheel(3, 200 + index * 16)),
    );
    expect(closing.at(-1)).toBeCloseTo(1, 9);
  });

  test("however far a pinch goes, the zoom stays within its limits", () => {
    const gesture = createWheelZoom({ mac: false });
    const opened = zoomsOf(
      gesture,
      1,
      Array.from({ length: 200 }, (_, index) => wheel(-5, index * 16)),
    );
    expect(opened.at(-1)).toBe(MAX_ZOOM);
    const closed = zoomsOf(
      gesture,
      MAX_ZOOM,
      Array.from({ length: 300 }, (_, index) => wheel(5, 4000 + index * 16)),
    );
    expect(closed.at(-1)).toBe(MIN_ZOOM);
  });

  test("a mouse wheel with Ctrl held takes a step at its first notch, then one per notch", () => {
    const gesture = createWheelZoom({ mac: false });
    gesture.keyChanged({ ctrlKey: true, metaKey: false });
    // However small the notch's delta, the first takes a step at once.
    expect(gesture.zoom(wheel(-4, 0), 1)).toBe(1.1);
    expect(gesture.zoom(wheel(-100, 50), 1.1)).toBe(1.25);
    expect(gesture.zoom(wheel(-100, 100), 1.25)).toBe(1.5);
    // Less than a notch more takes no step yet.
    expect(gesture.zoom(wheel(-40, 150), 1.5)).toBe(1.5);
    // After a pause, the next notch steps at once again.
    expect(gesture.zoom(wheel(4, 1000), 1.5)).toBe(1.25);
    // Released, a wheel with Ctrl set is a pinch again, and follows its delta.
    gesture.keyChanged({ ctrlKey: false, metaKey: false });
    expect(gesture.zoom(wheel(4, 2000), 1.25)).toBeCloseTo(1.25 * Math.exp(-0.04), 9);
  });

  test("⌘ with a wheel on macOS steps as Ctrl does", () => {
    const gesture = createWheelZoom({ mac: true });
    gesture.keyChanged({ ctrlKey: false, metaKey: true });
    expect(gesture.zoom(wheel(-2, 0, { ctrlKey: false, metaKey: true }), 1)).toBe(1.1);
  });

  test("a wheel that scrolls by lines takes a step per event", () => {
    const gesture = createWheelZoom({ mac: false });
    expect(gesture.zoom(wheel(3, 0, { deltaMode: LINE }), 1)).toBe(0.9);
    expect(gesture.zoom(wheel(3, 10, { deltaMode: LINE }), 0.9)).toBe(0.8);
    expect(gesture.zoom(wheel(-3, 20, { deltaMode: LINE }), 0.8)).toBe(0.9);
  });

  test("a wheel that moves only sideways doesn't zoom", () => {
    const gesture = createWheelZoom({ mac: true });
    expect(gesture.zoom(wheel(0, 0), 1)).toBe(1);
  });
});
