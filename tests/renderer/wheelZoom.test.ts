import { describe, expect, test } from "vitest";
import {
  createWheelZoom,
  MAX_ZOOM,
  MIN_ZOOM,
  stepZoom,
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

describe("zoom steps", () => {
  test("go to the next step up or down, from a step or from a fitted zoom between steps", () => {
    expect(stepZoom(1, 1)).toBe(1.1);
    expect(stepZoom(1, -1)).toBe(0.9);
    expect(stepZoom(1, 3)).toBe(1.5);
    expect(stepZoom(0.73, 1)).toBe(0.75);
    expect(stepZoom(0.73, -1)).toBe(0.67);
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

  test("a pinch takes a step for each stretch of its movement, in or out", () => {
    const zoom = createWheelZoom({ mac: true });
    // Pinching out: small deltas, a step once they add up.
    const out = [-3, -3, -3, -3, -3, -3, -3].map((delta, index) =>
      zoom.steps(wheel(delta, index * 16)),
    );
    expect(out.reduce((sum, steps) => sum + steps, 0)).toBe(2);
    expect(out.slice(0, 3)).toEqual([0, 0, 0]);
    // Turning round mid-gesture starts the count again.
    expect(zoom.steps(wheel(4, 200))).toBe(0);
    expect(zoom.steps(wheel(4, 216))).toBe(0);
    expect(zoom.steps(wheel(4, 232))).toBe(-1);
  });

  test("a mouse wheel with Ctrl held takes a step at its first notch, then one per notch", () => {
    const zoom = createWheelZoom({ mac: false });
    zoom.keyChanged({ ctrlKey: true, metaKey: false });
    // However small the notch's delta, the first takes a step at once.
    expect(zoom.steps(wheel(-4, 0))).toBe(1);
    expect(zoom.steps(wheel(-100, 50))).toBe(1);
    expect(zoom.steps(wheel(-100, 100))).toBe(1);
    // After a pause, the next notch steps at once again.
    expect(zoom.steps(wheel(4, 1000))).toBe(-1);
    // Released, a wheel with Ctrl set is a pinch again.
    zoom.keyChanged({ ctrlKey: false, metaKey: false });
    expect(zoom.steps(wheel(4, 2000))).toBe(0);
  });

  test("a wheel that scrolls by lines takes a step per event", () => {
    const zoom = createWheelZoom({ mac: false });
    expect(zoom.steps(wheel(3, 0, { deltaMode: LINE }))).toBe(-1);
    expect(zoom.steps(wheel(3, 10, { deltaMode: LINE }))).toBe(-1);
    expect(zoom.steps(wheel(-3, 20, { deltaMode: LINE }))).toBe(1);
  });

  test("a wheel that moves only sideways doesn't zoom", () => {
    const zoom = createWheelZoom({ mac: true });
    expect(zoom.steps(wheel(0, 0))).toBe(0);
  });
});
