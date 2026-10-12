import { describe, expect, test } from "vitest";
import { type Box, pieCardPlace } from "../../src/renderer/src/editor/chartCardPlace";

const HEIGHT = 106;

const box = (left: number, top: number, right: number, bottom: number): Box => ({
  left,
  top,
  right,
  bottom,
});

function apart(card: { x: number; y: number; width: number }, other: Box): boolean {
  return (
    card.x + card.width <= other.left ||
    other.right <= card.x ||
    card.y + HEIGHT <= other.top ||
    other.bottom <= card.y
  );
}

function inside(card: { x: number; y: number; width: number }, within: Box): boolean {
  return (
    card.x >= within.left &&
    card.x + card.width <= within.right &&
    card.y >= within.top &&
    card.y + HEIGHT <= within.bottom
  );
}

// The Mind's column across and the chart's body up and down, the pie's slices, its key and the
// slice in focus, as the app lays them out at 1,600 and 1,100 px with the assistant open.
describe("a pie's value card (#578)", () => {
  test("in a wide Mind it goes past the key, full width, level with the slice in focus", () => {
    const within = box(256, 431, 1264, 667);
    const key = box(693, 486, 766, 600);
    const focus = box(450, 548, 569, 652);
    const place = pieCardPlace(HEIGHT, focus, { slices: box(449, 452, 655, 652), key }, within);
    expect(place).toEqual({ x: 774, y: 547, width: 300 });
    expect(inside(place, within) && apart(place, key) && apart(place, focus)).toBe(true);
  });

  test("with room on the side away from the key, it goes there", () => {
    const within = box(0, 431, 2000, 667);
    const slices = box(500, 452, 706, 652);
    const place = pieCardPlace(
      HEIGHT,
      box(501, 548, 620, 652),
      { slices, key: box(744, 486, 830, 600) },
      within,
    );
    expect(place).toEqual({ x: 192, y: 547, width: 300 });
    expect(apart(place, slices)).toBe(true);
  });

  test("past a wide Chinese key at 1,100 px, it narrows to the room there", () => {
    const within = box(256, 431, 764, 667);
    const key = box(489, 486, 589, 600);
    const focus = box(314, 548, 410, 632);
    const slices = box(313, 471, 473, 631);
    const place = pieCardPlace(HEIGHT, focus, { slices, key }, within);
    expect(place).toEqual({ x: 597, y: 537, width: 163 });
    expect(inside(place, within) && apart(place, key) && apart(place, slices)).toBe(true);
  });

  test("in a Mind too narrow for a card beside it, it goes over the pie, on the half away from the slice in focus", () => {
    const within = box(256, 431, 520, 667);
    const slices = box(280, 471, 440, 631);
    const key = box(456, 486, 520, 600);
    const low = box(282, 560, 360, 631);
    const high = box(360, 471, 440, 551);
    const forLow = pieCardPlace(HEIGHT, low, { slices, key }, within);
    const forHigh = pieCardPlace(HEIGHT, high, { slices, key }, within);
    expect(forLow).toEqual({ x: 260, y: 435, width: 188 });
    expect(forHigh).toEqual({ x: 260, y: 557, width: 188 });
    expect(apart(forLow, low) && apart(forLow, key) && inside(forLow, within)).toBe(true);
    expect(apart(forHigh, high) && apart(forHigh, key) && inside(forHigh, within)).toBe(true);
  });
});
