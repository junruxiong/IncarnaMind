import { describe, expect, test } from "vitest";
import { roomForBar } from "../../src/renderer/src/editor/barRoom";

const card = { left: 256, right: 764, bottom: 892 };
const window = { width: 1100, height: 900 };

describe("Where a change's bar may sit", () => {
  test("is the card on the side the assistant is docked beside it, so the bar never goes under it", () => {
    expect(roomForBar(card, "right", window)).toEqual({ x: 0, y: 0, width: 764, height: 900 });
    expect(roomForBar(card, "left", window)).toEqual({ x: 256, y: 0, width: 844, height: 900 });
    expect(roomForBar(card, "bottom", window)).toEqual({ x: 0, y: 0, width: 1100, height: 892 });
  });

  test("is the whole window with no assistant beside the card, as it always was", () => {
    expect(roomForBar(card, "float", window)).toBe("viewport");
    expect(roomForBar(card, undefined, window)).toBe("viewport");
    expect(roomForBar(null, "right", window)).toBe("viewport");
  });
});
