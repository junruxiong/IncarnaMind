import { describe, expect, test } from "vitest";
import { nameKeyAction } from "../../src/renderer/src/nameKeys";

describe("the keys of a name typed in place", () => {
  test("Enter saves and Esc cancels", () => {
    expect(nameKeyAction({ key: "Enter", keyCode: 13, isComposing: false })).toBe("save");
    expect(nameKeyAction({ key: "Escape", keyCode: 27, isComposing: false })).toBe("cancel");
  });

  test("while an input method is composing, Enter and Esc are the method's", () => {
    expect(nameKeyAction({ key: "Enter", keyCode: 13, isComposing: true })).toBeNull();
    expect(nameKeyAction({ key: "Escape", keyCode: 27, isComposing: true })).toBeNull();
    // Some systems report the composing Enter after the composition ended, as keyCode 229.
    expect(nameKeyAction({ key: "Enter", keyCode: 229, isComposing: false })).toBeNull();
  });

  test("other keys are typed", () => {
    expect(nameKeyAction({ key: "a", keyCode: 65, isComposing: false })).toBeNull();
  });
});
