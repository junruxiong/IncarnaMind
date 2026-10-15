import { describe, expect, test } from "vitest";
import { asksToRemove, waitsForUser } from "../../src/core/edits/applies";

describe("An edit the User asked for (#880)", () => {
  test("asks for text to go only when its words say so, in English or Chinese", () => {
    expect(asksToRemove("instruction", "Delete the last paragraph")).toBe(true);
    expect(asksToRemove("instruction", "remove the second point")).toBe(true);
    expect(asksToRemove("instruction", "Get rid of the repetition")).toBe(true);
    expect(asksToRemove("instruction", "删除最后一段")).toBe(true);
    expect(asksToRemove("instruction", "把重复的去掉")).toBe(true);
    expect(asksToRemove("instruction", "Make the second paragraph more formal")).toBe(false);
    expect(asksToRemove("instruction", "Removal costs are in clause 4: make it clearer")).toBe(
      false,
    );
    expect(asksToRemove("shorten", "")).toBe(false);
    expect(asksToRemove("improve", "")).toBe(false);
  });

  test("waits for the User only when its text doesn't map onto the selection and more than half of it would go", () => {
    const edit = { action: "instruction" as const, mapped: false, selected: 100 };
    expect(waitsForUser({ ...edit, removed: 50, added: 0 })).toBe(false);
    expect(waitsForUser({ ...edit, removed: 51, added: 0 })).toBe(true);
    expect(waitsForUser({ ...edit, removed: 90, added: 40 })).toBe(false);
    expect(waitsForUser({ ...edit, removed: 100, added: 20 })).toBe(true);
    expect(waitsForUser({ ...edit, mapped: true, removed: 100, added: 0 })).toBe(false);
    expect(waitsForUser({ ...edit, action: "translate", removed: 100, added: 0 })).toBe(false);
    expect(waitsForUser({ ...edit, selected: 0, removed: 0, added: 0 })).toBe(false);
  });
});
