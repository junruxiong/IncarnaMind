import { describe, expect, test } from "vitest";
import { holdWordLook, useWordLooks } from "../../src/renderer/src/wordLooks";

describe("the Look menu's Minds", () => {
  test("a Mind open in two panes stays in the menu until both have closed it", () => {
    const entry = { shown: "word" as const, originalId: null, choose: () => {} };
    const first = holdWordLook("mind-1", entry);
    const second = holdWordLook("mind-1", { ...entry, shown: "plain" });
    expect(useWordLooks.getState()["mind-1"]?.shown).toBe("plain");
    first();
    expect(useWordLooks.getState()["mind-1"]).toBeDefined();
    second();
    expect(useWordLooks.getState()["mind-1"]).toBeUndefined();
  });
});
