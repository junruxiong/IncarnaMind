import { describe, expect, test } from "vitest";
import { asksToRemove, mapsOnto, waitsForUser } from "../../src/core/edits/applies";

describe("An edit the User asked for (#880)", () => {
  test("asks for text to go only when it names a part of the text, and doesn't say not to, in English or Chinese", () => {
    for (const asked of [
      "Delete the last paragraph",
      "remove the second point",
      "Cut this sentence",
      "take out everything after the first section",
      "Remove them",
      "删除最后一段",
      "把第二段删掉",
      "删掉这句",
      "删掉第二段的最后一句",
    ]) {
      expect(asksToRemove("instruction", asked), asked).toBe(true);
    }
    for (const asked of [
      "remove the jargon from this paragraph",
      "remove the jargon in the second paragraph",
      "Get rid of the repetition",
      "Make the second paragraph more formal",
      "Removal costs are in clause 4: make it clearer",
      "don't remove anything, just fix the typos",
      "Rewrite it without deleting a paragraph",
      "Keep every paragraph and remove the filler words",
      "去掉第二段里的口语化表达",
      "把重复的去掉",
      "不要删除任何段落",
      "保留所有段落，删掉多余的词",
    ]) {
      expect(asksToRemove("instruction", asked), asked).toBe(false);
    }
    expect(asksToRemove("shorten", "delete the last paragraph")).toBe(false);
    expect(asksToRemove("improve", "")).toBe(false);
  });

  test("new text maps onto the selection when at least half of it is found there", () => {
    expect(mapsOnto(50, 100)).toBe(true);
    expect(mapsOnto(49, 100)).toBe(false);
    expect(mapsOnto(0, 0)).toBe(true);
  });

  test("waits for the User only when its text doesn't map onto the selection, it wasn't asked to remove, and more than half of it would go", () => {
    const edit = { action: "instruction" as const, removing: false, mapped: false, selected: 100 };
    expect(waitsForUser({ ...edit, removed: 50, added: 0 })).toBe(false);
    expect(waitsForUser({ ...edit, removed: 51, added: 0 })).toBe(true);
    expect(waitsForUser({ ...edit, removed: 90, added: 40 })).toBe(false);
    expect(waitsForUser({ ...edit, removed: 100, added: 20 })).toBe(true);
    expect(waitsForUser({ ...edit, removing: true, removed: 100, added: 0 })).toBe(false);
    expect(waitsForUser({ ...edit, mapped: true, removed: 100, added: 0 })).toBe(false);
    expect(waitsForUser({ ...edit, action: "translate", removed: 100, added: 0 })).toBe(false);
    expect(waitsForUser({ ...edit, selected: 0, removed: 0, added: 0 })).toBe(false);
  });
});
