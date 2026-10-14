import { describe, expect, it } from "vitest";
import { en } from "../../src/shared/i18n/en";

describe("the English strings", () => {
  it("quote with curly marks (“like this”), never straight double quotes", () => {
    const straight = Object.entries(en)
      .filter(([, text]) => text.includes('"'))
      .map(([key]) => key);
    expect(straight).toEqual([]);
  });
});
