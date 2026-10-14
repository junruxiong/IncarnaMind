import { describe, expect, test } from "vitest";
import { judgesUnreliably, SMALL_JUDGE_PARAMETERS } from "../../src/shared/smallModel";

describe("Which models' verdicts carry the notice that they may be wrong (#680)", () => {
  // protects=a model on this computer under 20 billion parameters gets the notice, 20 billion and up doesn't; fails_when=the line moves or becomes inclusive; no_claim=how the size is read, tests/core/ollamaChat.test.ts.
  test("below 20 billion parameters it does, from 20 billion it doesn't", () => {
    expect(SMALL_JUDGE_PARAMETERS).toBe(20_000_000_000);
    expect(judgesUnreliably({ parameters: 4.5e9 })).toBe(true);
    expect(judgesUnreliably({ parameters: 19_999_999_999 })).toBe(true);
    expect(judgesUnreliably({ parameters: 20e9 })).toBe(false);
    expect(judgesUnreliably({ parameters: 30e9 })).toBe(false);
  });

  // protects=a model whose size isn't known, and a cloud model (the core gives none a size), never gets the notice; fails_when=an unknown size counts as small; no_claim=none beyond protects.
  test("an unknown size never does, and a cloud model, which has none, never does", () => {
    expect(judgesUnreliably({})).toBe(false);
    expect(judgesUnreliably({ parameters: null })).toBe(false);
    expect(judgesUnreliably(null)).toBe(false);
  });
});
