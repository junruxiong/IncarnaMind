import { describe, expect, test } from "vitest";
import { type BadgeInput, badgeOf } from "../../src/renderer/src/editor/badgeState";

/* The check badge (#909): what it shows of a Mind's health and its checks, or nothing. */

const QUIET: BadgeInput = {
  marks: {},
  toCheck: 0,
  running: false,
  asking: false,
  problem: false,
  checked: null,
};

describe("The check badge", () => {
  test.each([
    ["a contradicted sentence", "contradicted"],
    ["an unsupported sentence", "unsupported"],
    ["a Citation not found", "notFound"],
    ["a sentence that couldn't be checked", "notChecked"],
  ] as const)("shows %s, with its mark and count", (_, part) => {
    expect(badgeOf({ ...QUIET, marks: { [part]: 2 } })).toEqual({
      marks: [{ part, count: 2 }],
      toCheck: 0,
      running: false,
      checked: null,
    });
  });

  test("shows the claims no check has looked at", () => {
    expect(badgeOf({ ...QUIET, toCheck: 3 })?.toCheck).toBe(3);
  });

  test("puts its marks in the margin's order, contradicted first, and leaves out parts with none", () => {
    const badge = badgeOf({
      ...QUIET,
      marks: { notChecked: 1, unsupported: 4, contradicted: 1, notFound: 0, found: 6 },
      toCheck: 2,
    });
    expect(badge?.marks).toEqual([
      { part: "contradicted", count: 1 },
      { part: "unsupported", count: 4 },
      { part: "notChecked", count: 1 },
    ]);
    expect(badge?.toCheck).toBe(2);
  });

  test("shows nothing for a Mind with nothing to check: Citations found, needs re-check and no Citation alone keep the page bare", () => {
    expect(badgeOf(QUIET)).toBeNull();
    expect(badgeOf({ ...QUIET, marks: { found: 6, recheck: 2, noCitation: 5 } })).toBeNull();
  });

  test("shows while a check runs, while a paid run asks first, and when a check the User started couldn't run", () => {
    expect(badgeOf({ ...QUIET, running: true })?.running).toBe(true);
    expect(badgeOf({ ...QUIET, asking: true })).not.toBeNull();
    expect(badgeOf({ ...QUIET, problem: true })).not.toBeNull();
  });

  test("says how many a check the User started checked, until they type", () => {
    expect(badgeOf({ ...QUIET, checked: 24 })?.checked).toBe(24);
    expect(badgeOf({ ...QUIET, marks: { found: 24 }, checked: null })).toBeNull();
  });
});
