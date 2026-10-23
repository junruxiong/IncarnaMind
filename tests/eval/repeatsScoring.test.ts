/**
 * The repeated checks' scoring (eval/repeats): how a mark's card shows, and
 * what a run of checks of an unchanged Mind comes to.
 */
import { describe, expect, test } from "vitest";
import { SETS } from "../../eval/repeats/cases";
import { type Outcome, outcomeOf } from "../../eval/repeats/repeat";
import { type RepeatRow, report, summarise } from "../../eval/repeats/score";
import type { ClaimCheckAttributes } from "../../src/core";

const QUOTE = {
  passageId: "p1",
  documentId: "d1",
  documentName: "Agreement",
  contentHash: "h1",
  pageFrom: 1,
  pageTo: 1,
  location: null,
  quote: "The Employee may work from home two days a week",
  check: "found" as const,
  checkReason: null,
};

const mark = (change: Partial<ClaimCheckAttributes>): ClaimCheckAttributes => ({
  id: "c1",
  request: "r1",
  action: "check-against-documents",
  state: "unsupported",
  reason: null,
  detail: null,
  providerId: null,
  modelId: null,
  text: "Staff may work from home two days a week.",
  evidence: null,
  says: null,
  unsupported: { reason: null, covered: null, uncovered: null },
  calculation: null,
  citedCaseUnchecked: false,
  citedCaseFound: false,
  scope: { folderId: null, widened: false },
  versions: [],
  excluded: [],
  checkedAt: "2026-10-22T20:00:00.000Z",
  ...change,
});

const RUN = { commit: "abc", spent: 0, repeats: 1 };

const row = (repeat: number, key: string, outcome: Outcome, set = "en"): RepeatRow => {
  const sentence = SETS.find((each) => each.id === set)?.sentences.find((each) => each.key === key);
  if (!sentence) throw new Error(`No sentence ${key}`);
  return {
    repeat,
    sentence: sentence.text,
    outcome,
    says: null,
    covered: null,
    uncovered: null,
    quote: null,
    failure: null,
    set,
    model: "anthropic/claude-haiku-5-5",
    key,
    kind: sentence.kind,
    expected: sentence.expected,
    ...(sentence.knownMiss !== undefined && { knownMiss: sentence.knownMiss }),
  };
};

describe("the repeated checks", () => {
  test("name an Unsupported card by what it shows: a quoted part, a different line, or absent", () => {
    const partly = { reason: null, covered: "two days a week", uncovered: "all staff" };
    expect(outcomeOf(mark({ unsupported: { ...partly, quoted: QUOTE } }))).toBe("partly");
    expect(outcomeOf(mark({ unsupported: partly }))).toBe("absent");
    expect(outcomeOf(mark({ evidence: QUOTE, unsupported: { ...partly, quoted: QUOTE } }))).toBe(
      "differs",
    );
    expect(outcomeOf(mark({}))).toBe("absent");
    expect(outcomeOf(mark({ state: "supported" }))).toBe("supported");
    expect(outcomeOf(mark({ state: "not-checked", reason: "failed" }))).toBe("not-checked");
    expect(outcomeOf(undefined)).toBe("no-mark");
  });

  test("count false Supported, absent on a stated sentence, and checks that differ from the one before", () => {
    const [summary] = summarise([
      row(1, "notice", "supported"),
      row(1, "car", "absent"),
      row(2, "notice", "partly"),
      row(2, "car", "absent"),
      row(3, "notice", "absent"),
      row(3, "car", "supported"),
    ]);

    expect(summary).toMatchObject({
      repeats: 3,
      falseSupported: 1,
      absentOnStated: 1,
      partlyOnAbsent: 0,
    });
    expect(summary?.disagreeing).toBe(2);
    expect(summary?.followed).toBe(2);
    expect(summary?.sentences.find((each) => each.key === "notice")).toMatchObject({
      steady: 2,
      outcomes: { supported: 1, partly: 1, absent: 1 },
    });
    expect(summary?.misses.map((each) => [each.repeat, each.key])).toEqual([
      [2, "notice"],
      [3, "notice"],
      [3, "car"],
    ]);
  });

  test("expect the partly-supported card for a general word for the agreement's term", () => {
    const [summary] = summarise([
      row(1, "staff-home", "partly"),
      row(2, "staff-home", "supported"),
    ]);
    expect(summary?.misses.map((each) => [each.repeat, each.outcome])).toEqual([[2, "supported"]]);
    expect(summary?.falseSupported).toBe(0);
  });

  test("count a genuine conflict Supported as a false Supported, and a known miss apart", () => {
    const [summary] = summarise([
      row(1, "number", "contradicted", "en-conflicts"),
      row(1, "place", "supported", "en-conflicts"),
      row(1, "scope", "partly", "en-conflicts"),
      row(1, "duty", "supported", "en-conflicts"),
    ]);
    expect(summary).toMatchObject({
      falseSupported: 1,
      conflicts: { contradicted: 1, checked: 3 },
    });
    expect(summary?.misses.map((each) => each.key)).toEqual(["place", "scope"]);
    expect(summary?.known.map((each) => [each.key, each.knownMiss])).toEqual([["duty", 1668]]);
    const shown = report(
      summarise([
        row(1, "duty", "supported", "en-conflicts"),
        row(2, "duty", "contradicted", "en-conflicts"),
        row(1, "employer", "partly", "en-conflicts"),
      ]),
      RUN,
    );
    expect(shown).toContain("### Known misses (#1668), outside the counts above");
    expect(shown).toContain(
      "- duty: 1 of 2 Contradicted, 1 Supported, 0 unsupported, 0 not checked or not named",
    );
    expect(shown).toContain(
      "- employer: 0 of 1 Contradicted, 0 Supported, 1 unsupported, 0 not checked or not named",
    );
    expect(shown).not.toContain("Genuine conflicts Contradicted");
  });

  test("mark as known misses (#1668) the conflicts the first verdict or the claims miss today, and keep them in the sets", () => {
    const known = (id: string) =>
      SETS.find((each) => each.id === id)
        ?.sentences.filter((each) => each.knownMiss === 1668)
        .map((each) => each.key);
    expect(known("en-conflicts")).toEqual(["employer", "duty"]);
    expect(known("zh-conflicts")).toEqual(["negation", "payer", "landlord", "share"]);
  });

  test("expect every genuine conflict to be Contradicted, a dozen kinds in each language", () => {
    for (const id of ["en-conflicts", "zh-conflicts"]) {
      const sentences = SETS.find((each) => each.id === id)?.sentences ?? [];
      expect(sentences.length).toBeGreaterThanOrEqual(12);
      expect(new Set(sentences.map((each) => each.expected))).toEqual(new Set(["contradicted"]));
    }
  });

  test("give every set's sentences a verdict to expect, the same keys in English and Chinese", () => {
    const keys = (id: string) => SETS.find((each) => each.id === id)?.sentences.map((s) => s.key);
    expect(keys("en")).toEqual(keys("zh"));
    expect(keys("en-control")).toEqual(keys("zh-control"));
  });
});
