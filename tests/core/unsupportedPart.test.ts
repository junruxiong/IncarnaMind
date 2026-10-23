import { describe, expect, test } from "vitest";
import { readVerdict } from "../../src/core/answers/checks/prompt";

describe("What a reply says its Citations cover and don't", () => {
  test("are the phrases on the lines after the reason, whatever the label's markup", () => {
    expect(
      readVerdict(
        "Verdict: unsupported\nThe Passages compare speed, not memory or robustness.\nCovered: the speed comparison\nNot covered: memory and robustness",
      ),
    ).toEqual({
      verdict: "unsupported",
      says: "The Passages compare speed, not memory or robustness.",
      covered: "the speed comparison",
      uncovered: "memory and robustness",
      scope: null,
    });
    expect(
      readVerdict(
        "**Verdict:** unsupported\n**Covered:** “速度比较”\n**Not covered:** 内存和鲁棒性。",
      ),
    ).toMatchObject({ covered: "速度比较", uncovered: "内存和鲁棒性" });
    expect(
      readVerdict("Verdict: unsupported\n已涵盖：速度比较\n未涵盖：内存和鲁棒性"),
    ).toMatchObject({
      covered: "速度比较",
      uncovered: "内存和鲁棒性",
    });
  });

  test("never take the place of the reason", () => {
    expect(readVerdict("Verdict: unsupported\nCovered: the speed\nNot covered: memory")).toEqual({
      verdict: "unsupported",
      says: null,
      covered: "the speed",
      uncovered: "memory",
      scope: null,
    });
    expect(
      readVerdict("Verdict: unsupported\nNothing here is covered.\nNot covered: memory"),
    ).toMatchObject({ says: "Nothing here is covered.", covered: null, uncovered: "memory" });
  });

  test("read as phrases: no quotes or closing stop, and capitals as written", () => {
    expect(
      readVerdict(
        "Verdict: unsupported\nCovered: “The speed comparison.”\nNot covered: GraphST's memory, TensorFlow and Python runs;",
      ),
    ).toMatchObject({
      covered: "The speed comparison",
      uncovered: "GraphST's memory, TensorFlow and Python runs",
    });
  });

  test.each(["none", "None.", "NOTHING", "N/A", "n/a", "无", "“无。”"])(
    "are none for a side the reply says %j of",
    (said) => {
      expect(
        readVerdict(`Verdict: unsupported\nCovered: ${said}\nNot covered: memory`),
      ).toMatchObject({ covered: null, uncovered: "memory" });
    },
  );

  test("are none when the reply gives none, or one too long to be a phrase", () => {
    expect(readVerdict("Verdict: unsupported")).toEqual({
      verdict: "unsupported",
      says: null,
      covered: null,
      uncovered: null,
      scope: null,
    });
    expect(readVerdict("Verdict: unsupported\nNo Passage mentions penguins.")).toEqual({
      verdict: "unsupported",
      says: "No Passage mentions penguins.",
      covered: null,
      uncovered: null,
      scope: null,
    });
    expect(readVerdict("Verdict: unsupported\nCovered:\nNot covered: ")).toMatchObject({
      covered: null,
      uncovered: null,
    });
    const long = "x".repeat(200);
    expect(
      readVerdict(`Verdict: unsupported\nCovered: ${long}\nNot covered: memory`),
    ).toMatchObject({
      covered: null,
      uncovered: "memory",
    });
  });

  test("are none for a run of stops and spaces far too long to be a phrase", () => {
    const run = `${".  ".repeat(100_000)}x`;
    expect(readVerdict(`Verdict: unsupported\nCovered: ${run}\nNot covered: memory`)).toMatchObject(
      {
        covered: null,
        uncovered: "memory",
      },
    );
  });

  test("keep the reason only while it is one sentence long", () => {
    expect(
      readVerdict(`Verdict: unsupported\n${"The Documents say little. ".repeat(40)}`),
    ).toMatchObject({
      says: null,
    });
    expect(readVerdict(`Verdict: unsupported\n${"x".repeat(250)}`)?.says).toHaveLength(250);
  });

  test("come with a JSON reply's verdict", () => {
    expect(
      readVerdict(
        '{"verdict": "unsupported", "says": "Not stated.", "covered": "speed", "uncovered": "Memory"}',
      ),
    ).toEqual({
      verdict: "unsupported",
      says: "Not stated.",
      covered: "speed",
      uncovered: "Memory",
      scope: null,
    });
  });

  test("a reply's scope is its own line, read apart from the explanation", () => {
    expect(
      readVerdict(
        "Verdict: supported\nScope: wider\nThe agreement says the Employee may work from home [^1].\nCovered: two days a week for the Employee\nNot covered: staff in general",
      ),
    ).toEqual({
      verdict: "supported",
      says: "The agreement says the Employee may work from home.",
      covered: "two days a week for the Employee",
      uncovered: "staff in general",
      scope: "wider",
    });
    expect(readVerdict("**Verdict:** supported\n**Scope:** same\nIt says so [^1].")).toMatchObject({
      says: "It says so.",
      scope: "same",
    });
    expect(readVerdict('{"verdict": "supported", "says": "Yes.", "scope": "wider"}')).toMatchObject(
      {
        scope: "wider",
      },
    );
  });

  test("a supported reply's explanation is still its line", () => {
    expect(readVerdict("Verdict: supported\nThe Document says so [^1].")).toEqual({
      verdict: "supported",
      says: "The Document says so.",
      covered: null,
      uncovered: null,
      scope: null,
    });
  });
});
