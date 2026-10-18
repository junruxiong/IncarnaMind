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
    });
    expect(
      readVerdict("Verdict: unsupported\nNothing here is covered.\nNot covered: memory"),
    ).toMatchObject({ says: "Nothing here is covered.", covered: null, uncovered: "memory" });
  });

  test("read as phrases: no quotes or closing stop, a capital that only starts the phrase made small", () => {
    expect(
      readVerdict(
        "Verdict: unsupported\nCovered: “The speed comparison.”\nNot covered: Memory, and GD runs;",
      ),
    ).toMatchObject({ covered: "the speed comparison", uncovered: "memory, and GD runs" });
    expect(readVerdict("Verdict: unsupported\nCovered: QN methods\nNot covered: GD")).toMatchObject(
      {
        covered: "QN methods",
        uncovered: "GD",
      },
    );
  });

  test("are none when the reply gives none, or one too long to be a phrase", () => {
    expect(readVerdict("Verdict: unsupported")).toEqual({
      verdict: "unsupported",
      says: null,
      covered: null,
      uncovered: null,
    });
    expect(readVerdict("Verdict: unsupported\nNo Passage mentions penguins.")).toEqual({
      verdict: "unsupported",
      says: "No Passage mentions penguins.",
      covered: null,
      uncovered: null,
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

  test("come with a JSON reply's verdict", () => {
    expect(
      readVerdict(
        '{"verdict": "unsupported", "says": "Not stated.", "covered": "speed", "uncovered": "Memory"}',
      ),
    ).toEqual({
      verdict: "unsupported",
      says: "Not stated.",
      covered: "speed",
      uncovered: "memory",
    });
  });

  test("a supported reply's explanation is still its line", () => {
    expect(readVerdict("Verdict: supported\nThe Document says so [^1].")).toEqual({
      verdict: "supported",
      says: "The Document says so.",
      covered: null,
      uncovered: null,
    });
  });
});
