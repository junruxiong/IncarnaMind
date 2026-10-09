import { describe, expect, test } from "vitest";
import { looksLikeCode } from "../../src/renderer/src/viewer/plainText";

describe("a plain-text file is set in a monospaced font only when it looks like code", () => {
  test("prose, notes and lists are not code", () => {
    expect(
      looksLikeCode(
        [
          "Field notes, 3 March.",
          "",
          "Arrived at the harbour before dawn. The tide was turning; we waited.",
          "Gauges read at high water: 4.1 m, then 3.9 m (after the surge).",
          "",
          "- Bring the spare battery.",
          "- Call the harbour master: {number withheld}.",
        ].join("\n"),
      ),
    ).toBe(false);
    expect(looksLikeCode("")).toBe(false);
    // Two spaces after each sentence are not columns.
    expect(
      looksLikeCode(
        [
          "We left at six.  The boat was late.",
          "Gauges were read.  Two were broken.",
          "The surge came at noon.  It passed.",
        ].join("\n"),
      ),
    ).toBe(false);
  });

  test("source code, logs aligned in columns and ASCII tables are", () => {
    expect(
      looksLikeCode(
        [
          "def tide(height, hour):",
          "    if hour < 0:",
          "        raise ValueError(hour)",
          "    return height * 2",
          "",
          "for hour in range(24):",
          "    print(tide(1.5, hour))",
        ].join("\n"),
      ),
    ).toBe(true);
    expect(
      looksLikeCode(
        ["function read() {", "  const value = gauge.read();", "  return value;", "}"].join("\n"),
      ),
    ).toBe(true);
    expect(
      looksLikeCode(
        [
          "+---------+--------+",
          "| Site    | Gauges |",
          "+---------+--------+",
          "| Harbour |      5 |",
          "| Estuary |      4 |",
          "+---------+--------+",
        ].join("\n"),
      ),
    ).toBe(true);
    expect(
      looksLikeCode(
        [
          "TIME      LEVEL   FLOW",
          "06:00     4.10    12.5",
          "06:15     4.05    12.1",
          "06:30     3.98    11.8",
        ].join("\n"),
      ),
    ).toBe(true);
    expect(looksLikeCode("#!/bin/sh\necho hello\n")).toBe(true);
  });
});
