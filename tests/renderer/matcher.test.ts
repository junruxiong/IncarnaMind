import { describe, expect, test } from "vitest";
import { matchName, queryWords } from "../../src/renderer/src/matcher";

const marked = (query: string, name: string) =>
  matchName(queryWords(query), name)?.ranges.map((range) => name.slice(range.start, range.end));

const ranked = (query: string, names: string[]) =>
  names
    .map((name) => ({ name, match: matchName(queryWords(query), name) }))
    .filter((each) => each.match !== null)
    .sort((a, b) => (b.match?.score ?? 0) - (a.match?.score ?? 0))
    .map((each) => each.name);

describe("matching names", () => {
  test("every word typed must be in the name, in any order, whatever its case and accents", () => {
    expect(queryWords("  Notice   PERIOD ")).toEqual(["notice", "period"]);
    expect(marked("2026 contract", "Contracts register 2026")).toEqual(["Contract", "2026"]);
    expect(marked("alesund", "Tides at Ålesund")).toEqual(["Ålesund"]);
    expect(matchName(queryWords("contract giraffe"), "Contracts register 2026")).toBeNull();
    expect(matchName([], "Anything")).toBeNull();
  });

  test("Chinese is found inside a name, and traditional and simplified characters match", () => {
    expect(marked("合同", "供应商合同审查")).toEqual(["合同"]);
    expect(marked("財務", "2026 财务报表")).toEqual(["财务"]);
    expect(marked("财务", "財務報表")).toEqual(["財務"]);
    expect(marked("报表 2026", "2026 财务报表")).toEqual(["2026", "报表"]);
  });

  test("the name that is the words comes first, then one that starts with them, then one with them at a word's start", () => {
    expect(
      ranked("contract", ["Subcontractors", "Master contract", "Contracts register", "Contract"]),
    ).toEqual(["Contract", "Contracts register", "Master contract", "Subcontractors"]);
  });

  test("what is marked points at the name as written, past emoji and accents", () => {
    expect(marked("notes", "📄 Notes")).toEqual(["Notes"]);
    expect(marked("cafe", "Café notes")).toEqual(["Café"]);
  });
});
