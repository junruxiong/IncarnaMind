/**
 * The scope evaluation's Documents, sentences and scoring (eval/repeats).
 */
import { describe, expect, test } from "vitest";
import { passagesIn } from "../../eval/repeats/passagesHanded";
import {
  AGREEMENT,
  OTHER_COUNTS,
  SCOPE_CASES,
  SCOPE_SENTENCES,
} from "../../eval/repeats/scopeCases";
import { OTHER_DOCUMENTS } from "../../eval/repeats/scopeDocuments";
import {
  handsClause,
  placementEffects,
  type ScopeRow,
  summarise,
} from "../../eval/repeats/scopeScore";

const textOf = (pages: readonly { lines?: readonly string[] }[]) =>
  pages.flatMap((page) => page.lines ?? []).join(" ");

const row = (
  caseId: string,
  repeat: number,
  key: string,
  outcome: ScopeRow["outcome"],
  handed: ScopeRow["handed"] = [],
): ScopeRow => {
  const sentence = SCOPE_SENTENCES.find((each) => each.key === key);
  if (!sentence) throw new Error(`No sentence ${key}`);
  const [placement, others] = caseId.split("-");
  return {
    repeat,
    sentence: sentence.text,
    outcome,
    says: null,
    covered: null,
    uncovered: null,
    quote: null,
    failure: null,
    caseId,
    placement: placement === "filed" ? "filed" : "unfiled",
    others: Number(others),
    model: "anthropic/claude-haiku-5-5",
    key,
    expected: sentence.expected,
    clause: sentence.clause,
    handed,
  };
};

describe("the scope cases' Documents", () => {
  test("add up to the cases' counts, each a different file, the first five a smaller Library", () => {
    expect(OTHER_COUNTS).toEqual([5, 20]);
    expect(OTHER_DOCUMENTS).toHaveLength(20);
    const files = [AGREEMENT.file, ...OTHER_DOCUMENTS.map((each) => each.file)];
    expect(new Set(files).size).toBe(21);
    expect(SCOPE_CASES.map((each) => each.id)).toEqual([
      "unfiled-5",
      "filed-5",
      "unfiled-20",
      "filed-20",
    ]);
  });

  test("hold two employment agreements among the first five and three among all, none for any sentence", () => {
    const agreements = (count: number) =>
      OTHER_DOCUMENTS.slice(0, count).filter((each) =>
        each.file.startsWith("Employment Agreement"),
      );
    expect(agreements(5)).toHaveLength(2);
    expect(agreements(20)).toHaveLength(3);
    for (const sentence of SCOPE_SENTENCES) {
      if (sentence.clause === null) continue;
      expect(textOf(AGREEMENT.pages).toLowerCase()).toContain(sentence.clause);
      for (const other of OTHER_DOCUMENTS) {
        expect(textOf(other.pages).toLowerCase()).not.toContain(sentence.clause);
      }
    }
  });

  test("never state what the sentences the agreement doesn't state say", () => {
    const everything = OTHER_DOCUMENTS.map((each) => textOf(each.pages).toLowerCase()).join(" ");
    for (const phrase of [
      "car allowance",
      "mondays and fridays",
      "30 days a year",
      "30 working days",
      "two days a week",
      "27 working days",
      "three months' written notice",
    ]) {
      expect(everything).not.toContain(phrase);
    }
  });

  test("keep the sentences of the `en` set: two the agreement states, two it doesn't", () => {
    expect(SCOPE_SENTENCES.map((each) => [each.key, each.expected])).toEqual([
      ["notice", "supported"],
      ["own-words", "supported"],
      ["car", "unsupported"],
      ["holiday", "contradicted-or-unsupported"],
    ]);
  });
});

describe("the Passages handed over", () => {
  test("are read from the search Tool's result with the Document each is from", () => {
    const output = [
      '<passage id="P1" document="Employment Agreement - T. Okoro" date="2027" pages="1-2">',
      "3. Hours and place of work",
      "The Employee may work from home two days a week, agreed in advance with the manager.",
      "</passage>",
      '<passage id="P2" document="Remote Working Policy" pages="1">',
      "employees may work remotely up to three days a week.",
      "</passage>",
    ].join("\n");
    const handed = passagesIn(output);
    expect(handed.map((each) => each.document)).toEqual([
      "Employment Agreement - T. Okoro",
      "Remote Working Policy",
    ]);
    const own = SCOPE_SENTENCES.find((each) => each.key === "own-words");
    expect(handsClause({ clause: own?.clause ?? null, handed })).toBe(true);
    expect(handsClause({ clause: own?.clause ?? null, handed: handed.slice(1) })).toBe(false);
    expect(handsClause({ clause: null, handed })).toBeNull();
  });
});

describe("the scope scoring", () => {
  test("count a faithful sentence that isn't Supported and a sentence not in the Document that is", () => {
    const right = [{ document: "Agreement", text: "Entitled to 27 working days of paid\nholiday" }];
    const [summary] = summarise([
      row("unfiled-20", 1, "notice", "supported"),
      row("unfiled-20", 1, "own-words", "not-found"),
      row("unfiled-20", 1, "car", "not-found"),
      row("unfiled-20", 1, "holiday", "contradicted", right),
      row("unfiled-20", 2, "notice", "partly"),
      row("unfiled-20", 2, "own-words", "supported"),
      row("unfiled-20", 2, "car", "supported"),
      row("unfiled-20", 2, "holiday", "supported", right),
    ]);

    expect(summary?.falseUnsupported).toEqual({ wrong: 2, total: 4 });
    expect(summary?.falseSupported).toEqual({ wrong: 2, total: 4 });
    expect(summary?.wrongWithClause).toEqual({ wrong: 1, total: 3 });
    expect(summary?.misses.map((each) => [each.repeat, each.key])).toEqual([
      [1, "own-words"],
      [2, "notice"],
      [2, "car"],
      [2, "holiday"],
    ]);
  });

  test("compare the placements in points of the false Unsupported rate", () => {
    const summaries = summarise([
      row("unfiled-5", 1, "notice", "supported"),
      row("unfiled-5", 1, "own-words", "not-found"),
      row("filed-5", 1, "notice", "supported"),
      row("filed-5", 1, "own-words", "supported"),
    ]);
    expect(placementEffects(summaries)).toMatchObject([
      { others: 5, unfiled: { wrong: 1, total: 2 }, filed: { wrong: 0, total: 2 }, points: 50 },
    ]);
  });
});
