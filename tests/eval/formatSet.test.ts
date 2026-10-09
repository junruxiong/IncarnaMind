/**
 * The every-format evaluation set (eval/retrieval/formats.json, #70),
 * checked against what processing stores, without the models: each quote is
 * on its expected Units and on no other Unit of its Document, those Units can
 * be cited together, and a Passage that covers them holds the quote, so a hit
 * and a "found" Citation are both possible. A known gap's quote is in none of
 * its Document's Units. Every format has at least 3 Questions per hard place
 * in each language.
 */
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, test } from "vitest";
import { FORMATS_SET, loadEvaluationSet } from "../../eval/lib/evaluationSet";
import { FORMATS, formatOf } from "../../eval/lib/formats";
import { kindOf } from "../../src/core/documents/files";
import type { BuiltPassage, PageText } from "../../src/core/documents/passages";
import { processFile } from "../../src/core/documents/processing";
import { quoteInUnits } from "../../src/shared/locations";
import { findQuote } from "../../src/shared/quoteMatch";
import { sameRun } from "../../src/shared/units";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const SET = loadEvaluationSet(ROOT, FORMATS_SET);

interface Stored {
  outcome: string;
  units: PageText[];
  passages: BuiltPassage[];
}

const stored = new Map<string, Stored>();

beforeAll(async () => {
  for (const document of SET.documents) {
    const kind = kindOf(document.path);
    if (!kind) throw new Error(`${document.path} isn't a Document format.`);
    const result = await processFile({
      task: "process",
      documentId: document.key,
      kind,
      file: document.path,
    });
    stored.set(document.key, {
      outcome: result.outcome,
      units: result.outcome === "ready" ? result.pages : [],
      passages: result.outcome === "ready" ? result.passages : [],
    });
  }
}, 120_000);

const inRange = (unit: PageText, [from, to]: [number, number]) =>
  unit.page !== null && unit.page >= from && unit.page <= to;

describe("The every-format evaluation set", () => {
  test.each(
    SET.questions
      .filter((question) => !question.knownGap)
      .map((question) => [question.id, question] as const),
  )(
    "%s: the quote is on its expected Units only, and in a Passage that covers them",
    (_id, question) => {
      const { document, pages, quote } = question.expected;
      const { outcome, units, passages } = stored.get(document) as Stored;
      expect(outcome).toBe("ready");
      const expected = units.filter((unit) => inRange(unit, pages));
      expect(expected.map((unit) => unit.page)).toEqual(
        Array.from({ length: pages[1] - pages[0] + 1 }, (_, index) => pages[0] + index),
      );
      // A Citation names one Unit, or two consecutive ones of one run.
      expect(expected.length).toBeLessThanOrEqual(2);
      if (expected.length === 2)
        expect(sameRun(expected[0] as PageText, expected[1] as PageText)).toBe(true);
      expect(quoteInUnits(expected, quote), "on its Units").not.toBeNull();
      for (const unit of units.filter((each) => !inRange(each, pages))) {
        expect(quoteInUnits([unit], quote), `also on Unit ${unit.page}`).toBeNull();
      }
      const covering = passages.filter(
        (passage) =>
          passage.pageFrom !== null &&
          passage.pageTo !== null &&
          passage.pageFrom <= pages[0] &&
          passage.pageTo >= pages[1],
      );
      expect(covering.some((passage) => findQuote(passage.text, quote) !== null)).toBe(true);
    },
  );

  test("a known gap's quote is in none of its Document's Units: a scan has no text, a comment isn't read", () => {
    const gaps = SET.questions.filter((question) => question.knownGap);
    expect(gaps.length).toBeGreaterThan(0);
    for (const question of gaps) {
      const { outcome, units } = stored.get(question.expected.document) as Stored;
      if (question.place === "scanned") expect(outcome, question.id).toBe("no-text");
      else expect(quoteInUnits(units, question.expected.quote), question.id).toBeNull();
    }
  });

  test("every format has at least 3 Questions per hard place in each language", () => {
    const counts = new Map<string, number>();
    for (const question of SET.questions.filter((each) => !each.crossLingual)) {
      const format = formatOf(
        SET.documents.find((document) => document.key === question.expected.document)?.path ?? "",
      );
      const key = `${format} ${question.place} ${question.language}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    for (const format of FORMATS) {
      const places = new Set(
        [...counts.keys()]
          .filter((key) => key.startsWith(`${format.id} `))
          .map((key) => key.split(" ")[1]),
      );
      expect(places.size, format.label).toBeGreaterThanOrEqual(3);
      for (const place of places) {
        for (const language of ["en", "zh"]) {
          expect(
            counts.get(`${format.id} ${place} ${language}`) ?? 0,
            `${format.label}, ${place}, ${language}`,
          ).toBeGreaterThanOrEqual(3);
        }
      }
    }
  });

  test("every cross-lingual Question has a translated query, and only those do", () => {
    for (const question of SET.questions) {
      expect(question.translatedQuery !== undefined, question.id).toBe(question.crossLingual);
    }
  });
});
