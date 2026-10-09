/**
 * The Citation check on every Location kind (#70): Citations to the
 * every-format set's Documents (eval/retrieval/formats.json), checked against
 * the Units processing stores for them, as the core checks an Answer's
 * Citations, and sorted as the evaluation sorts them (`outcomeOf`). One group
 * per kind of Location: PDF pages (two columns, tables, footnotes, figure
 * captions), slides (tables, charts, speaker notes), sections of Word files
 * (tables, footnotes) and of Markdown files (tables, lists, code), blocks of
 * rows of workbooks and CSV files, and blocks of lines of plain text.
 *
 * The Citations are in tests/fixtures/eval-format-citations.json, written as
 * models write them before any model was asked the set. Its "false-not-found"
 * ones are quotes on their cited Unit that today's check doesn't find: a
 * table row written with pipes on a slide, a chart's value written with a
 * thousands separator, and a Markdown table row without its pipes.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, test } from "vitest";
import { type CitationOutcome, outcomeOf } from "../../eval/lib/citations";
import { FORMATS_SET, loadEvaluationSet } from "../../eval/lib/evaluationSet";
import { checkCitation } from "../../src/core/answers/citations";
import { kindOf } from "../../src/core/documents/files";
import type { PageText } from "../../src/core/documents/passages";
import { processFile } from "../../src/core/documents/processing";
import type { UnitKind } from "../../src/shared/units";

interface Written {
  kind: UnitKind;
  /** A key of formats.json's documents. */
  document: string;
  pageFrom: number;
  pageTo: number;
  quote: string;
  outcome: CitationOutcome;
  case: string;
}

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const CITATIONS = (
  JSON.parse(readFileSync(join(ROOT, "tests/fixtures/eval-format-citations.json"), "utf8")) as {
    citations: Written[];
  }
).citations;
const KINDS: readonly UnitKind[] = ["page", "slide", "section", "rows", "lines"];

/** Each cited Document's stored Units, as processing gives them. */
const units = new Map<string, PageText[]>();

beforeAll(async () => {
  const documents = loadEvaluationSet(ROOT, FORMATS_SET).documents;
  for (const key of new Set(CITATIONS.map((citation) => citation.document))) {
    const file = documents.find((document) => document.key === key)?.path;
    const kind = file && kindOf(file);
    if (!file || !kind) throw new Error(`No Document "${key}" in the every-format set.`);
    const result = await processFile({ task: "process", documentId: key, kind, file });
    if (result.outcome !== "ready") throw new Error(`"${key}" couldn't be processed.`);
    units.set(key, result.pages);
  }
}, 120_000);

/** What the check says of a Citation now, and how the evaluation sorts it. */
function recheck(citation: Written) {
  const all = units.get(citation.document) ?? [];
  const range = { pageFrom: citation.pageFrom, pageTo: citation.pageTo };
  const cited = all.filter(
    ({ page }) => page !== null && page >= citation.pageFrom && page <= citation.pageTo,
  );
  const { check, checkReason } = checkCitation({
    quote: citation.quote,
    range,
    passage: range,
    documentDeleted: false,
    pages: cited,
  });
  return { cited, outcome: outcomeOf({ ...citation, check, checkReason }, all) };
}

describe.each(KINDS)("A Citation to %s Units", (kind) => {
  const citations = CITATIONS.filter((citation) => citation.kind === kind);

  test("has written cases: found, and on the wrong Unit", () => {
    const outcomes = new Set(citations.map((citation) => citation.outcome));
    expect(outcomes.has("found")).toBe(true);
    expect(outcomes.has("wrong-page")).toBe(true);
  });

  test.each(citations.map((citation) => [`${citation.document}: ${citation.case}`, citation]))(
    "%s",
    (_label, citation) => {
      const { cited, outcome } = recheck(citation as Written);
      expect(cited.map((unit) => unit.kind ?? "page")).toEqual(cited.map(() => kind));
      expect(outcome).toBe((citation as Written).outcome);
    },
  );
});
