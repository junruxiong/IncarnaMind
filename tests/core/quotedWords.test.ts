/**
 * Words an Answer puts in quotation marks are checked against the Units its
 * Citations cite (#483). First how an Answer's Markdown is read: which spans
 * count, and which Citations are in their sentence; then, through the core
 * with scripted models, what a User sees.
 */
import { describe, expect, test } from "vitest";
import type { QuotedWords } from "../../src/core";
import { quotedWordsIn } from "../../src/core/answers/quotedWords";
import {
  askAndFinish,
  citationsIn,
  citingModel,
  onlyCitation,
  type ShownPassage,
  setUpWithDocuments,
} from "../helpers/citations";

describe("Reading the words an Answer puts in quotation marks", () => {
  test("a span of four words, with the Citations of its sentence", () => {
    expect(
      quotedWordsIn('The report says "training coverage reached 98%" [^1]. Other things [^2].'),
    ).toEqual([{ words: "training coverage reached 98%", markers: [1] }]);
  });

  test.each([
    ["curly", "The report says “training coverage reached 98%” [^1]."],
    ["single curly", "The report says ‘training coverage reached 98%’ [^1]."],
    ["corner", "报告称「training coverage reached 98%」[^1]。"],
    ["double corner", "报告称『training coverage reached 98%』[^1]。"],
    ["angle", "The report says «training coverage reached 98%» [^1]."],
  ])("in %s quotation marks", (_kind, text) => {
    expect(quotedWordsIn(text)).toEqual([{ words: "training coverage reached 98%", markers: [1] }]);
  });

  test("in Chinese, six characters or more count and fewer don't", () => {
    expect(quotedWordsIn("报告称“培训覆盖率达到98%”[^1]。")).toEqual([
      { words: "培训覆盖率达到98%", markers: [1] },
    ]);
    expect(quotedWordsIn("报告称“培训覆盖率”达到98%[^1]。")).toEqual([]);
    expect(quotedWordsIn("报告称“覆盖率”[^1]。")).toEqual([]);
    expect(quotedWordsIn("报告称“培训覆盖率已达”[^1]。")).toHaveLength(1);
  });

  test("a term or a name is not a quote", () => {
    expect(quotedWordsIn("The “Site Safety Report” covers “three sites” [^1].")).toEqual([]);
  });

  test("the marker may come before the closing punctuation, after it, or inside the quotation marks", () => {
    for (const text of [
      "It says “training coverage reached 98%”[^1].",
      "It says “training coverage reached 98%”. [^1] Next.",
      "It says “training coverage reached 98%[^1]”.",
    ]) {
      expect(quotedWordsIn(text)).toEqual([
        { words: "training coverage reached 98%", markers: [1] },
      ]);
    }
  });

  test("a comma or full stop inside the closing mark is typography, not the Document's words", () => {
    expect(quotedWordsIn("“Training coverage reached 98%,” it says [^1].")).toEqual([
      { words: "Training coverage reached 98%", markers: [1] },
    ]);
    expect(quotedWordsIn("报告称“培训覆盖率达到98%。”[^1]")).toEqual([
      { words: "培训覆盖率达到98%", markers: [1] },
    ]);
  });

  test("a sentence with no Citation is left alone", () => {
    expect(quotedWordsIn("It says “training coverage reached 98%”. Other things [^1].")).toEqual(
      [],
    );
  });

  test("a full stop inside the quotation marks does not end the sentence", () => {
    expect(
      quotedWordsIn("It says “Training was held. Coverage reached 98% of staff.” [^2]"),
    ).toEqual([{ words: "Training was held. Coverage reached 98% of staff", markers: [2] }]);
  });

  test("two quotes in one sentence, each with the sentence's Citations", () => {
    expect(
      quotedWordsIn(
        "It says “training coverage reached 98%” [^1] and “every site passed its audit” [^2].",
      ),
    ).toEqual([
      { words: "training coverage reached 98%", markers: [1, 2] },
      { words: "every site passed its audit", markers: [1, 2] },
    ]);
  });

  test("a quote may run across a line break, and an apostrophe is not a quotation mark", () => {
    expect(quotedWordsIn("It says “training coverage\nreached 98%” [^1].")).toEqual([
      { words: "training coverage reached 98%", markers: [1] },
    ]);
    expect(quotedWordsIn("The firm's board and the partners' vote were held [^1].")).toEqual([]);
  });

  test("none in code or a heading's Citation-less line; a table's cell is a sentence", () => {
    expect(
      quotedWordsIn(
        [
          "```",
          'say "training coverage reached 98%" [^1]',
          "```",
          "",
          "| Claim | Where |",
          "|:--|:--|",
          '| "training coverage reached 98%" [^1] | p. 3 |',
        ].join("\n"),
      ),
    ).toEqual([{ words: "training coverage reached 98%", markers: [1] }]);
  });

  test("a stray mark doesn't take in the sentences after it", () => {
    expect(quotedWordsIn('A 5" pipe is fine [^1]. Then "it ended here [^2].')).toEqual([]);
  });
});

/** The Document: one sentence, as a Markdown file, in English and in Chinese. */
const REPORTS = {
  en: {
    name: "Site safety report.md",
    contents:
      "# Site safety report\n\nSafety training: coverage reached 98% of site staff, up 6 points on 2024.\n",
    quote: "coverage reached 98% of site staff",
    query: "training coverage",
    question: "How many staff were trained?",
    found: "The report says “coverage reached 98% of site staff” [^1].",
    changed: "The report says “training coverage reached 98% of site staff” [^1].",
    term: "The report says “site staff” reached 98% [^1].",
    uncited:
      "Coverage is high [^1]. The report says “training coverage reached 98% of site staff”.",
    answer: [
      { text: "training ", found: false },
      { text: "coverage reached 98% of site staff", found: true },
    ],
    closest: [
      { text: "Safety training: ", found: false },
      { text: "coverage reached 98% of site staff", found: true },
      { text: ", up 6 points on 2024.", found: false },
    ],
  },
  zh: {
    name: "安全生产报告.md",
    contents: "# 安全生产报告\n\n安全培训，覆盖率达到98%，较上年提高 6 个百分点。\n",
    quote: "覆盖率达到98%",
    query: "覆盖率",
    question: "培训覆盖率是多少？",
    found: "报告称“覆盖率达到98%”[^1]。",
    changed: "报告称“培训覆盖率达到98%”[^1]。",
    term: "报告称“覆盖率”达到98%[^1]。",
    uncited: "覆盖率很高[^1]。报告称“培训覆盖率达到98%”。",
    answer: [
      { text: "培训", found: false },
      { text: "覆盖率达到98%", found: true },
    ],
    closest: [
      { text: "安全培训，", found: false },
      { text: "覆盖率达到98%", found: true },
      { text: "，较上年提高 6 个百分点。", found: false },
    ],
  },
} as const;

/** Asks one Question whose Answer is `answer`, citing the report's quote; gives the Answer's Citations. */
async function askWithQuote(language: keyof typeof REPORTS, answer: string) {
  const report = REPORTS[language];
  const model = citingModel({
    query: report.query,
    records: (passages: ShownPassage[]) => [
      { marker: 1, passage: passages[0]?.id, quote: report.quote },
    ],
    answer,
  });
  const { core, client, mind } = await setUpWithDocuments(model, [
    { name: report.name, contents: report.contents },
  ]);
  const { answerId, finished } = await askAndFinish(core, client, mind.id, report.question);
  return { client, answerId, finished };
}

describe.each(["en", "zh"] as const)("Quoted words in an Answer's sentence (%s)", (language) => {
  const report = REPORTS[language];

  test("words that are in the cited Unit change nothing", async () => {
    const { client, answerId, finished } = await askWithQuote(language, report.found);

    const citation = onlyCitation(client, answerId);
    expect(citation).toMatchObject({ check: "found", checkReason: null });
    expect(citation.quotedWords).toBeUndefined();
    expect(finished.citations).toHaveLength(1);
    expect(finished.citations[0]?.quotedWords).toBeUndefined();
  });

  test("words that differ by one word are not found: the Citation shows not found, with both texts", async () => {
    const { client, answerId, finished } = await askWithQuote(language, report.changed);

    const citation = onlyCitation(client, answerId);
    expect(citation).toMatchObject({
      check: "not-found",
      checkReason: "quoted-words-not-on-pages",
      // The Citation's own quote is still the record's, which the Document has.
      quote: report.quote,
    });
    const expected: QuotedWords = {
      answer: [...report.answer],
      closest: [...report.closest],
    };
    expect(citation.quotedWords).toEqual(expected);
    expect(finished.citations[0]).toMatchObject({
      check: "not-found",
      checkReason: "quoted-words-not-on-pages",
      quotedWords: expected,
    });
  });

  test("a term in quotation marks is not a quote", async () => {
    const { client, answerId } = await askWithQuote(language, report.term);

    expect(onlyCitation(client, answerId)).toMatchObject({ check: "found", checkReason: null });
  });

  test("words in a sentence with no Citation are left alone", async () => {
    const { client, answerId } = await askWithQuote(language, report.uncited);

    const citations = citationsIn(client, answerId);
    expect(citations).toHaveLength(1);
    expect(citations[0]).toMatchObject({ check: "found", checkReason: null });
  });
});
