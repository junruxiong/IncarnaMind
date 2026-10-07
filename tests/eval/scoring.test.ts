/**
 * How the evaluation (`npm run eval`, eval/README.md) scores what the core
 * returns. The evaluation itself needs the real model and isn't part of
 * `npm test`; its scoring is, since a mistake there would skew every report.
 */
import { describe, expect, test } from "vitest";
import {
  type AnswerRecord,
  type CitationOutcome,
  type CitationRecord,
  type CitationRun,
  outcomeOf,
  sentencesOf,
  summariseGroup,
} from "../../eval/lib/citations";
import type { PassageSearchResult } from "../../src/core";

const record = (overrides: Partial<AnswerRecord>): AnswerRecord => ({
  questionId: "en-01",
  language: "en",
  crossLingual: false,
  round: 1,
  question: "When are spring tides?",
  status: "done",
  error: null,
  citationSupport: "tools",
  searches: [],
  droppedMarkers: 0,
  droppedRecords: 0,
  sentences: [],
  citations: [],
  seconds: 1,
  ...overrides,
});

import { reviewerSheet } from "../../eval/lib/report";
import { checkPassage, isHit } from "../../eval/lib/retrieval";

const MARK = "\uE000";

describe("Sentences of an Answer and their Citations", () => {
  test("a Citation counts for the sentence it is in, or the one whose full stop it follows", () => {
    const text = `Spring tides come at full moon ${MARK}. Neap tides are smaller.${MARK} Tides vary.`;
    expect(sentencesOf(text, ["a", "b"], "en")).toEqual([
      { text: "Spring tides come at full moon .", citations: ["a"] },
      { text: "Neap tides are smaller.", citations: ["b"] },
      { text: "Tides vary.", citations: [] },
    ]);
  });

  test("Chinese sentences split at their full stops", () => {
    const text = `月球引力形成潮汐隆起${MARK}。大潮发生在满月时。${MARK}${MARK}小潮最小。`;
    expect(sentencesOf(text, [1, 2, 3], "zh")).toEqual([
      { text: "月球引力形成潮汐隆起。", citations: [1] },
      { text: "大潮发生在满月时。", citations: [2, 3] },
      { text: "小潮最小。", citations: [] },
    ]);
  });
});

describe("What became of a Citation", () => {
  const pages = [
    { page: 1, text: "Tides rise and fall." },
    { page: 2, text: "Revenue grew by ten per-\ncent in the third quarter." },
    { page: 3, text: "Costs fell." },
  ];
  const citation = (quote: string, pageFrom = 2, pageTo = 2) => ({
    check: "not-found" as const,
    checkReason: "quote-not-on-pages" as const,
    quote,
    pageFrom,
    pageTo,
  });

  test("a quote on the cited page under the looser normalisation is a false 'not found'", () => {
    expect(outcomeOf(citation("revenue grew by ten percent, in the third quarter"), pages)).toBe(
      "false-not-found",
    );
  });

  test("a quote on another page is a wrong page; one nowhere in the Document isn't in it", () => {
    expect(outcomeOf(citation("Tides rise and fall"), pages)).toBe("wrong-page");
    expect(outcomeOf(citation("Revenue rose by 10%"), pages)).toBe("not-in-document");
  });

  test("the check's own results are kept", () => {
    expect(
      outcomeOf({ ...citation("Costs fell."), check: "found", checkReason: null }, pages),
    ).toBe("found");
    expect(
      outcomeOf({ ...citation("Costs fell.", 1, 3), checkReason: "too-many-pages" }, pages),
    ).toBe("page-range");
  });
});

describe("The retrieval hit rule", () => {
  const expected = {
    document: "tides",
    pages: [4, 4] as [number, number],
    quote: "international trade slowed",
  };
  const passage = (overrides: Partial<PassageSearchResult>): PassageSearchResult => ({
    passageId: "p",
    documentId: "doc-1",
    documentName: "Tides",
    pageFrom: 3,
    pageTo: 4,
    position: 0,
    text: "Growth in inter-\nnational trade slowed.",
    ...overrides,
  });

  test("needs the expected Document, pages covering the expected ones, and the quote", () => {
    expect(isHit(checkPassage(passage({}), expected, "doc-1"))).toBe(true);
    expect(checkPassage(passage({}), expected, "doc-2")).toMatchObject({ rightDocument: false });
    expect(checkPassage(passage({ pageTo: 3 }), expected, "doc-1")).toMatchObject({
      coversPages: false,
    });
    expect(checkPassage(passage({ text: "Trade slowed." }), expected, "doc-1")).toMatchObject({
      hasQuote: false,
    });
  });
});

const cited = (outcome: CitationOutcome, sentence = "A claim."): CitationRecord => ({
  sentence,
  quote: "a quote",
  documentName: "Tides",
  pageFrom: 1,
  pageTo: 1,
  check: outcome === "found" ? "found" : "not-found",
  checkReason: outcome === "found" ? null : "quote-not-on-pages",
  outcome,
});

describe("A language's Citation figures", () => {
  test("shares are of all its Citations, coverage of all its sentences", () => {
    const summary = summariseGroup([
      record({
        citations: [cited("found"), cited("found"), cited("false-not-found")],
        sentences: [
          { text: "A claim.", cited: true },
          { text: "Another.", cited: false },
        ],
        droppedMarkers: 1,
      }),
      record({
        status: "timed-out",
        citationSupport: null,
        citations: [cited("found")],
        sentences: [{ text: "A third.", cited: true }],
        droppedRecords: 2,
      }),
    ]);
    expect(summary).toMatchObject({
      answers: 2,
      failedAnswers: 1,
      citations: 4,
      foundShare: 0.75,
      falseNotFoundShare: 0.25,
      sentences: 3,
      citedSentences: 2,
      coverage: 2 / 3,
      droppedMarkers: 1,
      droppedRecords: 2,
      citationSupport: { tools: 1, unknown: 1 },
    });
    expect(summariseGroup([])).toMatchObject({ foundShare: null, coverage: null });
  });
});

describe("The reviewer sheet", () => {
  test("lists found quotes only, quoting cells that need it, readable as UTF-8", () => {
    const answer = record({
      questionId: "zh-01",
      language: "zh",
      question: "可持续发展目标一共包含多少项具体目标？",
      citations: [
        {
          ...cited("found", 'It has 169 "targets", in all.'),
          quote: "共有 169 项",
          documentName: "维基百科-可持续发展目标",
          pageTo: 2,
        },
        cited("not-in-document"),
      ],
    });
    const sheet = reviewerSheet({ answers: [answer] } as CitationRun);

    expect(sheet.startsWith("\uFEFF")).toBe(true);
    expect(sheet.slice(1).split("\r\n")).toEqual([
      "question_id,language,round,question,answer_sentence,quote,document,page,supports (y/n),note",
      'zh-01,zh,1,可持续发展目标一共包含多少项具体目标？,"It has 169 ""targets"", in all.",共有 169 项,维基百科-可持续发展目标,1–2,,',
      "",
    ]);
  });
});
