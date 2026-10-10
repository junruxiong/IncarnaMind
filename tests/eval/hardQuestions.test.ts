/**
 * The hard tier's Questions (eval/hard/questions.json, eval/hard/README.md):
 * the committed set's spread over difficulties, languages and domains, the
 * rules each difficulty's Questions follow, and the check of each Question
 * against the text the app stored, which every run makes before scoring.
 */
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";
import { type HardDocument, loadManifest, type Manifest } from "../../eval/hard/lib/manifest";
import {
  DIFFICULTIES,
  type HardQuestion,
  loadHardQuestions,
  questionProblems,
  type StoredText,
  textProblems,
} from "../../eval/hard/lib/questions";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));

const document = (key: string, extra: Partial<HardDocument> = {}): HardDocument => ({
  key,
  title: key,
  domain: "filings",
  language: "en",
  format: "pdf",
  source: "open",
  url: `https://example.org/${key}.pdf`,
  sha256: "0".repeat(64),
  bytes: 1,
  ...extra,
});

const LIBRARY: Manifest = {
  version: 1,
  description: "",
  sources: {},
  archives: [],
  documents: [
    document("report-2022", { group: "report", edition: "2022" }),
    document("report-2023", { group: "report", edition: "2023" }),
    document("contract", { domain: "contracts" }),
    document("bao-gao", { language: "zh", domain: "reports" }),
    document("sheet", { format: "xlsx" }),
  ],
};

const question = (extra: Partial<HardQuestion>): HardQuestion => ({
  id: "q",
  language: "en",
  domain: "filings",
  difficulty: "easy",
  question: "What was revenue?",
  expected: [{ document: "report-2023", pages: [3, 3], quote: "Revenue was 12 million" }],
  ...extra,
});

describe("The hard tier's Questions", () => {
  const library = loadManifest(ROOT);
  const set = loadHardQuestions(ROOT, library);

  test("the committed set has 150 to 250 Questions, at least 15 per difficulty, in both languages", () => {
    expect(set.questions.length).toBeGreaterThanOrEqual(150);
    expect(set.questions.length).toBeLessThanOrEqual(250);
    for (const difficulty of DIFFICULTIES) {
      const tagged = set.questions.filter((each) => each.difficulty === difficulty);
      expect(tagged.length, difficulty).toBeGreaterThanOrEqual(15);
      expect(
        tagged.some((each) => each.language === "en"),
        `${difficulty} in English`,
      ).toBe(true);
      expect(
        tagged.some((each) => each.language === "zh"),
        `${difficulty} in Chinese`,
      ).toBe(true);
    }
  });

  test("every domain has Questions, and every expected Document is in the library", () => {
    const keys = new Set(library.documents.map((each) => each.key));
    const domains = new Set(set.questions.map((each) => each.domain));
    expect(domains.size).toBe(6);
    for (const each of set.questions) {
      for (const expected of each.expected) expect(keys, each.id).toContain(expected.document);
    }
  });

  test("a sound Question has no problems", () => {
    expect(questionProblems(question({}), LIBRARY)).toEqual([]);
  });

  test("an unknown difficulty, domain, Document or page range is a problem", () => {
    expect(
      questionProblems(
        question({
          difficulty: "tricky" as never,
          domain: "cooking" as never,
          expected: [{ document: "nowhere", pages: [3, 5], quote: "x" }],
        }),
        LIBRARY,
      ),
    ).toEqual([
      'q: unknown domain "cooking".',
      'q: unknown difficulty "tricky".',
      'q: unknown Document "nowhere".',
      "q: expected pages must be [first, last], one Unit or two consecutive ones.",
      "q: the domain isn't any expected Document's.",
    ]);
  });

  test("an unanswerable Question expects nothing; the others expect a Passage", () => {
    expect(questionProblems(question({ difficulty: "unanswerable" }), LIBRARY)).toEqual([
      "q: an unanswerable Question expects no Passage.",
    ]);
    expect(
      questionProblems(question({ difficulty: "unanswerable", expected: [] }), LIBRARY),
    ).toEqual([]);
    expect(questionProblems(question({ expected: [] }), LIBRARY)).toEqual([
      "q: expects no Passage.",
    ]);
  });

  test("a multi-page Question needs two Passages of one Document, on different pages", () => {
    const at = (document: string, page: number) => ({
      document,
      pages: [page, page] as [number, number],
      quote: "x",
    });
    const multi = (expected: HardQuestion["expected"]) =>
      questionProblems(question({ difficulty: "multi-page", expected }), LIBRARY);
    expect(multi([at("report-2023", 3), at("report-2023", 9)])).toEqual([]);
    expect(multi([at("report-2023", 3)])).toEqual([
      "q: a multi-page Question expects two or more Passages of one Document.",
    ]);
    expect(multi([at("report-2023", 3), at("report-2023", 3)])).toEqual([
      "q: a multi-page Question's Passages must be on different pages.",
    ]);
    expect(multi([at("report-2023", 3), at("report-2022", 9)])).toEqual([
      "q: a multi-page Question expects two or more Passages of one Document.",
    ]);
  });

  test("a cross-document Question needs two Documents", () => {
    const expected = [
      { document: "report-2023", pages: [1, 1] as [number, number], quote: "a" },
      { document: "sheet", pages: [2, 2] as [number, number], quote: "b" },
    ];
    expect(questionProblems(question({ difficulty: "cross-document", expected }), LIBRARY)).toEqual(
      [],
    );
    expect(
      questionProblems(
        question({ difficulty: "cross-document", expected: [expected[0] as never] }),
        LIBRARY,
      ),
    ).toEqual(["q: a cross-document Question expects two or more Documents."]);
  });

  test("a cross-lingual Question is in the other language from its Document, with a translation; no other Question is", () => {
    const zh = { document: "bao-gao", pages: [1, 1] as [number, number], quote: "营业收入" };
    const crossLingual = (extra: Partial<HardQuestion>) =>
      questionProblems(
        question({ difficulty: "cross-lingual", domain: "reports", expected: [zh], ...extra }),
        LIBRARY,
      );
    expect(crossLingual({ translatedQuery: "营业收入是多少？" })).toEqual([]);
    expect(crossLingual({})).toEqual(["q: a cross-lingual Question needs a translatedQuery."]);
    expect(crossLingual({ language: "zh", translatedQuery: "x" })).toEqual([
      "q: a cross-lingual Question is asked in the other language from its Documents'.",
    ]);
    expect(questionProblems(question({ domain: "reports", expected: [zh] }), LIBRARY)).toEqual([
      "q: asked in another language from its Documents': tag it cross-lingual.",
    ]);
    expect(questionProblems(question({ translatedQuery: "x" }), LIBRARY)).toEqual([
      "q: only a cross-lingual Question has a translatedQuery.",
    ]);
  });

  test("a near-duplicate Question's Document has another version in its language", () => {
    expect(questionProblems(question({ difficulty: "near-duplicate" }), LIBRARY)).toEqual([]);
    expect(
      questionProblems(
        question({
          difficulty: "near-duplicate",
          domain: "contracts",
          expected: [{ document: "contract", pages: [1, 1], quote: "x" }],
        }),
        LIBRARY,
      ),
    ).toEqual(["q: contract has no other version in its language for a near-duplicate Question."]);
  });

  test("the loader fails on ids used twice, saying which", () => {
    expect(() =>
      loadHardQuestions(ROOT, library, "tests/fixtures/hard-questions-twice.json"),
    ).toThrow(/ids used twice: twice/);
  });
});

describe("Checking a Question against the stored text", () => {
  const page = (number: number, text: string) => ({ page: number, kind: "page" as const, text });
  const stored: Record<string, StoredText> = {
    "report-2023": {
      units: [page(1, "Contents"), page(3, "Revenue was 12 million in 2023."), page(4, "Outlook.")],
      passages: [{ pageFrom: 3, pageTo: 4, text: "Revenue was 12 million in 2023.\n\nOutlook." }],
    },
    "report-2022": {
      units: [page(3, "Revenue was 10 million in 2022.")],
      passages: [{ pageFrom: 3, pageTo: 3, text: "Revenue was 10 million in 2022." }],
    },
    sheet: {
      units: [
        { page: 1, kind: "rows", label: { sheet: "A" }, text: "Total\t4,812" },
        { page: 2, kind: "rows", label: { sheet: "B" }, text: "Total\t9" },
      ],
      passages: [{ pageFrom: 1, pageTo: 1, text: "Total\t4,812" }],
    },
  };
  const check = (extra: Partial<HardQuestion>) =>
    textProblems(question(extra), (key) => stored[key], LIBRARY);

  test("a quote on its expected Unit, and in a Passage that covers it, can be scored", () => {
    expect(check({})).toEqual([]);
  });

  test("a quote not on its Unit, also on another, or in no covering Passage can't", () => {
    expect(
      check({
        expected: [{ document: "report-2023", pages: [4, 4], quote: "Revenue was 12 million" }],
      }),
    ).toEqual(["report-2023: the quote isn't on Unit 4."]);
    expect(
      check({ expected: [{ document: "report-2023", pages: [3, 3], quote: "Revenue was" }] }),
    ).toEqual([]);
    const again: Record<string, StoredText> = {
      ...stored,
      "report-2023": {
        units: [page(3, "Revenue was 12 million."), page(7, "As said, revenue was 12 million.")],
        passages: [{ pageFrom: 3, pageTo: 3, text: "Other text." }],
      },
    };
    expect(textProblems(question({}), (key) => again[key], LIBRARY)).toEqual([
      "report-2023: the quote is also on Unit 7.",
      "report-2023: no Passage that covers the quote's Units holds it.",
    ]);
  });

  test("a near-duplicate's quote must be in no other version", () => {
    expect(
      check({
        difficulty: "near-duplicate",
        expected: [{ document: "report-2023", pages: [3, 3], quote: "Revenue was" }],
      }),
    ).toEqual(["report-2023: the quote is in report-2022 too."]);
    expect(check({ difficulty: "near-duplicate" })).toEqual([]);
  });

  test("a Document left out of the library, a missing Unit, and Units of two sheets are reported", () => {
    expect(check({ expected: [{ document: "contract", pages: [1, 1], quote: "x" }] })).toEqual([
      "contract isn't in the library.",
    ]);
    expect(check({ expected: [{ document: "report-2023", pages: [9, 9], quote: "x" }] })).toEqual([
      "report-2023 has no Unit 9.",
    ]);
    expect(
      check({ expected: [{ document: "sheet", pages: [1, 2], quote: "Total 4812" }] }),
    ).toContain("sheet: Units 1–2 can't be cited together.");
    expect(
      check({ expected: [{ document: "sheet", pages: [1, 1], quote: "Total 4,812" }] }),
    ).toEqual([]);
  });
});
