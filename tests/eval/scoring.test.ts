/**
 * How the evaluation (`npm run eval`, eval/README.md) scores what the core
 * returns. The evaluation itself needs the real model and isn't part of
 * `npm test`; its scoring is, since a mistake there would skew every report.
 */
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";
import {
  type AnswerRecord,
  type CitationOutcome,
  type CitationRecord,
  type CitationRun,
  checkQuestionIds,
  citationLine,
  evalOllamaModels,
  outcomeOf,
  questionsToAsk,
  quotePages,
  sentencesOf,
  summariseGroup,
} from "../../eval/lib/citations";
import type { PassageSearchResult } from "../../src/core";
import { checkCitation } from "../../src/core/answers/citations";

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

import { readConfig } from "../../eval/lib/config";
import { loadEvaluationSet } from "../../eval/lib/evaluationSet";
import {
  type EvalReport,
  reviewerSheet,
  terminalSummary,
  writeReports,
} from "../../eval/lib/report";
import {
  candidateCounts,
  checkPassage,
  GATING_LABEL,
  GATING_MODE,
  isHit,
  modesOf,
  type QuestionResult,
  rerankCandidates,
  retrievalFailures,
  scoreRanking,
  summarise,
} from "../../eval/lib/retrieval";
import {
  BUILT_IN_RERANKING_MODEL,
  type Core,
  type OllamaModelProfile,
  type OllamaModels,
  RERANKING_MODEL_CANDIDATES,
} from "../../src/core";

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

  test("where a quote is: the first page that holds it, or two consecutive ones, or nowhere", () => {
    const twoPages = [
      ...pages,
      { page: 4, text: "Costs fell, and then" },
      { page: 5, text: "rose again." },
    ];
    expect(quotePages("Revenue grew by ten percent", twoPages)).toEqual([2, 2]);
    expect(quotePages("and then rose again", twoPages)).toEqual([4, 5]);
    expect(quotePages("Revenue rose by 10%", twoPages)).toBeNull();
  });

  test("a Citation not found reads as a line: the check, the pages cited and its Passage's, where the quote is", () => {
    const line = citationLine({
      sentence: "Revenue grew.",
      quote: "Revenue grew by ten percent",
      documentName: "Report",
      pageFrom: 1,
      pageTo: 1,
      check: "not-found",
      checkReason: "quote-not-on-pages",
      outcome: "wrong-page",
      passagePages: [1, 2],
      quoteOn: [2, 2],
    });
    expect(line).toBe(
      'wrong page (quote-not-on-pages): Report, cites p. 1 of a Passage on pp. 1–2; the quote is on p. 2: "Revenue grew by ten percent"',
    );
  });

  test("a quote of a page whose stored text lost its f-ligatures is the check's miss: a false 'not found'", () => {
    // JP Morgan 2022 Environmental Social Governance Report, p. 8, as stored: pdf.js reads its
    // "fi" ligature as "f", so the page shows "finance" where its text reads "fnance".
    const stored = [
      {
        page: 8,
        text: "set our Sustainable Development Target (the “Target”) with the goal to fnance and\nfacilitate more than $2.5 trillion over 10 years—from 2021 through the end of 2030—",
      },
    ];
    const asShown = "with the goal to finance and facilitate more than $2.5 trillion over 10 years";
    const range = { pageFrom: 8, pageTo: 8 };
    const check = (quote: string) =>
      checkCitation({
        quote,
        range,
        passage: { pageFrom: 7, pageTo: 8 },
        documentDeleted: false,
        pages: stored,
      });

    // The check finds the quote as the stored text reads it, not as the page shows it.
    expect(check(asShown.replace("finance", "fnance")).check).toBe("found");
    const checked = check(asShown);
    expect(checked).toEqual({ check: "not-found", checkReason: "quote-not-on-pages" });
    expect(outcomeOf({ ...checked, ...range, quote: asShown }, stored)).toBe("false-not-found");
    // "efforts" for "eforts", "office" for "ofce", "fifty" for a lost ligature's "ffty".
    const pages = [{ page: 1, text: "our eforts in the ofce, ffty in all" }];
    expect(
      outcomeOf(
        { ...checked, pageFrom: 1, pageTo: 1, quote: "our efforts in the office, fifty in all" },
        pages,
      ),
    ).toBe("false-not-found");
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

  test("a ranking is a hit in its top 5, and a near miss shows its rank in the top 20", () => {
    const miss = passage({ documentId: "doc-2" });
    const ranked = (at: number) =>
      Array.from({ length: 20 }, (_, index) => (index === at ? passage({}) : miss));
    expect(scoreRanking(ranked(4), expected, "doc-1")).toMatchObject({ hit: true, rank: 5 });
    expect(scoreRanking(ranked(5), expected, "doc-1")).toMatchObject({ hit: false, rank: 6 });
    expect(scoreRanking([miss], expected, "doc-1")).toMatchObject({ hit: false, rank: null });
    expect(scoreRanking(ranked(0), expected, "doc-1").top).toHaveLength(5);
  });
});

describe("Retrieval tallies", () => {
  const result = (
    id: string,
    modes: QuestionResult["modes"],
    extra: Partial<QuestionResult> = {},
  ): QuestionResult => ({
    id,
    language: id.startsWith("zh") ? "zh" : "en",
    crossLingual: id.startsWith("xl"),
    question: id,
    modes,
    ...extra,
  });
  const at = (rank: number | null) => ({ hit: rank !== null && rank <= 5, rank, top: [] });

  test("reranked modes are tallied like the others; a translated second query counts when either search hits", () => {
    const results = [
      result("en-01", { hybrid: at(7), "rerank:mmarco-minilm": at(2) }),
      result("zh-01", { hybrid: at(1), "rerank:mmarco-minilm": at(1) }),
      result(
        "xl-01",
        { hybrid: at(null), "rerank:mmarco-minilm": at(null) },
        { translated: { hybrid: at(3), "rerank:mmarco-minilm": at(9) } },
      ),
      result(
        "xl-02",
        { hybrid: at(4), "rerank:mmarco-minilm": at(1) },
        { translated: { hybrid: at(null), "rerank:mmarco-minilm": at(1) } },
      ),
      // No translation: it isn't in the translated tally.
      result("xl-03", { hybrid: at(2), "rerank:mmarco-minilm": at(2) }),
    ];

    const summary = summarise(results);

    expect(modesOf(results)).toEqual(["hybrid", "rerank:mmarco-minilm"]);
    expect(summary.hybrid).toEqual({
      en: { hits: 0, total: 1 },
      zh: { hits: 1, total: 1 },
      core: { hits: 1, total: 2 },
      crossLingual: { hits: 2, total: 3 },
      crossLingualTranslated: { hits: 2, total: 2 },
    });
    expect(summary["rerank:mmarco-minilm"]).toMatchObject({
      en: { hits: 1, total: 1 },
      crossLingual: { hits: 2, total: 3 },
      crossLingualTranslated: { hits: 1, total: 2 },
    });
    // Keyword search never ran on the translated query: no tally rather than a misleading one.
    expect(summarise([result("xl-01", { keyword: at(1) })]).keyword?.crossLingualTranslated).toBe(
      null,
    );
  });
});

describe("What the reranked modes rerank", () => {
  const passage = (id: string): PassageSearchResult => ({
    passageId: id,
    documentId: "doc-1",
    documentName: "Tides",
    pageFrom: 1,
    pageTo: 1,
    position: 0,
    text: id,
  });

  test("keyword search's top 10 and vector search's top 10, each Passage once, as the search Tool hands them", async () => {
    const asked: string[] = [];
    const ranked = (prefix: string) =>
      Array.from({ length: 12 }, (_, index) => passage(`${prefix}${index}`));
    const core = {
      searchPassages: async (_query: string, options: { mode: string; limit: number }) => {
        asked.push(`${options.mode} ${options.limit}`);
        // Both lists share k0 and k1; vector search adds 8 of its own in its top 10.
        return options.mode === "keyword"
          ? ranked("k").slice(0, options.limit)
          : [passage("k1"), passage("k0"), ...ranked("v")].slice(0, options.limit);
      },
    } as unknown as Core;

    const candidates = await rerankCandidates(core, "tides");

    expect(asked.sort()).toEqual(["keyword 10", "vector 10"]);
    expect(candidates.map((each) => each.passageId)).toEqual([
      ...Array.from({ length: 10 }, (_, index) => `k${index}`),
      ...Array.from({ length: 8 }, (_, index) => `v${index}`),
    ]);
  });

  test("the report counts candidates per reranked search, the translated queries' too", () => {
    const result = (rerank?: QuestionResult["rerankCandidates"]): QuestionResult => ({
      id: "x",
      language: "en",
      crossLingual: false,
      question: "x",
      modes: {},
      ...(rerank && { rerankCandidates: rerank }),
    });

    expect(candidateCounts([result()])).toBeNull();
    expect(
      candidateCounts([
        result({ question: 12 }),
        result({ question: 20, translated: 16 }),
        result(),
      ]),
    ).toEqual({ perList: 10, searches: 3, mean: 16, min: 12, max: 20 });
  });
});

describe("The retrieval gate", () => {
  const tally = (hits: number, total = 20) => ({ hits, total });
  const summary = (en: number, zh: number) => ({
    en: tally(en),
    zh: tally(zh),
    core: tally(en + zh, 40),
    crossLingual: tally(2, 10),
    crossLingualTranslated: tally(7, 10),
  });

  test("is what the search Tool does by default: hybrid search reranked by the built-in model", () => {
    expect(GATING_MODE).toBe(`rerank:${BUILT_IN_RERANKING_MODEL.id}`);
    expect(GATING_LABEL).toBe(`hybrid + ${BUILT_IN_RERANKING_MODEL.name}`);
    expect(retrievalFailures(summary(16, 20))).toEqual([]);
    expect(retrievalFailures(summary(13, 19))).toEqual([
      `Retrieval (hybrid + ${BUILT_IN_RERANKING_MODEL.name}, the built-in models), English: 13 of 20, needs 16.`,
    ]);
    expect(retrievalFailures(undefined)).toEqual([
      `No hybrid + ${BUILT_IN_RERANKING_MODEL.name} results.`,
    ]);
  });

  test("the summary marks the gating row, and reports plain hybrid search next to it", () => {
    const report = {
      result: "pass",
      failures: [],
      run: {
        startedAt: "2026-10-09T08:00:00.000Z",
        seconds: 60,
        commit: "abc1234",
        node: "v25",
        platform: "darwin arm64",
        cpu: "M2",
      },
      evaluationSet: {
        source: "eval/retrieval/questions.json",
        hitRule: "",
        questions: { gating: { en: 20, zh: 20 }, crossLingual: 10 },
      },
      documents: [],
      retrieval: {
        topK: 5,
        gatingMode: GATING_MODE,
        runs: [
          {
            embedding: "multilingual-e5-small (built-in)",
            gating: true,
            passageCount: 1199,
            processingSeconds: 60,
            questions: [],
            summary: { hybrid: summary(13, 19), [GATING_MODE]: summary(16, 20) },
          },
        ],
      },
      citations: { skipped: "no chat model" },
    } satisfies EvalReport;

    const lines = terminalSummary(report, "/repo/eval/results/x", "/repo").split("\n");

    expect(lines.find((line) => line.includes("(gating)"))).toContain(
      `hybrid + ${BUILT_IN_RERANKING_MODEL.name} (gating) English 16/20`,
    );
    expect(lines.find((line) => line.trim().startsWith("hybrid "))).toContain("English 13/20");
  });
});

describe("The evaluation's settings for reranking", () => {
  test("INCARNAMIND_EVAL_RERANK names candidates, or all of them; none by default", () => {
    expect(readConfig("/repo", {}).rerank).toEqual([]);
    expect(
      readConfig("/repo", { INCARNAMIND_EVAL_RERANK: "all" }).rerank.map((each) => each.id),
    ).toEqual(RERANKING_MODEL_CANDIDATES.map((each) => each.id));
    expect(
      readConfig("/repo", { INCARNAMIND_EVAL_RERANK: "bge-m3, mmarco-minilm" }).rerank.map(
        (each) => each.id,
      ),
    ).toEqual(["bge-m3", "mmarco-minilm"]);
    expect(() => readConfig("/repo", { INCARNAMIND_EVAL_RERANK: "jina-v2" })).toThrow(/jina-v2/);
  });

  test("every cross-lingual Question has a translated query, and only those do", () => {
    const set = loadEvaluationSet(fileURLToPath(new URL("../..", import.meta.url)));
    for (const question of set.questions) {
      expect(question.translatedQuery !== undefined, question.id).toBe(question.crossLingual);
    }
  });
});

describe("The evaluation's settings for a short check", () => {
  test("INCARNAMIND_EVAL_QUESTIONS names Questions by id, each once; all of them by default", () => {
    expect(readConfig("/repo", {}).questionIds).toBeNull();
    expect(
      readConfig("/repo", { INCARNAMIND_EVAL_QUESTIONS: " en-07, zh-02,,en-07 " }).questionIds,
    ).toEqual(["en-07", "zh-02"]);
    expect(() => readConfig("/repo", { INCARNAMIND_EVAL_QUESTIONS: ", ," })).toThrow(
      /Question ids separated by commas/,
    );
  });

  test("INCARNAMIND_EVAL_FORMATS=off skips the every-format set, which runs by default", () => {
    expect(readConfig("/repo", {}).formats).toBe(true);
    expect(readConfig("/repo", { INCARNAMIND_EVAL_FORMATS: "on" }).formats).toBe(true);
    expect(readConfig("/repo", { INCARNAMIND_EVAL_FORMATS: "off" }).formats).toBe(false);
    expect(() => readConfig("/repo", { INCARNAMIND_EVAL_FORMATS: "no" })).toThrow(/"on" or "off"/);
  });

  test("the Questions named are asked in the set's order; an id in no set asked from is refused", () => {
    const set = loadEvaluationSet(fileURLToPath(new URL("../..", import.meta.url)));
    expect(questionsToAsk(set.questions, null)).toHaveLength(set.questions.length);
    expect(questionsToAsk(set.questions, ["zh-02", "en-07"]).map((each) => each.id)).toEqual([
      "en-07",
      "zh-02",
    ]);
    expect(() => checkQuestionIds(["en-07", "zh-02"], [set.questions])).not.toThrow();
    expect(() => checkQuestionIds(["en-07", "en-99", "docx-en-01"], [set.questions])).toThrow(
      /en-99, docx-en-01/,
    );
  });
});

describe("The evaluation's settings for a model in Ollama", () => {
  test("an Ollama run may set num_ctx and the citing mode; no other kind may", () => {
    const ollama = {
      INCARNAMIND_EVAL_CHAT_KIND: "ollama",
      INCARNAMIND_EVAL_CHAT_MODEL: "qwen3.5:4b",
    };
    expect(readConfig("/repo", ollama).chat).toMatchObject({ numCtx: null, citing: null });
    expect(
      readConfig("/repo", {
        ...ollama,
        INCARNAMIND_EVAL_CHAT_NUM_CTX: "8192",
        INCARNAMIND_EVAL_CHAT_CITING: "structured-output",
      }).chat,
    ).toMatchObject({ numCtx: 8192, citing: "structured-output" });
    expect(() => readConfig("/repo", { ...ollama, INCARNAMIND_EVAL_CHAT_CITING: "json" })).toThrow(
      /json/,
    );
    expect(() =>
      readConfig("/repo", {
        INCARNAMIND_EVAL_CHAT_KIND: "anthropic",
        INCARNAMIND_EVAL_CHAT_MODEL: "a-model",
        INCARNAMIND_EVAL_CHAT_KEY: "a-key",
        INCARNAMIND_EVAL_CHAT_NUM_CTX: "8192",
      }),
    ).toThrow(/only apply to "ollama"/);
  });

  test("an Ollama run's window and citing mode replace the app's choice", async () => {
    const profile: OllamaModelProfile = {
      digest: "d1",
      capabilities: ["completion", "tools"],
      contextLength: 262_144,
      parameters: 9_653_104_368,
      support: "tools",
      chat: true,
      settings: { numCtx: 16_384, outputTokens: 4_096, keepAlive: "30m", think: false },
    };
    const app: OllamaModels = { describe: async () => profile, loaded: async () => true };
    const chat = {
      kind: "ollama" as const,
      modelId: "qwen3.5:4b",
      apiKey: null,
      baseUrl: null,
    };
    expect(evalOllamaModels({ ...chat, numCtx: null, citing: null }, app)).toBeNull();
    const models = evalOllamaModels({ ...chat, numCtx: 8_192, citing: "structured-output" }, app);
    expect(await models?.describe("http://127.0.0.1:11434", "qwen3.5:4b")).toEqual({
      ...profile,
      support: "structured-output",
      settings: { ...profile.settings, numCtx: 8_192, outputTokens: 2_048 },
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
  passagePages: [1, 1],
  quoteOn: outcome === "not-in-document" ? null : [1, 1],
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
        seconds: 4,
      }),
      record({ sentences: [{ text: "Uncited.", cited: false }], seconds: 2 }),
    ]);
    expect(summary).toMatchObject({
      answers: 3,
      failedAnswers: 1,
      citedAnswers: 2,
      citedAnswerShare: 2 / 3,
      medianSeconds: 2,
      citations: 4,
      foundShare: 0.75,
      falseNotFoundShare: 0.25,
      sentences: 4,
      citedSentences: 2,
      coverage: 2 / 4,
      droppedMarkers: 1,
      droppedRecords: 2,
      citationSupport: { tools: 2, unknown: 1 },
    });
    expect(summariseGroup([])).toMatchObject({
      foundShare: null,
      coverage: null,
      citedAnswerShare: null,
      medianSeconds: null,
    });
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

describe("Why Citations aren't found, in report.md", () => {
  test("lists each Citation not found and each Answer without one; a short run says it doesn't gate", async () => {
    const answers = [
      record({
        citations: [
          cited("found"),
          {
            ...cited("wrong-page"),
            quote: "a quote | with a pipe",
            pageFrom: 7,
            pageTo: 7,
            passagePages: [7, 8],
            quoteOn: [8, 8],
          },
        ],
      }),
      record({
        questionId: "zh-02",
        language: "zh",
        citationSupport: "structured-output",
        droppedMarkers: 2,
        searches: ["交通事故"],
        sentences: [{ text: "文档没有提到。", cited: false }],
      }),
    ];
    const citations: CitationRun = {
      model: "ollama/qwen3.5:4b",
      service: null,
      gating: false,
      subset: ["en-01", "zh-02"],
      overrides: ["num_ctx 8192"],
      minCitations: 30,
      rounds: 1,
      answers,
      summary: {
        en: summariseGroup([answers[0] as AnswerRecord]),
        zh: summariseGroup([answers[1] as AnswerRecord]),
        crossLingual: summariseGroup([]),
      },
      failures: [],
    };
    const report = {
      result: "pass",
      failures: [],
      run: {
        startedAt: "2026-10-10T08:00:00.000Z",
        seconds: 1,
        commit: "",
        node: "",
        platform: "",
        cpu: "",
      },
      evaluationSet: {
        source: "",
        hitRule: "",
        questions: { gating: { en: 1, zh: 1 }, crossLingual: 0 },
      },
      documents: [],
      retrieval: {
        topK: 5,
        gatingMode: GATING_MODE,
        runs: [
          {
            embedding: "multilingual-e5-small (built-in)",
            gating: true,
            passageCount: 1,
            processingSeconds: 1,
            questions: [],
            summary: summarise([]),
          },
        ],
      },
      citations,
      formats: { skipped: "INCARNAMIND_EVAL_FORMATS is off." },
    } satisfies EvalReport;
    const results = await mkdtemp(join(tmpdir(), "why-report-"));
    try {
      const dir = await writeReports(report, results, "/");
      const markdown = (await readFile(join(dir, "report.md"), "utf8")).split("\n");
      expect(markdown.join("\n")).toContain(
        "Only 2 of the Questions were asked (INCARNAMIND_EVAL_QUESTIONS: en-01, zh-02), once each: a short check, reported, never gating.",
      );
      expect(markdown).toContain(
        '| en-01 | 1 | "Not found": quote on other pages | quote-not-on-pages | Tides | p. 7 | pp. 7–8 | p. 8 | a quote \\| with a pipe |',
      );
      expect(markdown).toContain(
        "| zh-02 | 1 | done | structured-output | 2 | 0 | 交通事故 | 文档没有提到。 |",
      );
      expect(markdown).toContain("Skipped: INCARNAMIND_EVAL_FORMATS is off.");
      const summary = terminalSummary(report, "/repo/eval/results/x", "/repo");
      expect(summary).toContain(
        "Citation quality, ollama/qwen3.5:4b (2 Questions only, not gating)",
      );
      expect(summary).toContain("Every format: skipped. INCARNAMIND_EVAL_FORMATS is off.");
    } finally {
      await rm(results, { recursive: true, force: true });
    }
  });
});
