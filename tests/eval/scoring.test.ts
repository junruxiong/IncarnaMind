/**
 * How the evaluation (`npm run eval`, eval/README.md) scores what the core
 * returns. The evaluation itself needs the real model and isn't part of
 * `npm test`; its scoring is, since a mistake there would skew every report.
 */
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
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

import { readConfig } from "../../eval/lib/config";
import { type EvalQuestion, isGating, loadEvaluationSet } from "../../eval/lib/evaluationSet";
import {
  type EvalReport,
  reviewerSheet,
  terminalSummary,
  writeReports,
} from "../../eval/lib/report";
import type { OpenReranker, RerankerInfo } from "../../eval/lib/rerank";
import {
  type CandidateRecord,
  candidateCounts,
  checkPassage,
  coreSource,
  GATING_LABEL,
  GATING_MODE,
  GATING_SEARCH,
  HYBRID_RERANK_LABEL,
  HYBRID_RERANK_MODE,
  isHit,
  keywordRerankCandidates,
  modesOf,
  OTHER_SEARCHES,
  type QuestionResult,
  RERANKED_SEARCHES,
  type RetrievalRun,
  rerankCandidates,
  rerankedLabel,
  rerankedSearchOf,
  rerankMode,
  retrievalFailures,
  runReranked,
  scoreRanking,
  summarise,
} from "../../eval/lib/retrieval";
import { BUILT_IN_RERANKING_MODEL, type Core, RERANKING_MODEL_CANDIDATES } from "../../src/core";
import { SEARCH_TOOL_PARAMETERS } from "../../src/core/documents/searchTool";

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
      paraphrase: {
        en: { hits: 0, total: 0 },
        zh: { hits: 0, total: 0 },
        all: { hits: 0, total: 0 },
      },
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

  test("paraphrase Questions are tallied apart, out of the gating counts and the bar", () => {
    const paraphrase = { paraphrase: true };
    const results = [
      ...["en-01", "en-02", "en-03", "en-04", "en-05", "zh-01", "zh-02", "zh-03", "zh-04"].map(
        (id) => result(id, { hybrid: at(1) }),
      ),
      result("para-en-01", { hybrid: at(null) }, paraphrase),
      result("para-en-02", { hybrid: at(7) }, paraphrase),
      result("para-zh-01", { hybrid: at(4) }, { ...paraphrase, language: "zh" }),
    ];

    const hybrid = summarise(results).hybrid;

    expect(hybrid).toMatchObject({
      en: { hits: 5, total: 5 },
      zh: { hits: 4, total: 4 },
      core: { hits: 9, total: 9 },
      paraphrase: {
        en: { hits: 0, total: 2 },
        zh: { hits: 1, total: 1 },
        all: { hits: 1, total: 3 },
      },
    });
    // Counted in, the two English misses would take English to 5 of 7, under 80%.
    expect(retrievalFailures(hybrid)).toEqual([]);
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

  test("the report counts candidates per reranked search, the translated queries' too, and how often they held the expected Passage", () => {
    const result = (
      hybrid?: CandidateRecord,
      extra: Partial<QuestionResult> = {},
    ): QuestionResult => ({
      id: "x",
      language: "en",
      crossLingual: false,
      question: "x",
      modes: {},
      ...(hybrid && { candidates: { hybrid } }),
      ...extra,
    });

    expect(candidateCounts([result()])).toBeNull();
    expect(
      candidateCounts([
        result({ question: 12, reached: 3 }),
        result({ question: 20, translated: 16, reached: null }, { crossLingual: true }),
        result({ question: 18, reached: 18 }, { paraphrase: true }),
        result(),
      ]),
    ).toEqual({
      search: "hybrid",
      searches: 4,
      mean: 16.5,
      min: 12,
      max: 20,
      reached: { gating: { hits: 1, total: 1 }, paraphrase: { hits: 1, total: 1 } },
    });
    // Keyword + rerank's counts are its own.
    expect(candidateCounts([result({ question: 12, reached: null })], "keyword")).toBeNull();
  });

  test("the gate reranks keyword search's top 60 alone, with no vector search, as the search Tool does with embeddings off", async () => {
    const asked: string[] = [];
    const core = {
      searchPassages: async (_query: string, options: { mode: string; limit: number }) => {
        asked.push(`${options.mode} ${options.limit}`);
        return Array.from({ length: options.limit }, (_, index) => passage(`k${index}`));
      },
    } as unknown as Core;

    const candidates = await keywordRerankCandidates(core, "tides");

    expect(SEARCH_TOOL_PARAMETERS.keywordRerankCandidates).toBe(60);
    expect(asked).toEqual(["keyword 60"]);
    expect(candidates.map((each) => each.passageId)).toEqual(
      Array.from({ length: 60 }, (_, index) => `k${index}`),
    );
    const gate = await coreSource(core, GATING_SEARCH)?.("tides", {} as EvalQuestion);
    expect(gate?.candidates).toHaveLength(60);
  });

  test("keyword top 20 + rerank, the gate before, reranks keyword search's candidates and records their counts apart", async () => {
    const asked: string[] = [];
    // Keyword search finds 14 Passages; the expected one is its last.
    const keyword = Array.from({ length: 14 }, (_, index) => ({
      ...passage(`k${index}`),
      text: index === 13 ? "Spring tides happen at full moon." : `k${index}`,
    }));
    const core = {
      searchPassages: async (query: string, options: { mode: string; limit: number }) => {
        asked.push(`${query}: ${options.mode} ${options.limit}`);
        return options.mode === "keyword" ? keyword.slice(0, options.limit) : [];
      },
    } as unknown as Core;
    const reversing = (mode: string): OpenReranker => ({
      mode,
      rerank: async (_query, passages) => [...passages].reverse(),
      info: () => ({}) as RerankerInfo,
      close: () => {},
    });
    const question: EvalQuestion = {
      id: "xl-01",
      language: "zh",
      crossLingual: true,
      question: "大潮什么时候出现？",
      translatedQuery: "When are spring tides?",
      expected: { document: "tides", pages: [1, 1], quote: "Spring tides happen at full moon." },
    };
    const results: QuestionResult[] = [
      {
        id: "xl-01",
        language: "zh",
        crossLingual: true,
        question: question.question,
        modes: {},
        translated: {},
      },
    ];
    const ids = new Map([["tides", "doc-1"]]);

    const found = await runReranked(
      core,
      [question],
      ids,
      reversing("keyword-rerank:fake"),
      results,
    );

    expect(asked).toEqual(["大潮什么时候出现？: keyword 20", "When are spring tides?: keyword 20"]);
    expect(results[0]?.modes["keyword-rerank:fake"]).toMatchObject({ hit: true, rank: 1 });
    expect(results[0]?.translated?.["keyword-rerank:fake"]).toMatchObject({ hit: true, rank: 1 });
    // The expected Passage was the 14th candidate, before the reranker put it first.
    expect(results[0]?.candidates).toEqual({
      keyword: { question: 14, translated: 14, reached: 14 },
    });
    expect(found.queries).toBe(2);
    expect(candidateCounts(results, "keyword")).toEqual({
      search: "keyword",
      searches: 2,
      mean: 14,
      min: 14,
      max: 14,
      reached: { gating: { hits: 0, total: 0 }, paraphrase: { hits: 0, total: 0 } },
    });
    expect(summarise(results)["keyword-rerank:fake"]?.crossLingualTranslated).toEqual({
      hits: 1,
      total: 1,
    });
  });
});

describe("The reranked modes", () => {
  test("keyword top 60 + rerank with the built-in model is the gate; hybrid + rerank, as with embeddings on, and keyword top 20, the gate before, are reported", () => {
    expect(GATING_MODE).toBe(`keyword-60-rerank:${BUILT_IN_RERANKING_MODEL.id}`);
    expect(GATING_LABEL).toBe(`keyword top 60 + ${BUILT_IN_RERANKING_MODEL.name}`);
    expect(HYBRID_RERANK_MODE).toBe(`rerank:${BUILT_IN_RERANKING_MODEL.id}`);
    expect(HYBRID_RERANK_LABEL).toBe(`hybrid + ${BUILT_IN_RERANKING_MODEL.name}`);
    expect(rerankMode(BUILT_IN_RERANKING_MODEL, "keyword-60")).toBe(GATING_MODE);
    expect(rerankMode(BUILT_IN_RERANKING_MODEL)).toBe(HYBRID_RERANK_MODE);
    expect([GATING_MODE, HYBRID_RERANK_MODE, "hybrid"].map(rerankedSearchOf)).toEqual([
      "keyword-60",
      "hybrid",
      null,
    ]);
    // Every reranking model runs the gate's search and hybrid's; the other seven, the built-in one.
    expect(RERANKED_SEARCHES).toEqual(["keyword-60", "hybrid"]);
    expect(OTHER_SEARCHES).toEqual([
      "keyword",
      "keyword-40",
      "feedback",
      "rewrites",
      "sub-questions",
      "small-to-big",
      "document-first",
    ]);
    expect(rerankedLabel("keyword", BUILT_IN_RERANKING_MODEL.name)).toBe(
      `keyword top 20 + ${BUILT_IN_RERANKING_MODEL.name}`,
    );
  });
});

describe("The retrieval gate", () => {
  const tally = (hits: number, total = 20) => ({ hits, total });
  /** A mode's summary over today's set: 20 + 20 gating Questions, 10 cross-lingual, 8 + 7 paraphrase. */
  const summary = (en: number, zh: number, [paraphraseEn, paraphraseZh] = [0, 0]) => ({
    en: tally(en),
    zh: tally(zh),
    core: tally(en + zh, 40),
    crossLingual: tally(2, 10),
    crossLingualTranslated: tally(7, 10),
    paraphrase: {
      en: tally(paraphraseEn, 8),
      zh: tally(paraphraseZh, 7),
      all: tally(paraphraseEn + paraphraseZh, 15),
    },
  });

  test("is what the search Tool does by default: keyword search reranked by the built-in model", () => {
    expect(retrievalFailures(summary(16, 20))).toEqual([]);
    expect(retrievalFailures(summary(13, 19))).toEqual([
      `Retrieval (keyword top 60 + ${BUILT_IN_RERANKING_MODEL.name}, the built-in reranking model), English: 13 of 20, needs 16.`,
    ]);
    expect(retrievalFailures(undefined)).toEqual([
      `No keyword top 60 + ${BUILT_IN_RERANKING_MODEL.name} results.`,
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
        questions: { gating: { en: 20, zh: 20 }, crossLingual: 10, paraphrase: 0 },
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
      `keyword top 60 + ${BUILT_IN_RERANKING_MODEL.name} (gating) English 16/20`,
    );
    expect(lines.find((line) => line.trim().startsWith("hybrid "))).toContain("English 13/20");
  });

  test("hybrid + rerank is reported next to the gating row, never gating, and paraphrase Questions in a column of their own", async () => {
    const reranker = (mode: string): RerankerInfo => ({
      mode,
      name: BUILT_IN_RERANKING_MODEL.name,
      licence: "Apache-2.0",
      downloadBytes: 136e6,
      loadSeconds: 1.5,
      latency: { queries: 74, mean: 700, median: 690, p95: 900, max: 950 },
      candidates: { queries: 75, mean: 3, median: 2, p95: 6, max: 9 },
    });
    const reached = (gating: number, paraphrase: number) => ({
      gating: { hits: gating, total: 40 },
      paraphrase: { hits: paraphrase, total: 15 },
    });
    const run = {
      embedding: "multilingual-e5-small (built-in)",
      gating: true,
      passageCount: 1199,
      processingSeconds: 60,
      questions: [],
      summary: {
        hybrid: summary(13, 19, [5, 4]),
        [GATING_MODE]: summary(16, 20, [6, 5]),
        // Under the bar in English: reported, and the run still passes.
        [HYBRID_RERANK_MODE]: summary(15, 20, [4, 3]),
      },
      rerankers: [reranker(GATING_MODE), reranker(HYBRID_RERANK_MODE)],
      candidates: [
        {
          search: "keyword-60",
          searches: 75,
          mean: 57.5,
          min: 9,
          max: 60,
          reached: reached(39, 13),
        },
        { search: "hybrid", searches: 75, mean: 16, min: 12, max: 20, reached: reached(38, 13) },
      ],
    } satisfies RetrievalRun;
    const failures = retrievalFailures(run.summary[GATING_MODE]);
    const report = {
      result: failures.length === 0 ? "pass" : "fail",
      failures,
      run: {
        startedAt: "2026-10-10T08:00:00.000Z",
        seconds: 60,
        commit: "abc1234",
        node: "v25",
        platform: "darwin arm64",
        cpu: "M2",
      },
      evaluationSet: {
        source: "eval/retrieval/questions.json",
        hitRule: "",
        questions: { gating: { en: 20, zh: 20 }, crossLingual: 10, paraphrase: 15 },
      },
      documents: [],
      retrieval: { topK: 5, gatingMode: GATING_MODE, runs: [run] },
      citations: { skipped: "no chat model" },
    } satisfies EvalReport;

    expect(report.result).toBe("pass");
    const lines = terminalSummary(report, "/repo/eval/results/x", "/repo").split("\n");
    expect(lines.find((line) => line.trim().startsWith("hybrid + "))).toMatch(
      /English 15\/20 .*; paraphrase 7\/15$/,
    );
    expect(lines.some((line) => line.includes("keyword top 60 + rerank search"))).toBe(true);
    expect(lines.some((line) => line.includes("hybrid + rerank search"))).toBe(true);

    const results = await mkdtemp(join(tmpdir(), "retrieval-report-"));
    try {
      const markdown = (
        await readFile(join(await writeReports(report, results, "/"), "report.md"), "utf8")
      ).split("\n");
      const model = BUILT_IN_RERANKING_MODEL.name;
      expect(markdown).toContain(
        "- Evaluation set: `eval/retrieval/questions.json`, 20 English and 20 Chinese gating Questions, 10 cross-lingual, 15 paraphrase",
      );
      expect(markdown).toContain(
        `| multilingual-e5-small (built-in) | **keyword top 60 + ${model} (gating)** | **16/20** | **20/20** | **36/40** | 2/10 | 7/10 | 11/15 (6 + 5) |`,
      );
      expect(markdown).toContain(
        `| multilingual-e5-small (built-in) | hybrid + ${model} | 15/20 | 20/20 | 35/40 | 2/10 | 7/10 | 7/15 (4 + 3) |`,
      );
      expect(markdown).toContain(
        "Candidates per keyword top 60 + rerank search (keyword search's top 60, no vector search): 57.5 on average, from 9 to 60, over 75 searches; the expected Passage among them for 39/40 gating and 13/15 paraphrase Questions.",
      );
      expect(markdown).toContain(
        "Candidates per hybrid + rerank search (keyword search's top 10 and vector search's top 10, each Passage once): 16.0 on average, from 12 to 20, over 75 searches; the expected Passage among them for 38/40 gating and 13/15 paraphrase Questions.",
      );
      expect(markdown.filter((line) => line.includes("| Apache-2.0 | 136 MB | 3 ms"))).toEqual([
        `| keyword top 60 + ${model} | Apache-2.0 | 136 MB | 3 ms | 700 ms | 690 ms | 900 ms | 950 ms | 1.5 s |`,
        `| hybrid + ${model} | Apache-2.0 | 136 MB | 3 ms | 700 ms | 690 ms | 900 ms | 950 ms | 1.5 s |`,
      ]);
      // The other searches didn't run: no section for them.
      expect(markdown.some((line) => line.includes("Other ways to find the candidates"))).toBe(
        false,
      );
    } finally {
      await rm(results, { recursive: true, force: true });
    }
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

describe("Paraphrase Questions", () => {
  /** Reads an evaluation set of one Document and these Questions, written to a temporary folder. */
  async function setOf(questions: unknown[]) {
    const root = await mkdtemp(join(tmpdir(), "evaluation-set-"));
    try {
      await writeFile(join(root, "tides.pdf"), "");
      await writeFile(
        join(root, "set.json"),
        JSON.stringify({ documents: { tides: "tides.pdf" }, questions }),
      );
      return loadEvaluationSet(root, "set.json");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }
  const asked = (id: string, extra: Record<string, unknown> = {}) => ({
    id,
    language: "en",
    question: "When is the sea highest?",
    expected: { document: "tides", pages: [1, 1], quote: "Spring tides" },
    ...extra,
  });

  test("are marked with paraphrase: true, and only true is kept", async () => {
    const set = await setOf([
      asked("en-01"),
      asked("para-en-01", { paraphrase: true }),
      asked("en-02", { paraphrase: false }),
    ]);

    expect(set.questions.map((question) => [question.id, question.paraphrase])).toEqual([
      ["en-01", undefined],
      ["para-en-01", true],
      ["en-02", undefined],
    ]);
    expect(set.questions.map(isGating)).toEqual([true, false, true]);
  });

  test("a flag that isn't true or false, or a cross-lingual paraphrase, is refused", async () => {
    await expect(setOf([asked("para-en-01", { paraphrase: "yes" })])).rejects.toThrow(
      "para-en-01: paraphrase must be true or false.",
    );
    await expect(
      setOf([
        asked("para-xl-01", {
          paraphrase: true,
          crossLingual: true,
          translatedQuery: "大潮什么时候出现？",
        }),
      ]),
    ).rejects.toThrow(/para-xl-01: a paraphrase question isn't cross-lingual/);
  });

  test("today's set has 8 English and 7 Chinese ones, and its gating Questions are unchanged", () => {
    const set = loadEvaluationSet(fileURLToPath(new URL("../..", import.meta.url)));
    const paraphrase = set.questions.filter((question) => question.paraphrase);
    const gating = set.questions.filter(isGating);

    expect(paraphrase.filter((question) => question.language === "en")).toHaveLength(8);
    expect(paraphrase.filter((question) => question.language === "zh")).toHaveLength(7);
    expect(paraphrase.every((question) => !question.crossLingual)).toBe(true);
    expect(gating.filter((question) => question.language === "en")).toHaveLength(20);
    expect(gating.filter((question) => question.language === "zh")).toHaveLength(20);
    expect(set.questions.filter((question) => question.crossLingual)).toHaveLength(10);
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
