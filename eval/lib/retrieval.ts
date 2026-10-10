/**
 * Retrieval: each Question searched through the core's `searchPassages`, in
 * each search mode, scored with ADR-0009's hit rule. A Question is a hit when
 * one of the top 5 Passages (a) belongs to the expected Document, (b) covers
 * the expected pages, and (c) contains the expected quote, matched as the
 * Citation check matches quotes (both normalised by the shared normaliser).
 *
 * Keyword top 60 + rerank, the gating mode: keyword search's top 60,
 * reordered by a reranking model (see ./rerank), as the search Tool does by
 * default, with embeddings off (ADR-0009, 2026-10-10). The built-in reranking
 * model's is the gate; the other candidates' are reported only.
 *
 * Hybrid + rerank: what the search Tool hands a reranker while the User has
 * embeddings on (keyword search's top 10 and vector search's top 10, each
 * Passage once, built with the Tool's own `topsOfEach`), reordered by the
 * same models. Reported next to the gate with the plain search modes, never
 * gating: it measures what embeddings add.
 *
 * Also reported:
 * - A translated second query: a cross-lingual Question's own search, and a
 *   second one with the Question translated into its Document's language, as
 *   an Answer is told to search again. A hit in either top 5 counts: the
 *   model sees both searches' Passages. The translation is written by hand,
 *   so this is the most the approach can bring.
 * - Paraphrase Questions: asked in words that avoid their passage's own, as a
 *   group of their own, out of the gating counts.
 * - Other ways to find what the reranker sees, with embeddings off (keyword
 *   search's top 20 or 40, feedback terms, a chat model's rewrites,
 *   small-to-big, document first; see ./searches), each reranked by the
 *   built-in model: never gating, to choose what the search Tool does next.
 * - For each Question, where plain keyword search ranks its expected Passage,
 *   up to 200: whether more of keyword search's candidates could reach it.
 * - For each reranked mode, whether its candidates held the expected Passage
 *   at all: what any reranker could have found.
 */
import {
  BUILT_IN_RERANKING_MODEL,
  type Core,
  type PassageSearchResult,
  type RerankingModelDefinition,
  type SearchMode,
} from "../../src/core";
import { SEARCH_TOOL_PARAMETERS, topsOfEach } from "../../src/core/documents/searchTool";
import { findQuote } from "../../src/shared/quoteMatch";
import {
  type EvalLanguage,
  type EvalQuestion,
  type ExpectedPassage,
  isGating,
} from "./evaluationSet";
import { type Latency, latencyOf, type OpenReranker, type RerankerInfo } from "./rerank";
import { MOST_QUERIES, type QueryModelCost } from "./rewrites";
import { CANDIDATES, type Candidates, DOCUMENT_FIRST, FEEDBACK, keywordTop } from "./searches";
import {
  AGGREGATIONS,
  type Aggregation,
  SUB_CHUNK_TOKENS,
  type SubChunkIndex,
  type SubChunkStats,
} from "./subChunks";

export const TOP_K = 5;

/** Ranks are looked for this deep, to show near misses; only the top 5 count as hits. */
export const RANK_DEPTH = 20;

/** Plain keyword search's rank of the expected Passage is looked for this deep: the most `searchPassages` returns. */
export const KEYWORD_RANK_DEPTH = 200;

/** A reranker sees this many of keyword search's best Passages, and of vector search's: the search Tool's. */
export const RERANK_PER_LIST = SEARCH_TOOL_PARAMETERS.rerankPerList;

/** The gate sees this many of keyword search's best: the search Tool's, with embeddings off. */
export const KEYWORD_RERANK_DEPTH = SEARCH_TOOL_PARAMETERS.keywordRerankCandidates;

export const SEARCH_MODES: readonly SearchMode[] = ["hybrid", "keyword", "vector"];

/** Hybrid search: what the search Tool runs with embeddings on before it reranks, and the second query's search. */
export const HYBRID: SearchMode = "hybrid";

/**
 * What a reranked mode reranks: keyword search's top 60 alone ("keyword-60"),
 * what the search Tool hands a reranker by default, or keyword search's top
 * 10 and vector search's top 10 ("hybrid"), as it does with embeddings on.
 * The others are other ways to find candidates with embeddings off (see
 * ./searches), reported only: keyword search's top 20 ("keyword", the gate
 * before keyword search's top 60) and top 40 among them.
 */
export type RerankedSearch =
  | "hybrid"
  | "keyword"
  | "keyword-40"
  | "keyword-60"
  | "feedback"
  | "rewrites"
  | "sub-questions"
  | "small-to-big"
  | "document-first";

/** The gate's search: keyword search's top 60, what the search Tool hands its reranker with embeddings off. */
export const GATING_SEARCH: RerankedSearch = "keyword-60";

/** The reranked searches each reranking model runs, in the order the report lists them: the gate's first. */
export const RERANKED_SEARCHES: readonly RerankedSearch[] = [GATING_SEARCH, "hybrid"];

/**
 * The other ways to find candidates, run with the built-in reranking model on
 * the gating set only, after its two (see ./searches). Never gating.
 */
export const OTHER_SEARCHES: readonly RerankedSearch[] = [
  "keyword",
  "keyword-40",
  "feedback",
  "rewrites",
  "sub-questions",
  "small-to-big",
  "document-first",
];

/** How deep keyword search's candidates go in the modes that are keyword search's alone. */
export const KEYWORD_DEPTHS: Partial<Record<RerankedSearch, number>> = {
  keyword: 2 * RERANK_PER_LIST,
  "keyword-40": 40,
  "keyword-60": KEYWORD_RERANK_DEPTH,
};

/** Each reranked search, as its mode's label starts. */
export const SEARCH_LABELS: Record<RerankedSearch, string> = {
  keyword: "keyword top 20",
  hybrid: "hybrid",
  "keyword-40": "keyword top 40",
  "keyword-60": "keyword top 60",
  feedback: "keyword + feedback terms",
  rewrites: "keyword + rewrites",
  "sub-questions": "keyword + sub-questions",
  "small-to-big": "small-to-big",
  "document-first": "document first",
};

/** What each reranked search hands a reranker, for the report and the log. */
export const SEARCH_DESCRIPTIONS: Record<RerankedSearch, string> = {
  keyword: `keyword search's top ${KEYWORD_DEPTHS.keyword}, no vector search, the gate before keyword search's top ${KEYWORD_RERANK_DEPTH}`,
  hybrid: `keyword search's top ${RERANK_PER_LIST} and vector search's top ${RERANK_PER_LIST}, each Passage once`,
  "keyword-40": "keyword search's top 40, no vector search",
  "keyword-60": `keyword search's top ${KEYWORD_RERANK_DEPTH}, no vector search`,
  feedback: `keyword search's top ${CANDIDATES}, fused with a second search for the Question's words and ${FEEDBACK.terms} feedback terms from its top ${FEEDBACK.passages}, the best ${CANDIDATES}`,
  rewrites: `keyword search's top ${CANDIDATES} for the Question and for each of a chat model's ${MOST_QUERIES.rewrites} rephrasings, fused, the best ${CANDIDATES}`,
  "sub-questions": `keyword search's top ${CANDIDATES} for the Question and for each one-hop question a chat model broke it into, if any, fused, the best ${CANDIDATES}`,
  "small-to-big": `Passages by their best sub-chunk (at most ${SUB_CHUNK_TOKENS} tokens) in BM25 over sub-chunks, the best ${CANDIDATES}`,
  "document-first": `the ${DOCUMENT_FIRST.documents} Documents that hold most of keyword search's top ${CANDIDATES}, each searched inside with its own term statistics, fused, the best ${CANDIDATES}`,
};

/**
 * A mode of the report: one of the core's search modes, or a reranked mode:
 * hybrid search's candidates ("rerank:") or another search's
 * ("keyword-rerank:", "feedback-rerank:" and so on) reranked by a candidate.
 */
export type RetrievalMode = SearchMode | `rerank:${string}` | `${RerankedSearch}-rerank:${string}`;

/** The reranked mode of a reranking model, over hybrid search's candidates unless another search is named. */
export const rerankMode = (
  definition: Pick<RerankingModelDefinition, "id">,
  search: RerankedSearch = "hybrid",
): RetrievalMode =>
  search === "hybrid" ? `rerank:${definition.id}` : `${search}-rerank:${definition.id}`;

/** What a mode reranks; null for the core's own search modes. */
export function rerankedSearchOf(mode: string): RerankedSearch | null {
  if (mode.startsWith("rerank:")) return "hybrid";
  return (
    (Object.keys(SEARCH_LABELS) as RerankedSearch[]).find(
      (search) => search !== "hybrid" && mode.startsWith(`${search}-rerank:`),
    ) ?? null
  );
}

/** A reranked mode as people read it: "hybrid + <model>", "keyword top 60 + <model>", "small-to-big + <model>". */
export const rerankedLabel = (search: RerankedSearch, modelName: string) =>
  `${SEARCH_LABELS[search]} + ${modelName}`;

/**
 * The gating mode: what the search Tool runs by default, keyword search's top
 * 60 reranked by the built-in reranking model, with embeddings off (ADR-0009,
 * 2026-10-10). Before, it was keyword search's top 20, and before that hybrid
 * search reranked (#31).
 */
export const GATING_MODE: RetrievalMode = rerankMode(BUILT_IN_RERANKING_MODEL, GATING_SEARCH);

/** The gating mode, as people read it. */
export const GATING_LABEL = rerankedLabel(GATING_SEARCH, BUILT_IN_RERANKING_MODEL.name);

/** Hybrid + rerank with the built-in reranking model, as with embeddings on: reported next to the gate, never gating. */
export const HYBRID_RERANK_MODE: RetrievalMode = rerankMode(BUILT_IN_RERANKING_MODEL);

/** Hybrid + rerank with the built-in reranking model, as people read it. */
export const HYBRID_RERANK_LABEL = rerankedLabel("hybrid", BUILT_IN_RERANKING_MODEL.name);

/** The v1 design's bar: 80% of the gating Questions overall and in each language (32 of 40, and 16 of 20 per language, with today's set). */
const RETRIEVAL_TARGET = { share: 0.8 } as const;

/** One retrieved Passage, against the expected one. */
export interface RetrievedPassage {
  documentName: string;
  pageFrom: number | null;
  pageTo: number | null;
  rightDocument: boolean;
  coversPages: boolean;
  hasQuote: boolean;
}

export interface ModeResult {
  /** A Passage in the top 5 is a hit. */
  hit: boolean;
  /** The rank of the first Passage that is a hit, from 1, within the top 20; null if none is. */
  rank: number | null;
  /** The top 5. */
  top: RetrievedPassage[];
}

export interface QuestionResult {
  id: string;
  language: EvalLanguage;
  crossLingual: boolean;
  /** A paraphrase Question: reported apart, never gating. */
  paraphrase?: boolean;
  question: string;
  modes: Partial<Record<RetrievalMode, ModeResult>>;
  /** The hand-written translation searched as a second query, if the Question has one. */
  translatedQuery?: string;
  /** That second query's results: hybrid, and each reranked mode. */
  translated?: Partial<Record<RetrievalMode, ModeResult>>;
  /**
   * Plain keyword search's rank of the first Passage that meets the hit
   * rule, within its top 200; null if none does.
   */
  keywordRank?: number | null;
  /** What each reranked search handed a reranker for the Question (and its translation). */
  candidates?: Partial<Record<RerankedSearch, CandidateRecord>>;
  /** What a search looked for besides the Question: a model's queries, feedback terms, the Documents searched inside. */
  queries?: Partial<Record<RerankedSearch, string[]>>;
}

/** The candidates one reranked search handed a reranker for a Question. */
export interface CandidateRecord {
  /** How many, for the Question. */
  question: number;
  /** How many for its translated query, when that was searched too. */
  translated?: number;
  /** The rank among the Question's candidates, from 1, of the first Passage that meets the hit rule; null if none does. */
  reached: number | null;
}

export interface Tally {
  hits: number;
  total: number;
}

/**
 * How many candidates a reranker saw per search, from one reranked search,
 * and how often they held the expected Passage at all.
 */
export interface CandidateCounts {
  search: RerankedSearch;
  /** Searches reranked: the Questions, and the translated queries. */
  searches: number;
  mean: number;
  min: number;
  max: number;
  /** The Questions whose own candidates held a Passage that meets the hit rule: gating ones, and paraphrase ones. */
  reached: { gating: Tally; paraphrase: Tally };
}

/** Hits in each language, and in both. */
export interface LanguageTallies {
  en: Tally;
  zh: Tally;
  all: Tally;
}

/**
 * A mode's hits: per language and overall over the gating Questions, over
 * the cross-lingual ones, and over the paraphrase ones.
 */
export interface ModeSummary {
  en: Tally;
  zh: Tally;
  core: Tally;
  crossLingual: Tally;
  /**
   * The cross-lingual Questions with a translated query, searched twice: a
   * hit when either search's top 5 has one. Null for modes the translated
   * query isn't searched in (keyword, vector).
   */
  crossLingualTranslated: Tally | null;
  /** The paraphrase Questions, per language: out of `en`, `zh` and `core`, so the gating counts stay comparable. */
  paraphrase: LanguageTallies;
}

export interface RetrievalRun {
  /** "built-in" or the cloud model, e.g. "openai/text-embedding-3-small". */
  embedding: string;
  /** Only the built-in model gates. */
  gating: boolean;
  passageCount: number;
  processingSeconds: number;
  questions: QuestionResult[];
  summary: Partial<Record<RetrievalMode, ModeSummary>>;
  /** The reranking candidates of the reranked modes, one per mode, if any were given. */
  rerankers?: RerankerInfo[];
  /** How many Passages each reranked search handed a reranker, for those that ran: keyword + rerank's first. */
  candidates?: CandidateCounts[];
  /** The other ways to find candidates (`OTHER_SEARCHES`), when they ran. */
  otherSearches?: OtherSearchesReport;
}

/** What the report says about the other searches beyond their hits. */
export interface OtherSearchesReport {
  /** Why a search didn't run, such as no chat model given. */
  skipped: Partial<Record<RerankedSearch, string>>;
  /** Small-to-big's index, and how often each way of scoring Passages by their sub-chunks reached the expected one. */
  smallToBig?: SubChunkStats & {
    aggregations: { aggregation: Aggregation; reached: { gating: Tally; paraphrase: Tally } }[];
  };
  /** The chat model's calls for rewrites and sub-questions. */
  queryModel: QueryModelCost[];
}

export function checkPassage(
  passage: PassageSearchResult,
  expected: ExpectedPassage,
  expectedDocumentId: string,
): RetrievedPassage {
  const [first, last] = expected.pages;
  return {
    documentName: passage.documentName,
    pageFrom: passage.pageFrom,
    pageTo: passage.pageTo,
    rightDocument: passage.documentId === expectedDocumentId,
    coversPages:
      passage.pageFrom !== null &&
      passage.pageTo !== null &&
      passage.pageFrom <= first &&
      passage.pageTo >= last,
    hasQuote: findQuote(passage.text, expected.quote) !== null,
  };
}

export const isHit = (passage: RetrievedPassage) =>
  passage.rightDocument && passage.coversPages && passage.hasQuote;

/** The rank, from 1, of the first Passage in a list that meets the hit rule, however deep; null if none does. */
export function firstHit(
  found: readonly PassageSearchResult[],
  expected: ExpectedPassage,
  expectedDocumentId: string,
): number | null {
  const index = found.findIndex((passage) =>
    isHit(checkPassage(passage, expected, expectedDocumentId)),
  );
  return index >= 0 ? index + 1 : null;
}

/** Scores a ranked list: a hit in the top 5, and the first hit's rank in the top 20. */
export function scoreRanking(
  found: readonly PassageSearchResult[],
  expected: ExpectedPassage,
  expectedDocumentId: string,
): ModeResult {
  const checked = found
    .slice(0, RANK_DEPTH)
    .map((passage) => checkPassage(passage, expected, expectedDocumentId));
  const index = checked.findIndex(isHit);
  return {
    hit: index >= 0 && index < TOP_K,
    rank: index >= 0 ? index + 1 : null,
    top: checked.slice(0, TOP_K),
  };
}

function expectedIdOf(question: EvalQuestion, documentIds: ReadonlyMap<string, string>): string {
  const expectedId = documentIds.get(question.expected.document);
  if (!expectedId) throw new Error(`${question.id}: its Document wasn't added.`);
  return expectedId;
}

/** Hybrid search's top 20 for a query. */
const hybridTop = (core: Core, query: string) =>
  core.searchPassages(query, { mode: HYBRID, limit: RANK_DEPTH });

/**
 * What the search Tool hands its reranker for a query: keyword search's top
 * 10 and vector search's top 10, each Passage once, put together by the
 * Tool's own `topsOfEach`. The Tool orders them by their fused score first,
 * which only matters to a reranker for ties.
 */
export async function rerankCandidates(core: Core, query: string): Promise<PassageSearchResult[]> {
  const [keyword, vector] = await Promise.all(
    (["keyword", "vector"] as const).map((mode) =>
      core.searchPassages(query, { mode, limit: RERANK_PER_LIST }),
    ),
  );
  return topsOfEach([keyword ?? [], vector ?? []], RERANK_PER_LIST, (passage) => passage.passageId);
}

/** What the gate hands a reranker for a query: keyword search's top 60, with no vector search. */
export const keywordRerankCandidates = (
  core: Core,
  query: string,
): Promise<PassageSearchResult[]> => keywordTop(core, query, KEYWORD_RERANK_DEPTH);

/** Builds a reranked search's candidates for a query of a Question: its own, or its translation. */
export type CandidateSource = (query: string, question: EvalQuestion) => Promise<Candidates>;

/**
 * The candidates of the reranked searches that need nothing but the core:
 * hybrid + rerank's, and keyword search's top 60 (the gate's), 20 and 40.
 * Null for the others, whose candidates `runReranked` is given.
 */
export function coreSource(core: Core, search: RerankedSearch): CandidateSource | null {
  if (search === "hybrid") {
    return async (query) => ({ candidates: await rerankCandidates(core, query) });
  }
  const depth = KEYWORD_DEPTHS[search];
  if (depth === undefined) return null;
  return async (query) => ({ candidates: await keywordTop(core, query, depth) });
}

/**
 * How many candidates a reranked search had (hybrid's, unless another is
 * named), from the records `runReranked` kept, and how often they held the
 * expected Passage; null if it didn't run.
 */
export function candidateCounts(
  results: readonly QuestionResult[],
  search: RerankedSearch = "hybrid",
): CandidateCounts | null {
  const recorded = results.flatMap((result) => {
    const record = result.candidates?.[search];
    return record ? [{ result, record }] : [];
  });
  const counts = recorded.flatMap(({ record }) => [
    record.question,
    ...(record.translated === undefined ? [] : [record.translated]),
  ]);
  if (counts.length === 0) return null;
  const reached = (group: readonly { record: CandidateRecord }[]) => ({
    hits: group.filter(({ record }) => record.reached !== null).length,
    total: group.length,
  });
  return {
    search,
    searches: counts.length,
    mean: counts.reduce((sum, count) => sum + count, 0) / counts.length,
    min: Math.min(...counts),
    max: Math.max(...counts),
    reached: {
      gating: reached(recorded.filter(({ result }) => isGating(result))),
      paraphrase: reached(
        recorded.filter(({ result }) => result.paraphrase && !result.crossLingual),
      ),
    },
  };
}

/**
 * Searches each Question in each mode and scores the top 5 (and finds the
 * first hit in the top 20); a translated query is searched in hybrid mode too.
 * Keyword search's rank of the expected Passage is looked for in its top 200.
 */
export async function runRetrieval(
  core: Core,
  questions: readonly EvalQuestion[],
  documentIds: ReadonlyMap<string, string>,
  modes: readonly SearchMode[] = SEARCH_MODES,
): Promise<QuestionResult[]> {
  const results: QuestionResult[] = [];
  for (const question of questions) {
    const expectedId = expectedIdOf(question, documentIds);
    const result: QuestionResult = {
      id: question.id,
      language: question.language,
      crossLingual: question.crossLingual,
      ...(question.paraphrase && { paraphrase: true }),
      question: question.question,
      modes: {},
    };
    for (const mode of modes) {
      // A deeper search returns the same top 5: each list's top 50 is fused before the limit,
      // and keyword search's order doesn't depend on how many it returns.
      const deep = mode === "keyword";
      const found = await core.searchPassages(question.question, {
        mode,
        limit: deep ? KEYWORD_RANK_DEPTH : RANK_DEPTH,
      });
      result.modes[mode] = scoreRanking(found, question.expected, expectedId);
      if (deep) result.keywordRank = firstHit(found, question.expected, expectedId);
    }
    if (question.translatedQuery) {
      result.translatedQuery = question.translatedQuery;
      result.translated = {
        [HYBRID]: scoreRanking(
          await hybridTop(core, question.translatedQuery),
          question.expected,
          expectedId,
        ),
      };
    }
    results.push(result);
  }
  return results;
}

/**
 * Adds the reranker's mode to `results`: each Question's candidates (and its
 * translated query's, unless `translated` is false), reordered by the
 * candidate model, scored like the others. What is reranked follows from the
 * mode (see `rerankedSearchOf`): what the search Tool hands a reranker with
 * embeddings on (`rerankCandidates`) or off (keyword search's top 60,
 * `keywordRerankCandidates`), keyword search's top 20 or 40, or what
 * `source` gives. Records how many candidates
 * each search had and whether they held the expected Passage, and returns how
 * long finding them took per search.
 */
export async function runReranked(
  core: Core,
  questions: readonly EvalQuestion[],
  documentIds: ReadonlyMap<string, string>,
  reranker: OpenReranker,
  results: readonly QuestionResult[],
  options: { source?: CandidateSource; translated?: boolean } = {},
): Promise<Latency> {
  const mode = reranker.mode as RetrievalMode;
  const search = rerankedSearchOf(mode);
  if (!search) throw new Error(`${mode} isn't a reranked mode.`);
  const source = options.source ?? coreSource(core, search);
  if (!source) throw new Error(`${mode}: its candidates weren't given.`);
  const timings: number[] = [];
  for (const question of questions) {
    const result = results.find((each) => each.id === question.id);
    if (!result) throw new Error(`${question.id}: no results to add the reranked mode to.`);
    const expectedId = expectedIdOf(question, documentIds);
    const reranked = async (query: string) => {
      const started = performance.now();
      const { candidates, queries } = await source(query, question);
      timings.push(performance.now() - started);
      const ranking = scoreRanking(
        await reranker.rerank(query, candidates),
        question.expected,
        expectedId,
      );
      return {
        ranking,
        count: candidates.length,
        reached: firstHit(candidates, question.expected, expectedId),
        queries,
      };
    };
    const own = await reranked(question.question);
    result.modes[mode] = own.ranking;
    const record: CandidateRecord = { question: own.count, reached: own.reached };
    if (own.queries) result.queries = { ...result.queries, [search]: own.queries };
    if (options.translated !== false && question.translatedQuery && result.translated) {
      const translated = await reranked(question.translatedQuery);
      result.translated[mode] = translated.ranking;
      record.translated = translated.count;
    }
    result.candidates = { ...result.candidates, [search]: record };
  }
  return latencyOf(timings);
}

/**
 * How often each way of scoring Passages by their sub-chunks puts the
 * expected Passage in small-to-big's 20 candidates, over the gating Questions
 * and the paraphrase ones: before any reranking, which runs on "best" only.
 */
export function compareAggregations(
  index: SubChunkIndex,
  questions: readonly EvalQuestion[],
  documentIds: ReadonlyMap<string, string>,
): { aggregation: Aggregation; reached: { gating: Tally; paraphrase: Tally } }[] {
  return AGGREGATIONS.map((aggregation) => {
    const reached = (group: readonly EvalQuestion[]): Tally => ({
      hits: group.filter(
        (question) =>
          firstHit(
            index.search(question.question, CANDIDATES, aggregation),
            question.expected,
            expectedIdOf(question, documentIds),
          ) !== null,
      ).length,
      total: group.length,
    });
    return {
      aggregation,
      reached: {
        gating: reached(questions.filter(isGating)),
        paraphrase: reached(
          questions.filter((question) => question.paraphrase && !question.crossLingual),
        ),
      },
    };
  });
}

const tally = (results: readonly QuestionResult[], mode: RetrievalMode): Tally => ({
  hits: results.filter((result) => result.modes[mode]?.hit).length,
  total: results.length,
});

const byLanguage = (results: readonly QuestionResult[], mode: RetrievalMode): LanguageTallies => ({
  en: tally(
    results.filter((result) => result.language === "en"),
    mode,
  ),
  zh: tally(
    results.filter((result) => result.language === "zh"),
    mode,
  ),
  all: tally(results, mode),
});

/** The modes the results have, the core's search modes first, in the order they were run. */
export function modesOf(results: readonly QuestionResult[]): RetrievalMode[] {
  const modes = new Set<RetrievalMode>();
  for (const result of results) {
    for (const mode of Object.keys(result.modes) as RetrievalMode[]) modes.add(mode);
  }
  return [...modes];
}

export function summarise(
  results: readonly QuestionResult[],
  modes: readonly RetrievalMode[] = modesOf(results),
): Partial<Record<RetrievalMode, ModeSummary>> {
  const core = results.filter(isGating);
  const crossLingual = results.filter((result) => result.crossLingual);
  const paraphrase = results.filter((result) => result.paraphrase && !result.crossLingual);
  const withTranslation = crossLingual.filter((result) => result.translated);
  const summary: Partial<Record<RetrievalMode, ModeSummary>> = {};
  for (const mode of modes) {
    const translatedSearched = withTranslation.some((result) => result.translated?.[mode]);
    const gating = byLanguage(core, mode);
    summary[mode] = {
      en: gating.en,
      zh: gating.zh,
      core: gating.all,
      crossLingual: tally(crossLingual, mode),
      paraphrase: byLanguage(paraphrase, mode),
      crossLingualTranslated: translatedSearched
        ? {
            hits: withTranslation.filter(
              (result) => result.modes[mode]?.hit || result.translated?.[mode]?.hit,
            ).length,
            total: withTranslation.length,
          }
        : null,
    };
  }
  return summary;
}

const meets = ({ hits, total }: Tally) =>
  total > 0 && hits >= Math.ceil(total * RETRIEVAL_TARGET.share);

/** Why the gating mode misses the bar; empty when it passes. */
export function retrievalFailures(summary: ModeSummary | undefined): string[] {
  if (!summary) return [`No ${GATING_LABEL} results.`];
  const failures: string[] = [];
  const need = ({ total }: Tally) => Math.ceil(total * RETRIEVAL_TARGET.share);
  for (const [label, count] of [
    ["overall", summary.core],
    ["English", summary.en],
    ["Chinese", summary.zh],
  ] as const) {
    if (!meets(count)) {
      failures.push(
        `Retrieval (${GATING_LABEL}, the built-in reranking model), ${label}: ${count.hits} of ${count.total}, needs ${need(count)}.`,
      );
    }
  }
  return failures;
}
