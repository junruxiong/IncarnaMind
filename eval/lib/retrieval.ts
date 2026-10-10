/**
 * Retrieval: each Question searched through the core's `searchPassages`, in
 * each search mode, scored with ADR-0009's hit rule. A Question is a hit when
 * one of the top 5 Passages (a) belongs to the expected Document, (b) covers
 * the expected pages, and (c) contains the expected quote, matched as the
 * Citation check matches quotes (both normalised by the shared normaliser).
 *
 * Keyword + rerank, the gating mode: keyword search's top 20, reordered by a
 * reranking model (see ./rerank), as the search Tool does by default, with
 * embeddings off (ADR-0009, 2026-10-10). The built-in reranking model's is
 * the gate; the other candidates' are reported only.
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
import type { OpenReranker, RerankerInfo } from "./rerank";

export const TOP_K = 5;

/** Ranks are looked for this deep, to show near misses; only the top 5 count as hits. */
export const RANK_DEPTH = 20;

/** A reranker sees this many of keyword search's best Passages, and of vector search's: the search Tool's. */
export const RERANK_PER_LIST = SEARCH_TOOL_PARAMETERS.rerankPerList;

/** Keyword + rerank sees this many of keyword search's best: the search Tool's, with embeddings off. */
export const KEYWORD_RERANK_DEPTH = 2 * RERANK_PER_LIST;

export const SEARCH_MODES: readonly SearchMode[] = ["hybrid", "keyword", "vector"];

/** Hybrid search: what the search Tool runs with embeddings on before it reranks, and the second query's search. */
export const HYBRID: SearchMode = "hybrid";

/**
 * What a reranked mode reranks: keyword search's top 20 alone ("keyword"),
 * what the search Tool hands a reranker by default, or keyword search's top
 * 10 and vector search's top 10 ("hybrid"), as it does with embeddings on.
 */
export type RerankedSearch = "hybrid" | "keyword";

/** The reranked searches, in the order the report lists them for each reranking model: the gate's first. */
export const RERANKED_SEARCHES: readonly RerankedSearch[] = ["keyword", "hybrid"];

/**
 * A mode of the report: one of the core's search modes, or a reranked mode:
 * hybrid search's candidates ("rerank:") or keyword search's alone
 * ("keyword-rerank:") reranked by a candidate.
 */
export type RetrievalMode = SearchMode | `rerank:${string}` | `keyword-rerank:${string}`;

/** The reranked mode of a reranking model, over hybrid search's candidates unless keyword search's are named. */
export const rerankMode = (
  definition: Pick<RerankingModelDefinition, "id">,
  search: RerankedSearch = "hybrid",
): RetrievalMode =>
  search === "hybrid" ? `rerank:${definition.id}` : `keyword-rerank:${definition.id}`;

/** What a mode reranks; null for the core's own search modes. */
export function rerankedSearchOf(mode: string): RerankedSearch | null {
  if (mode.startsWith("rerank:")) return "hybrid";
  if (mode.startsWith("keyword-rerank:")) return "keyword";
  return null;
}

/** A reranked mode as people read it: "hybrid + <model>", or "keyword + <model>". */
export const rerankedLabel = (search: RerankedSearch, modelName: string) =>
  `${search} + ${modelName}`;

/**
 * The gating mode: what the search Tool runs by default, keyword search's top
 * 20 reranked by the built-in reranking model, with embeddings off (ADR-0009,
 * 2026-10-10). Before, the gate was hybrid search reranked (#31).
 */
export const GATING_MODE: RetrievalMode = rerankMode(BUILT_IN_RERANKING_MODEL, "keyword");

/** The gating mode, as people read it. */
export const GATING_LABEL = rerankedLabel("keyword", BUILT_IN_RERANKING_MODEL.name);

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
  /** How many candidates a reranker saw for the Question, and for its translation, when hybrid's reranked modes ran. */
  rerankCandidates?: { question: number; translated?: number };
  /** The same for keyword + rerank: keyword search's top 20, or fewer when it found fewer. */
  keywordRerankCandidates?: { question: number; translated?: number };
}

/**
 * How many candidates a reranker saw per search: keyword search's top 10 and
 * vector search's, each Passage once, or keyword search's top 20 alone.
 */
export interface CandidateCounts {
  /** What was reranked. Hybrid's when left out, as in reports from before keyword + rerank. */
  search?: RerankedSearch;
  /** How many of each list's best: 10 of each for hybrid's, 20 of keyword search's alone. */
  perList: number;
  /** Searches reranked: the Questions, and the translated queries. */
  searches: number;
  mean: number;
  min: number;
  max: number;
}

export interface Tally {
  hits: number;
  total: number;
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
  /** How many Passages hybrid's reranked modes reranked per search, if any ran. */
  rerankCandidates?: CandidateCounts;
  /** How many Passages keyword + rerank reranked per search, if it ran. */
  keywordRerankCandidates?: CandidateCounts;
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

/** What keyword + rerank hands a reranker for a query: keyword search's top 20, with no vector search. */
export const keywordRerankCandidates = (
  core: Core,
  query: string,
): Promise<PassageSearchResult[]> =>
  core.searchPassages(query, { mode: "keyword", limit: KEYWORD_RERANK_DEPTH });

/** Each reranked search's candidates, and where a Question's result records how many there were. */
const RERANKED: Record<
  RerankedSearch,
  {
    candidates: (core: Core, query: string) => Promise<PassageSearchResult[]>;
    perList: number;
    counted: "rerankCandidates" | "keywordRerankCandidates";
  }
> = {
  hybrid: { candidates: rerankCandidates, perList: RERANK_PER_LIST, counted: "rerankCandidates" },
  keyword: {
    candidates: keywordRerankCandidates,
    perList: KEYWORD_RERANK_DEPTH,
    counted: "keywordRerankCandidates",
  },
};

/**
 * How many candidates the reranked searches had (hybrid's, unless keyword
 * search's are named), from the counts `runReranked` recorded; null if none ran.
 */
export function candidateCounts(
  results: readonly QuestionResult[],
  search: RerankedSearch = "hybrid",
): CandidateCounts | null {
  const { counted, perList } = RERANKED[search];
  const counts = results.flatMap((result) => {
    const recorded = result[counted];
    if (!recorded) return [];
    return [recorded.question, ...(recorded.translated === undefined ? [] : [recorded.translated])];
  });
  if (counts.length === 0) return null;
  return {
    search,
    perList,
    searches: counts.length,
    mean: counts.reduce((sum, count) => sum + count, 0) / counts.length,
    min: Math.min(...counts),
    max: Math.max(...counts),
  };
}

/**
 * Searches each Question in each mode and scores the top 5 (and finds the
 * first hit in the top 20); a translated query is searched in hybrid mode too.
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
      // A deeper search returns the same top 5: each list's top 50 is fused before the limit.
      const found = await core.searchPassages(question.question, { mode, limit: RANK_DEPTH });
      result.modes[mode] = scoreRanking(found, question.expected, expectedId);
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
 * translated query's), reordered by the candidate model, scored like the
 * others. What is reranked follows from the mode (see `rerankedSearchOf`):
 * what the search Tool hands a reranker (`rerankCandidates`), or keyword
 * search's top 20 (`keywordRerankCandidates`). Records how many candidates
 * each search had.
 */
export async function runReranked(
  core: Core,
  questions: readonly EvalQuestion[],
  documentIds: ReadonlyMap<string, string>,
  reranker: OpenReranker,
  results: readonly QuestionResult[],
): Promise<void> {
  const mode = reranker.mode as RetrievalMode;
  const search = rerankedSearchOf(mode);
  if (!search) throw new Error(`${mode} isn't a reranked mode.`);
  const { candidates: candidatesOf, counted } = RERANKED[search];
  for (const question of questions) {
    const result = results.find((each) => each.id === question.id);
    if (!result) throw new Error(`${question.id}: no results to add the reranked mode to.`);
    const expectedId = expectedIdOf(question, documentIds);
    const reranked = async (query: string) => {
      const candidates = await candidatesOf(core, query);
      const ranking = scoreRanking(
        await reranker.rerank(query, candidates),
        question.expected,
        expectedId,
      );
      return { ranking, count: candidates.length };
    };
    const own = await reranked(question.question);
    result.modes[mode] = own.ranking;
    const counts: { question: number; translated?: number } = { question: own.count };
    if (question.translatedQuery && result.translated) {
      const translated = await reranked(question.translatedQuery);
      result.translated[mode] = translated.ranking;
      counts.translated = translated.count;
    }
    result[counted] = counts;
  }
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
