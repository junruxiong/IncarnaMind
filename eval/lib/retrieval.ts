/**
 * Retrieval: each Question searched through the core's `searchPassages`, in
 * each search mode, scored with ADR-0009's hit rule. A Question is a hit when
 * one of the top 5 Passages (a) belongs to the expected Document, (b) covers
 * the expected pages, and (c) contains the expected quote, matched as the
 * Citation check matches quotes (both normalised by the shared normaliser).
 *
 * Two more measurements, reported only:
 * - Reranked modes: the candidates the search Tool hands a reranker (keyword
 *   search's top 10 and vector search's top 10, each Passage once, built
 *   with the Tool's own `topsOfEach`), reordered by a built-in reranking
 *   candidate (see ./rerank), as the search Tool does with reranking on.
 * - A translated second query: a cross-lingual Question's own search, and a
 *   second one with the Question translated into its Document's language, as
 *   an Answer is told to search again. A hit in either top 5 counts: the
 *   model sees both searches' Passages. The translation is written by hand,
 *   so this is the most the approach can bring.
 */
import type { Core, PassageSearchResult, SearchMode } from "../../src/core";
import { SEARCH_TOOL_PARAMETERS, topsOfEach } from "../../src/core/documents/searchTool";
import { findQuote } from "../../src/shared/quoteMatch";
import type { EvalLanguage, EvalQuestion, ExpectedPassage } from "./evaluationSet";
import type { OpenReranker, RerankerInfo } from "./rerank";

export const TOP_K = 5;

/** Ranks are looked for this deep, to show near misses; only the top 5 count as hits. */
export const RANK_DEPTH = 20;

/** A reranker sees this many of keyword search's best Passages, and of vector search's: the search Tool's. */
export const RERANK_PER_LIST = SEARCH_TOOL_PARAMETERS.rerankPerList;

export const SEARCH_MODES: readonly SearchMode[] = ["hybrid", "keyword", "vector"];

/** The gating mode: what the search Tool runs. */
export const GATING_MODE: SearchMode = "hybrid";

/** A mode of the report: one of the core's search modes, or hybrid search reranked by a candidate. */
export type RetrievalMode = SearchMode | `rerank:${string}`;

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
  question: string;
  modes: Partial<Record<RetrievalMode, ModeResult>>;
  /** The hand-written translation searched as a second query, if the Question has one. */
  translatedQuery?: string;
  /** That second query's results: hybrid, and each reranked mode. */
  translated?: Partial<Record<RetrievalMode, ModeResult>>;
  /** How many candidates a reranker saw for the Question, and for its translation, when reranked modes ran. */
  rerankCandidates?: { question: number; translated?: number };
}

/** How many candidates a reranker saw per search: keyword search's top 10 and vector search's, each Passage once. */
export interface CandidateCounts {
  /** How many of each list's best: 10. */
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

/** A mode's hits: per language and overall over the gating Questions, and over the cross-lingual ones. */
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
  /** The reranking candidates of the reranked modes, if any were given. */
  rerankers?: RerankerInfo[];
  /** How many Passages they reranked per search, if any were given. */
  rerankCandidates?: CandidateCounts;
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
  core.searchPassages(query, { mode: GATING_MODE, limit: RANK_DEPTH });

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

/** How many candidates the reranked searches had, from the counts `runReranked` recorded; null if none ran. */
export function candidateCounts(results: readonly QuestionResult[]): CandidateCounts | null {
  const counts = results.flatMap((result) =>
    result.rerankCandidates
      ? [
          result.rerankCandidates.question,
          ...(result.rerankCandidates.translated === undefined
            ? []
            : [result.rerankCandidates.translated]),
        ]
      : [],
  );
  if (counts.length === 0) return null;
  return {
    perList: RERANK_PER_LIST,
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
        [GATING_MODE]: scoreRanking(
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
 * Adds a reranked mode to `results`: each Question's reranking candidates
 * (and its translated query's; see `rerankCandidates`), reordered by the
 * candidate model, scored like the others. Records how many candidates each
 * search had.
 */
export async function runReranked(
  core: Core,
  questions: readonly EvalQuestion[],
  documentIds: ReadonlyMap<string, string>,
  reranker: OpenReranker,
  results: readonly QuestionResult[],
): Promise<void> {
  const mode = reranker.mode as RetrievalMode;
  for (const question of questions) {
    const result = results.find((each) => each.id === question.id);
    if (!result) throw new Error(`${question.id}: no results to add the reranked mode to.`);
    const expectedId = expectedIdOf(question, documentIds);
    const reranked = async (query: string) => {
      const candidates = await rerankCandidates(core, query);
      const ranking = scoreRanking(
        await reranker.rerank(query, candidates),
        question.expected,
        expectedId,
      );
      return { ranking, count: candidates.length };
    };
    const own = await reranked(question.question);
    result.modes[mode] = own.ranking;
    result.rerankCandidates = { question: own.count };
    if (question.translatedQuery && result.translated) {
      const translated = await reranked(question.translatedQuery);
      result.translated[mode] = translated.ranking;
      result.rerankCandidates.translated = translated.count;
    }
  }
}

const tally = (results: readonly QuestionResult[], mode: RetrievalMode): Tally => ({
  hits: results.filter((result) => result.modes[mode]?.hit).length,
  total: results.length,
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
  const core = results.filter((result) => !result.crossLingual);
  const crossLingual = results.filter((result) => result.crossLingual);
  const withTranslation = crossLingual.filter((result) => result.translated);
  const summary: Partial<Record<RetrievalMode, ModeSummary>> = {};
  for (const mode of modes) {
    const translatedSearched = withTranslation.some((result) => result.translated?.[mode]);
    summary[mode] = {
      en: tally(
        core.filter((result) => result.language === "en"),
        mode,
      ),
      zh: tally(
        core.filter((result) => result.language === "zh"),
        mode,
      ),
      core: tally(core, mode),
      crossLingual: tally(crossLingual, mode),
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
  if (!summary) return [`No ${GATING_MODE} results.`];
  const failures: string[] = [];
  const need = ({ total }: Tally) => Math.ceil(total * RETRIEVAL_TARGET.share);
  for (const [label, count] of [
    ["overall", summary.core],
    ["English", summary.en],
    ["Chinese", summary.zh],
  ] as const) {
    if (!meets(count)) {
      failures.push(
        `Retrieval (${GATING_MODE}, built-in model), ${label}: ${count.hits} of ${count.total}, needs ${need(count)}.`,
      );
    }
  }
  return failures;
}
