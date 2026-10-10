/**
 * Retrieval on the hard tier: each answerable Question searched in every mode
 * the evaluation has (keyword, vector and hybrid search; hybrid search's
 * candidates and keyword search's top 20 reranked by the built-in reranking
 * model), scored with the gating set's hit rule for each Passage its answer
 * needs (eval/lib/retrieval). A Question is a hit when every Passage it needs
 * is in the top 5; a cross-lingual Question is searched again with its
 * translation, as an Answer is told to, and a Passage found by either search
 * counts. Unanswerable Questions aren't searched for: there is nothing to
 * find, and the Citation part scores them.
 */
import type { Core, PassageSearchResult, SearchMode } from "../../../src/core";
import type { OpenReranker } from "../../lib/rerank";
import {
  checkPassage,
  GATING_MODE,
  HYBRID_RERANK_MODE,
  isHit,
  keywordRerankCandidates,
  RANK_DEPTH,
  type RetrievalMode,
  rerankCandidates,
  rerankedSearchOf,
  TOP_K,
} from "../../lib/retrieval";
import type { Domain, HardLanguage } from "./manifest";
import { DIFFICULTIES, type Difficulty, type HardExpected, type HardQuestion } from "./questions";

/**
 * The modes, in the order the report lists them: keyword search, reranked as
 * the search Tool does by default (the gating set's gate), then hybrid
 * search, reranked as with embeddings on, and vector search.
 */
export const HARD_MODES: readonly RetrievalMode[] = [
  "keyword",
  GATING_MODE,
  "hybrid",
  HYBRID_RERANK_MODE,
  "vector",
];

const SEARCH_MODES: readonly SearchMode[] = ["keyword", "vector", "hybrid"];

/** One mode's result for one Question. */
export interface HardModeResult {
  /** Every Passage the answer needs is in the top 5 (of either search, for a cross-lingual Question). */
  hit: boolean;
  /** Some of them are: differs from `hit` only for Questions that need two or more. */
  partial: boolean;
  /** For each expected Passage, the rank of the first Passage that meets the hit rule (from 1, within the top 20), or null. */
  ranks: (number | null)[];
  /** The same for the translated query, when the Question has one. */
  translatedRanks?: (number | null)[];
}

export interface HardQuestionResult {
  id: string;
  language: HardLanguage;
  domain: Domain;
  difficulty: Difficulty;
  question: string;
  modes: Partial<Record<RetrievalMode, HardModeResult>>;
}

/** For each expected Passage, the rank of the first retrieved Passage that is it, within the top 20. */
export function ranksOf(
  found: readonly PassageSearchResult[],
  expected: readonly HardExpected[],
  documentIds: ReadonlyMap<string, string>,
): (number | null)[] {
  return expected.map((each) => {
    const id = documentIds.get(each.document) ?? "";
    const index = found
      .slice(0, RANK_DEPTH)
      .findIndex((passage) => isHit(checkPassage(passage, each, id)));
    return index >= 0 ? index + 1 : null;
  });
}

const inTop = (rank: number | null | undefined) =>
  rank !== null && rank !== undefined && rank <= TOP_K;

/** A mode's result from the ranks of the Question's own search and, if any, its translation's. */
export function modeResult(
  ranks: (number | null)[],
  translatedRanks?: (number | null)[],
): HardModeResult {
  const found = ranks.map((rank, index) => inTop(rank) || inTop(translatedRanks?.[index]));
  return {
    hit: found.length > 0 && found.every(Boolean),
    partial: found.some(Boolean),
    ranks,
    ...(translatedRanks && { translatedRanks }),
  };
}

/** The Questions retrieval scores: all but the unanswerable ones. */
export const searchable = (questions: readonly HardQuestion[]) =>
  questions.filter((question) => question.expected.length > 0);

/** Searches each Question (and its translation) in keyword, vector and hybrid mode. */
export async function searchAll(
  core: Core,
  questions: readonly HardQuestion[],
  documentIds: ReadonlyMap<string, string>,
): Promise<HardQuestionResult[]> {
  const results: HardQuestionResult[] = [];
  for (const question of searchable(questions)) {
    const result: HardQuestionResult = {
      id: question.id,
      language: question.language,
      domain: question.domain,
      difficulty: question.difficulty,
      question: question.question,
      modes: {},
    };
    for (const mode of SEARCH_MODES) {
      const ranks = async (query: string) =>
        ranksOf(
          await core.searchPassages(query, { mode, limit: RANK_DEPTH }),
          question.expected,
          documentIds,
        );
      result.modes[mode] = modeResult(
        await ranks(question.question),
        question.translatedQuery ? await ranks(question.translatedQuery) : undefined,
      );
    }
    results.push(result);
  }
  return results;
}

/**
 * Adds a reranked mode: hybrid search's candidates as the search Tool hands
 * them to a reranker, or keyword search's top 20, reordered by `reranker`.
 */
export async function rerankAll(
  core: Core,
  questions: readonly HardQuestion[],
  documentIds: ReadonlyMap<string, string>,
  reranker: OpenReranker,
  results: readonly HardQuestionResult[],
): Promise<void> {
  const mode = reranker.mode as RetrievalMode;
  const search = rerankedSearchOf(mode);
  if (!search) throw new Error(`${mode} isn't a reranked mode.`);
  const candidatesOf = search === "hybrid" ? rerankCandidates : keywordRerankCandidates;
  for (const question of searchable(questions)) {
    const result = results.find((each) => each.id === question.id);
    if (!result) throw new Error(`${question.id}: no results to add ${mode} to.`);
    const ranks = async (query: string) =>
      ranksOf(
        await reranker.rerank(query, await candidatesOf(core, query)),
        question.expected,
        documentIds,
      );
    result.modes[mode] = modeResult(
      await ranks(question.question),
      question.translatedQuery ? await ranks(question.translatedQuery) : undefined,
    );
  }
}

export interface HardTally {
  hits: number;
  /** Questions with some, not all, of their Passages in the top 5. */
  partial: number;
  total: number;
}

const tally = (results: readonly HardQuestionResult[], mode: RetrievalMode): HardTally => ({
  hits: results.filter((result) => result.modes[mode]?.hit).length,
  partial: results.filter((result) => result.modes[mode]?.partial && !result.modes[mode]?.hit)
    .length,
  total: results.filter((result) => result.modes[mode]).length,
});

/** A mode's hits: overall, and by difficulty, domain, language, and domain × difficulty. */
export interface HardModeSummary {
  all: HardTally;
  byDifficulty: Partial<Record<Difficulty, HardTally>>;
  byDomain: Partial<Record<Domain, HardTally>>;
  byLanguage: Partial<Record<HardLanguage, HardTally>>;
  /** Keyed "<domain> <difficulty>". */
  byDomainAndDifficulty: Record<string, HardTally>;
}

export function summariseHard(
  results: readonly HardQuestionResult[],
  modes: readonly RetrievalMode[] = HARD_MODES,
): Partial<Record<RetrievalMode, HardModeSummary>> {
  const summary: Partial<Record<RetrievalMode, HardModeSummary>> = {};
  const groupBy = <K extends string>(key: (result: HardQuestionResult) => K) => {
    const groups = new Map<K, HardQuestionResult[]>();
    for (const result of results)
      groups.set(key(result), [...(groups.get(key(result)) ?? []), result]);
    return groups;
  };
  for (const mode of modes) {
    if (!results.some((result) => result.modes[mode])) continue;
    const of = <K extends string>(key: (result: HardQuestionResult) => K) =>
      Object.fromEntries(
        [...groupBy(key)].map(([group, members]) => [group, tally(members, mode)]),
      );
    summary[mode] = {
      all: tally(results, mode),
      byDifficulty: of((result) => result.difficulty),
      byDomain: of((result) => result.domain),
      byLanguage: of((result) => result.language),
      byDomainAndDifficulty: of((result) => `${result.domain} ${result.difficulty}`),
    };
  }
  return summary;
}

/** The difficulties retrieval reports: every one but "unanswerable". */
export const SEARCHED_DIFFICULTIES: readonly Difficulty[] = DIFFICULTIES.filter(
  (difficulty) => difficulty !== "unanswerable",
);
