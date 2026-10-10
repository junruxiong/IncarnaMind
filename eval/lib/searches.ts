/**
 * Other ways to find the Passages a reranker sees, with embeddings off:
 * reranked modes reported next to the gate (keyword search's top 20,
 * reranked), never gating, to see which recovers the Passages keyword search
 * ranks too low (see `RerankedSearch` in ./retrieval). Each builds at most 20
 * candidates, like the gate's, unless it is about more candidates:
 *
 * - Keyword top 40 and top 60: keyword search's top 40 or 60.
 * - Keyword + feedback terms (pseudo-relevance feedback, RM3-style, no
 *   model): words from keyword search's top 10 Passages added to the query,
 *   searched again, fused with keyword search's own.
 * - Keyword + rewrites, keyword + sub-questions: a chat model's other
 *   phrasings of the Question, or the Question broken down (./rewrites), each
 *   searched, fused with the Question's own search.
 * - Small-to-big: BM25 over sub-chunks of at most 128 tokens (./subChunks),
 *   each Passage scored by its best sub-chunk.
 * - Document first: the 3 Documents that hold most of keyword search's top 20,
 *   then BM25 inside each with that Document's own term statistics, merged.
 *
 * The reranker always reorders the candidates by the Question itself.
 */
import type { Core, PassageSearchResult } from "../../src/core";
import { isStopword } from "../../src/core/documents/keywords";
import { RRF_K } from "../../src/core/documents/search";
import { type Corpus, type CorpusPassage, queryWeights } from "./corpus";
import type { SubChunkIndex } from "./subChunks";

/** How many candidates the modes hand a reranker, as the gate does: keyword search's top 20. */
export const CANDIDATES = 20;

/** What a search hands the reranker, and what it searched for besides the Question, if anything. */
export interface Candidates {
  candidates: PassageSearchResult[];
  /** The other queries or terms searched, or the Documents searched inside, for the report. */
  queries?: string[];
}

/**
 * Fuses rankings of Passages by reciprocal rank fusion, as hybrid search
 * fuses keyword and vector search's (k = 60): the best `limit`, each Passage
 * once. A list's weight scales its share; ties keep the order Passages first
 * appear in, the first list's first.
 */
export function fusePassages<P extends PassageSearchResult>(
  rankings: readonly (readonly P[])[],
  limit: number,
  weights: readonly number[] = [],
): P[] {
  const fused = new Map<string, { passage: P; score: number }>();
  rankings.forEach((ranking, list) => {
    const weight = weights[list] ?? 1;
    ranking.forEach((passage, index) => {
      const known = fused.get(passage.passageId);
      const score = weight / (RRF_K + index + 1);
      if (known) known.score += score;
      else fused.set(passage.passageId, { passage, score });
    });
  });
  return [...fused.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((each) => each.passage);
}

/** Keyword search's top `depth`. */
export const keywordTop = (core: Core, query: string, depth: number) =>
  core.searchPassages(query, { mode: "keyword", limit: depth });

/**
 * Pseudo-relevance feedback's parameters, RM3's usual ones (as in Anserini):
 * the top 10 Passages give 10 terms, and the Question's own words keep half
 * the expanded query's weight. With ~500-token Passages, 10 of them are about
 * 5,000 words to draw from, mostly from the right Document when keyword
 * search's top results are on topic; more terms would drift from the Question.
 */
export const FEEDBACK = { passages: 10, terms: 10, questionWeight: 0.5 } as const;

/**
 * Generic words feedback terms leave out besides keyword search's stopwords:
 * more function words, in English and Chinese, which a Passage repeats
 * without saying what it is about. Not tuned to any evaluation set.
 */
const FEEDBACK_STOPWORDS: ReadonlySet<string> = new Set([
  ..."also all any both but each either if into may might more most must no nor not now only other our ours over same she he him his her hers so some such than through too under until up upon us very via we whether while within without yet one two three use used using however thus therefore since per".split(
    " ",
  ),
  ..."也 而 并 但 但是 从 以 其 之 等 将 由 所 于 到 着 过 后 前 还 又 就 让 使 因为 所以 如果 我们 他们 它 它们 其中 以及 通过 进行 一种 不 没有 已 已经 并且 或者 以上 以下 之一 一些 这些 那些 此 该 各 即 及其".split(
    " ",
  ),
]);

/**
 * Whether a word can be a feedback term: not a stopword, not a number, and
 * at least 2 characters (a single CJK character, like a single letter, is
 * mostly a function word or part of a word the segmenter couldn't place).
 */
const feedbackWord = (word: string) =>
  [...word].length >= 2 &&
  !/^\p{N}+$/u.test(word) &&
  !isStopword(word) &&
  !FEEDBACK_STOPWORDS.has(word);

/**
 * Feedback terms for a query from its top Passages: each word's share of
 * each Passage (RM1's P(w|D)), weighted by the Passage's share of their BM25
 * scores for the query (P(Q|D)), summed, then times the word's inverse
 * document frequency, so the library's common words don't crowd out the
 * specific ones. The best `FEEDBACK.terms`, with weights that add up to 1.
 */
export function feedbackTerms(
  corpus: Corpus,
  query: string,
  feedback: readonly PassageSearchResult[],
): { term: string; weight: number }[] {
  const original = queryWeights(query);
  const passages = feedback
    .map((passage) => corpus.byId(passage.passageId))
    .filter((passage): passage is CorpusPassage => passage !== undefined);
  const scores = passages.map((passage) => corpus.index.score(passage.seq, original));
  const total = scores.reduce((sum, score) => sum + score, 0);
  const weights = new Map<string, number>();
  passages.forEach((passage, index) => {
    const share = total > 0 ? (scores[index] as number) / total : 1 / passages.length;
    const words = corpus.textTokens(passage.seq);
    for (const word of words) {
      if (original.has(word) || !feedbackWord(word)) continue;
      weights.set(word, (weights.get(word) ?? 0) + share / words.length);
    }
  });
  const ranked = [...weights]
    .map(([term, weight]) => ({ term, weight: weight * corpus.index.idf(term) }))
    .sort((a, b) => b.weight - a.weight || a.term.localeCompare(b.term))
    .slice(0, FEEDBACK.terms);
  const sum = ranked.reduce((all, each) => all + each.weight, 0);
  return ranked.map((each) => ({ term: each.term, weight: sum > 0 ? each.weight / sum : 0 }));
}

/**
 * Keyword + feedback terms: keyword search's top 20; then the query with the
 * feedback terms of its top 10 added (the Question's words weighing half in
 * all, the terms the other half, as RM3 mixes them), searched with BM25
 * over the library; the two rankings fused, the best 20.
 */
export async function feedbackCandidates(
  core: Core,
  corpus: Corpus,
  query: string,
): Promise<Candidates> {
  const own = await keywordTop(core, query, CANDIDATES);
  const terms = feedbackTerms(corpus, query, own.slice(0, FEEDBACK.passages));
  if (terms.length === 0) return { candidates: own, queries: [] };
  const original = queryWeights(query);
  const expanded = new Map<string, number>();
  for (const term of original.keys()) {
    expanded.set(term, FEEDBACK.questionWeight / original.size);
  }
  for (const { term, weight } of terms) {
    expanded.set(term, (expanded.get(term) ?? 0) + (1 - FEEDBACK.questionWeight) * weight);
  }
  const again = corpus.index
    .search(expanded, CANDIDATES)
    .flatMap(({ seq }) => corpus.passages.get(seq) ?? []);
  return {
    candidates: fusePassages<PassageSearchResult>([own, again], CANDIDATES),
    queries: terms.map(({ term }) => term),
  };
}

/** Keyword + rewrites or sub-questions: the Question's and each other query's keyword top 20, fused. */
export async function fusedQueriesCandidates(
  core: Core,
  query: string,
  others: readonly string[],
): Promise<Candidates> {
  const rankings = await Promise.all(
    [query, ...others].map((each) => keywordTop(core, each, CANDIDATES)),
  );
  return { candidates: fusePassages(rankings, CANDIDATES), queries: [...others] };
}

/** Small-to-big: Passages by their best sub-chunk's BM25 score. */
export function smallToBigCandidates(index: SubChunkIndex, query: string): Candidates {
  return { candidates: index.search(query, CANDIDATES, "best") };
}

/** Document first's parameters: how many Documents it searches inside. */
export const DOCUMENT_FIRST = { documents: 3 } as const;

/**
 * Document first: each Document scored by its Passages in keyword search's
 * top 20, each counting 1 / (60 + rank) as in fusion, so a Document with many
 * good Passages comes first; then, inside each of the best 3, BM25 with that
 * Document's own term statistics, so a word it uses everywhere weighs less
 * there. The Documents' rankings are fused, each weighing its score as a
 * share of the best Document's; the best 20.
 */
export async function documentFirstCandidates(
  core: Core,
  corpus: Corpus,
  query: string,
): Promise<Candidates> {
  const top = await keywordTop(core, query, CANDIDATES);
  const votes = new Map<string, number>();
  top.forEach((passage, index) => {
    votes.set(passage.documentId, (votes.get(passage.documentId) ?? 0) + 1 / (RRF_K + index + 1));
  });
  const chosen = [...votes].sort((a, b) => b[1] - a[1]).slice(0, DOCUMENT_FIRST.documents);
  const best = chosen[0]?.[1] ?? 1;
  const terms = queryWeights(query);
  const rankings = chosen.map(([documentId]) =>
    corpus
      .documentIndex(documentId)
      .search(terms, CANDIDATES)
      .flatMap(({ seq }) => corpus.passages.get(seq) ?? []),
  );
  return {
    candidates: fusePassages(
      rankings,
      CANDIDATES,
      chosen.map(([, score]) => score / best),
    ),
    queries: chosen.map(
      ([documentId]) =>
        corpus.documents.find((document) => document.id === documentId)?.name ?? documentId,
    ),
  };
}
