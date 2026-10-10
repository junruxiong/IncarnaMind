/**
 * The document-search Tool (ADR-0007, ADR-0009): what an Answer gets when it
 * searches the User's Documents.
 *
 * 1. Keyword search (FTS5) over the Search scope, and, while the User has
 *    embeddings on, vector search, fused by reciprocal rank fusion (hybrid
 *    search). Embeddings are off by default (ADR-0009, 2026-10-10).
 * 2. Rerank, unless the User turned reranking off: with the built-in reranking
 *    model by default, or a Cohere or Voyage key (see ../providers/rerank),
 *    which the retrieval evaluation gates on (#31). With embeddings off, the
 *    reranker sees keyword search's top 60. With them on, it sees keyword
 *    search's top 10 and vector search's top 10, each Passage once, instead
 *    of the fused list: fusion can push a hit that only one of them found
 *    below others that both found middlingly (#31: en-03, en-12, en-14 and
 *    zh-03); until vector search finds anything, keyword search's top 60
 *    again. Those are then all the hits.
 * 3. Group the hits by Document.
 * 4. The old backend's sliding-window clustering (its `find_overlaps`), ported:
 *    each Passage belongs to the overlapping windows of 3 consecutive Passages
 *    that its window range names; hits whose ranges intersect form a cluster;
 *    each cluster's centroid is the window that takes in most of it; and the
 *    Passages returned are that window's, so a hit comes with its neighbours.
 * 5. The best windows, a few per Document, become the Passages returned, with
 *    their Document, pages and text, in reading order. A window ranks by its
 *    best hit, then by the summed scores of its hits: fused scores are close
 *    together (reciprocal rank fusion, k = 60), so a sum mostly counts hits,
 *    and a long, repetitive Document's windows of three weaker hits would
 *    otherwise outrank a short one's single best match.
 *
 * Every count is a parameter; the retrieval evaluation (#31) tunes them.
 */
import type { WindowedPassage } from "./search";

export interface SearchToolParameters {
  /** How many fused hits hybrid search hands on to the clustering, without a reranker. */
  candidates: number;
  /**
   * With a reranker and vector search (embeddings on): how many of keyword
   * search's best, and of vector search's, it sees (each Passage once, so 10
   * to 20 with 10), and then the clustering. Each costs the built-in
   * reranking model tens of milliseconds.
   */
  rerankPerList: number;
  /**
   * With a reranker and no vector search (embeddings off, the default, or no
   * vectors yet in the Search scope): how many of keyword search's best it
   * sees. More than with vector search, since keyword search alone ranks
   * Passages worded unlike the Question lower (ADR-0009, 2026-10-10).
   */
  keywordRerankCandidates: number;
  /** The most Documents one search returns Passages from. */
  maxDocuments: number;
  /** The most windows (clusters) one search returns from a Document. */
  maxWindowsPerDocument: number;
  /** The most Passages one search returns in all. */
  maxPassages: number;
}

/**
 * 8 Passages of about 500 tokens is about 4,000 tokens a search, so a few
 * searches fit an Answer's context. The reranker's candidates are the
 * evaluation's (#31): with embeddings off, keyword search's top 60 found as
 * many Questions as keyword and vector search's top 10 each with them on,
 * where its top 20 found two fewer, for about 1.6 s a search with the
 * built-in reranking model against 0.8 s (ADR-0009, 2026-10-10). The other
 * counts are untuned starting values.
 */
export const SEARCH_TOOL_PARAMETERS: SearchToolParameters = {
  candidates: 30,
  rerankPerList: 10,
  keywordRerankCandidates: 60,
  maxDocuments: 4,
  maxWindowsPerDocument: 2,
  maxPassages: 8,
};

/** A hit from hybrid search, with its fused score (higher is better). */
export interface SearchCandidate extends WindowedPassage {
  score: number;
}

/**
 * The best `perList` of each ranked list, each item once, in the order they
 * first appear (the first list's, then what the next adds): the candidates a
 * reranker sees, from keyword and vector search. The evaluation's reranked
 * modes build theirs with it too.
 */
export function topsOfEach<Item>(
  lists: readonly (readonly Item[])[],
  perList: number,
  key: (item: Item) => unknown = (item) => item,
): Item[] {
  const seen = new Set<unknown>();
  const union: Item[] = [];
  for (const list of lists) {
    for (const item of list.slice(0, perList)) {
      const id = key(item);
      if (seen.has(id)) continue;
      seen.add(id);
      union.push(item);
    }
  }
  return union;
}

/**
 * Reorders (and may rescore or drop) the hybrid hits, best first, e.g. with
 * the built-in reranking model or a Cohere or Voyage one. Scores must stay
 * comparable and positive: higher is better.
 */
export type Reranker = (
  query: string,
  candidates: readonly SearchCandidate[],
  signal?: AbortSignal,
) => Promise<SearchCandidate[]>;

/** Where one hit sits: its sliding-window range, and how good a match it is. */
export interface WindowedHit {
  windowFrom: number;
  windowTo: number;
  score: number;
}

/** A cluster's centroid window, with the hits it takes in. */
export interface CentroidWindow<Hit extends WindowedHit = WindowedHit> {
  window: number;
  /** The summed scores of `hits`: which window takes in most of a cluster. */
  score: number;
  /** The best score among `hits`: how windows rank (see `byRank`). */
  best: number;
  hits: Hit[];
}

const contains = (hit: WindowedHit, window: number) =>
  hit.windowFrom <= window && window <= hit.windowTo;

/** Windows best first: by their best hit, then by their summed score. */
const byRank = (a: CentroidWindow, b: CentroidWindow) => b.best - a.best || b.score - a.score;

/**
 * Hits whose window ranges intersect, directly or through other hits: the
 * clusters of one Document. Interval intersection, as the old `find_overlaps`
 * did, but transitive and in order, so the result doesn't depend on the order
 * of the hits.
 */
export function clusterByOverlap<Hit extends WindowedHit>(hits: readonly Hit[]): Hit[][] {
  const sorted = [...hits].sort((a, b) => a.windowFrom - b.windowFrom || a.windowTo - b.windowTo);
  const clusters: Hit[][] = [];
  let current: Hit[] = [];
  let reach = Number.NEGATIVE_INFINITY;
  for (const hit of sorted) {
    if (current.length > 0 && hit.windowFrom > reach) {
      clusters.push(current);
      current = [];
      reach = Number.NEGATIVE_INFINITY;
    }
    current.push(hit);
    reach = Math.max(reach, hit.windowTo);
  }
  if (current.length > 0) clusters.push(current);
  return clusters;
}

/**
 * The centroid windows of one cluster: the window that takes in the most of
 * it, by summed score (ties: nearest the cluster's score-weighted middle, then
 * the earlier window), then the same for the hits it left out, until every hit
 * is in a window. A cluster whose ranges all intersect has one centroid, in
 * the middle of their intersection; a long chain of hits gets several.
 */
export function centroidWindows<Hit extends WindowedHit>(
  cluster: readonly Hit[],
): CentroidWindow<Hit>[] {
  const windows: CentroidWindow<Hit>[] = [];
  let remaining = [...cluster];
  while (remaining.length > 0) {
    const weight = remaining.reduce((sum, hit) => sum + hit.score, 0);
    const middle =
      weight > 0
        ? remaining.reduce(
            (sum, hit) => sum + hit.score * ((hit.windowFrom + hit.windowTo) / 2),
            0,
          ) / weight
        : remaining.reduce((sum, hit) => sum + (hit.windowFrom + hit.windowTo) / 2, 0) /
          remaining.length;
    const from = Math.min(...remaining.map((hit) => hit.windowFrom));
    const to = Math.max(...remaining.map((hit) => hit.windowTo));
    let best = from;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let window = from; window <= to; window++) {
      const score = remaining.reduce(
        (sum, hit) => (contains(hit, window) ? sum + hit.score : sum),
        0,
      );
      const closer = Math.abs(window - middle) < Math.abs(best - middle);
      if (score > bestScore || (score === bestScore && closer)) {
        best = window;
        bestScore = score;
      }
    }
    const hits = remaining.filter((hit) => contains(hit, best));
    const top = Math.max(...hits.map((hit) => hit.score));
    windows.push({ window: best, score: bestScore, best: top, hits });
    remaining = remaining.filter((hit) => !contains(hit, best));
  }
  return windows;
}

/**
 * One Document's windows, best first (`byRank`, then the earlier window): its
 * hits clustered, each cluster's centroids.
 */
export function documentWindows<Hit extends WindowedHit>(
  hits: readonly Hit[],
): CentroidWindow<Hit>[] {
  return clusterByOverlap(hits)
    .flatMap((cluster) => centroidWindows(cluster))
    .sort((a, b) => byRank(a, b) || a.window - b.window);
}

export interface SearchToolSources {
  /** Hybrid search: the best `limit` live Passages, best first, with fused scores. */
  candidates(
    query: string,
    limit: number,
    documentIds: readonly string[] | undefined,
  ): Promise<SearchCandidate[]>;
  /**
   * For a reranker: keyword search's best `perList` live Passages and vector
   * search's, each once (see `topsOfEach`), with their fused scores, in
   * fused order; keyword search's best `keywordAlone` while vector search
   * has none (embeddings off, or no vectors yet).
   */
  rerankCandidates(
    query: string,
    counts: { perList: number; keywordAlone: number },
    documentIds: readonly string[] | undefined,
  ): Promise<SearchCandidate[]>;
  /** The live Passages of a Document's window, in reading order. */
  window(documentId: string, window: number): WindowedPassage[];
}

export interface SearchToolOptions {
  /** Only these Documents (a Search scope). Omitted: every Document. */
  documentIds?: readonly string[];
  parameters?: Partial<SearchToolParameters>;
  /** The reranker, while reranking is on; undefined otherwise. */
  rerank?: Reranker;
  signal?: AbortSignal;
}

/**
 * Runs the document-search Tool: the Passages to give the model for `query`,
 * grouped by Document (the best Document first) and in reading order within one.
 */
export async function searchDocumentsTool(
  sources: SearchToolSources,
  query: string,
  options: SearchToolOptions = {},
): Promise<WindowedPassage[]> {
  const parameters = { ...SEARCH_TOOL_PARAMETERS, ...options.parameters };
  let candidates = options.rerank
    ? await sources.rerankCandidates(
        query,
        { perList: parameters.rerankPerList, keywordAlone: parameters.keywordRerankCandidates },
        options.documentIds,
      )
    : await sources.candidates(query, parameters.candidates, options.documentIds);
  if (options.rerank && candidates.length > 0) {
    candidates = await options.rerank(query, candidates, options.signal);
  }

  const byDocument = new Map<string, SearchCandidate[]>();
  for (const candidate of candidates) {
    const list = byDocument.get(candidate.documentId) ?? [];
    list.push(candidate);
    byDocument.set(candidate.documentId, list);
  }
  const windows = [...byDocument].flatMap(([documentId, hits]) =>
    documentWindows(hits).map((window) => ({ documentId, ...window })),
  );
  windows.sort(byRank);

  const documents: string[] = [];
  const windowsTaken = new Map<string, number>();
  const passages = new Map<string, WindowedPassage[]>();
  const seen = new Set<string>();
  let count = 0;
  for (const { documentId, window } of windows) {
    if (count >= parameters.maxPassages) break;
    if (!documents.includes(documentId)) {
      if (documents.length >= parameters.maxDocuments) continue;
      documents.push(documentId);
    }
    const taken = windowsTaken.get(documentId) ?? 0;
    if (taken >= parameters.maxWindowsPerDocument) continue;
    windowsTaken.set(documentId, taken + 1);
    const list = passages.get(documentId) ?? [];
    for (const passage of sources.window(documentId, window)) {
      if (count >= parameters.maxPassages) break;
      if (seen.has(passage.passageId)) continue;
      seen.add(passage.passageId);
      list.push(passage);
      count++;
    }
    passages.set(documentId, list);
  }
  return documents.flatMap((documentId) =>
    (passages.get(documentId) ?? []).sort((a, b) => a.position - b.position),
  );
}
