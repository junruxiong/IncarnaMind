/**
 * The document-search Tool (ADR-0007, ADR-0009): what an Answer gets when it
 * searches the User's Documents.
 *
 * 1. Hybrid search: keyword (FTS5) and vector search, fused by reciprocal rank
 *    fusion, over the Search scope.
 * 2. Rerank, if a reranker is plugged in (a hook: none is built in yet).
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
  /** How many fused hits hybrid search hands on to the reranker and the clustering. */
  candidates: number;
  /** The most Documents one search returns Passages from. */
  maxDocuments: number;
  /** The most windows (clusters) one search returns from a Document. */
  maxWindowsPerDocument: number;
  /** The most Passages one search returns in all. */
  maxPassages: number;
}

/**
 * Untuned starting values: 8 Passages of about 500 tokens is about 4,000
 * tokens a search, so a few searches fit an Answer's context.
 */
export const SEARCH_TOOL_PARAMETERS: SearchToolParameters = {
  candidates: 30,
  maxDocuments: 4,
  maxWindowsPerDocument: 2,
  maxPassages: 8,
};

/** A hit from hybrid search, with its fused score (higher is better). */
export interface SearchCandidate extends WindowedPassage {
  score: number;
}

/**
 * Reorders (and may rescore or drop) the hybrid hits, best first, e.g. with a
 * Cohere or Voyage reranking model. Scores must stay comparable: higher is better.
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
  /** The live Passages of a Document's window, in reading order. */
  window(documentId: string, window: number): WindowedPassage[];
}

export interface SearchToolOptions {
  /** Only these Documents (a Search scope). Omitted: every Document. */
  documentIds?: readonly string[];
  parameters?: Partial<SearchToolParameters>;
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
  let candidates = await sources.candidates(query, parameters.candidates, options.documentIds);
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
