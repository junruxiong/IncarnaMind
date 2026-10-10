/**
 * How long keyword search takes at the hard tier's size: FTS5 has no top-k
 * pruning, so every Passage that matches a query's words is scored before
 * the best are kept. Measurement only: each query (the Questions and their
 * translations) is run on the run's own data folder with the app's SQL,
 * ordered by `bm25(passages_fts)` as `keywordSearch` (src/core/documents/
 * search.ts) orders it, and again ordered by FTS5's `rank` column, which is
 * bm25() with the same (default) weights. Both keep the app's tie-breaks, so
 * they should return the same Passages; how many queries did is reported.
 */
import { keywordQuery } from "../../../src/core/documents/keywords";
import type { Database } from "../../../src/core/storage";

/** Times per query, in milliseconds. */
export interface QueryTiming {
  queries: number;
  mean: number;
  median: number;
  p95: number;
  max: number;
}

export interface KeywordTiming {
  /** Live Passages searched. */
  passages: number;
  /** How many Passages each query keeps: keyword + rerank's candidates. */
  limit: number;
  /** Ordered by `bm25(passages_fts)`, as the app orders keyword search. */
  bm25: QueryTiming;
  /** Ordered by FTS5's `rank` column (bm25() with the same weights). */
  rank: QueryTiming;
  /** Queries for which both orders returned the same Passages in the same order. */
  sameResults: number;
}

const select = (order: string) => `SELECT p.seq
       FROM passages_fts
       JOIN passages p ON p.seq = passages_fts.rowid
       JOIN documents d ON d.id = p.document_id
       WHERE passages_fts MATCH ? AND p.deleted_at IS NULL AND d.deleted_at IS NULL
       ORDER BY ${order}, p.position, p.seq
       LIMIT ?`;

/** The app's keyword search (`keywordSearch`), over the whole library. */
export const BM25_SQL = select("bm25(passages_fts)");

/** The same, ordered by FTS5's `rank` column. */
export const RANK_SQL = select("rank");

const quantile = (sorted: readonly number[], share: number) =>
  sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(share * sorted.length) - 1))] ?? 0;

export function timingOf(milliseconds: readonly number[]): QueryTiming {
  const sorted = [...milliseconds].sort((a, b) => a - b);
  return {
    queries: sorted.length,
    mean: sorted.reduce((sum, ms) => sum + ms, 0) / Math.max(1, sorted.length),
    median: quantile(sorted, 0.5),
    p95: quantile(sorted, 0.95),
    max: sorted.at(-1) ?? 0,
  };
}

/**
 * Runs every query once to warm the page cache, then times each in both
 * orders, alternating which goes first so neither always runs on a warmer
 * cache.
 */
export function timeKeywordSearch(
  db: Database,
  queries: readonly string[],
  limit: number,
): KeywordTiming {
  const matches = queries
    .map((query) => keywordQuery(query))
    .filter((match): match is string => match !== null);
  const run = (sql: string, match: string) => {
    const started = performance.now();
    const seqs = db.all<{ seq: number }>(sql, [match, BigInt(limit)]).map((row) => row.seq);
    return { ms: performance.now() - started, seqs };
  };
  for (const match of matches) run(BM25_SQL, match);
  const bm25: number[] = [];
  const rank: number[] = [];
  let sameResults = 0;
  matches.forEach((match, index) => {
    const [first, second] = index % 2 === 0 ? [BM25_SQL, RANK_SQL] : [RANK_SQL, BM25_SQL];
    const a = run(first, match);
    const b = run(second, match);
    const [byBm25, byRank] = first === BM25_SQL ? [a, b] : [b, a];
    bm25.push(byBm25.ms);
    rank.push(byRank.ms);
    if (byBm25.seqs.join() === byRank.seqs.join()) sameResults++;
  });
  return {
    passages:
      db.get<{ count: number }>("SELECT COUNT(*) AS count FROM passages WHERE deleted_at IS NULL")
        ?.count ?? 0,
    limit,
    bm25: timingOf(bm25),
    rank: timingOf(rank),
    sameResults,
  };
}
