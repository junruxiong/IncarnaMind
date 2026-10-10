/**
 * Searching Passages (ADR-0009). Keyword search runs FTS5 over each Passage's
 * segmented words (see ./keywords); vector search scans the in-memory vectors
 * (see ./vectors); hybrid search fuses the two by reciprocal rank fusion,
 * counting a Passage with no vector by its keyword rank for both. While
 * embeddings are off (the default), hybrid search is keyword search. The
 * search Tool (#30) and Search scopes (#36) build on these.
 */
import type { DocumentKind, PassageSearchResult } from "../api";
import type { Database } from "../storage";
import { keywordQuery } from "./keywords";

/** Reciprocal rank fusion's k: a Passage at rank r (from 1) in a list scores 1 / (k + r). */
const RRF_K = 60;

/** How many results each list hands to a hybrid search. */
export const HYBRID_CANDIDATES = 50;

/** SQL that limits a query to some Documents, if `documentIds` is given. */
const scopeClause = (documentIds: readonly string[] | undefined) =>
  documentIds ? "AND p.document_id IN (SELECT value FROM json_each(?))" : "";

const scopeParams = (documentIds: readonly string[] | undefined) =>
  documentIds ? [JSON.stringify(documentIds)] : [];

/** The `seq`s of the live Passages that best match the query's words, best first. */
export function keywordSearch(
  db: Database,
  query: string,
  limit: number,
  documentIds?: readonly string[],
): number[] {
  const match = keywordQuery(query);
  if (!match) return [];
  return db
    .all<{ seq: number }>(
      `SELECT p.seq
       FROM passages_fts
       JOIN passages p ON p.seq = passages_fts.rowid
       JOIN documents d ON d.id = p.document_id
       WHERE passages_fts MATCH ? AND p.deleted_at IS NULL AND d.deleted_at IS NULL
         ${scopeClause(documentIds)}
       ORDER BY bm25(passages_fts), p.position, p.seq
       LIMIT ?`,
      [match, ...scopeParams(documentIds), BigInt(limit)],
    )
    .map((row) => row.seq);
}

/**
 * The same file at two paths is two Documents with the same Passages (ADR-0010).
 * Search shows each Passage once: every `seq` in the rankings is replaced by
 * that of one copy, and each ranking keeps a Passage's first place. The copy
 * kept is in a Document whose file can be opened, if one is, else the
 * earliest added. Search is limited to the Search scope beforehand, so the
 * copy kept is always one inside it.
 */
export function distinctPassages(
  db: Database,
  rankings: readonly (readonly number[])[],
): number[][] {
  const seqs = [...new Set(rankings.flat())];
  if (seqs.length < 2) return rankings.map((ranking) => [...ranking]);
  const rows = db.all<{ seq: number; key: string }>(
    `SELECT p.seq,
       coalesce(p.content_hash, 'seq:' || p.seq) || ':' || p.position || ':' || length(p.text) AS key
     FROM passages p JOIN documents d ON d.id = p.document_id
     WHERE p.seq IN (SELECT value FROM json_each(?))
     ORDER BY d.file_status <> 'available', d.created_at, d.rowid`,
    [JSON.stringify(seqs)],
  );
  const kept = new Map<string, number>();
  const keyOf = new Map<number, string>();
  for (const row of rows) {
    keyOf.set(row.seq, row.key);
    if (!kept.has(row.key)) kept.set(row.key, row.seq);
  }
  return rankings.map((ranking) => {
    const seen = new Set<number>();
    const distinct: number[] = [];
    for (const seq of ranking) {
      const key = keyOf.get(seq);
      const canonical = key === undefined ? seq : (kept.get(key) ?? seq);
      if (seen.has(canonical)) continue;
      seen.add(canonical);
      distinct.push(canonical);
    }
    return distinct;
  });
}

/** A Passage's `seq` and its fused score: higher is better. */
export interface ScoredSeq {
  seq: number;
  score: number;
}

/**
 * Of these Passages, those with no vector from the current embedding model
 * (`modelId`), such as a Document's not yet embedded after embeddings were
 * turned on: only keyword search can rank them.
 */
export function keywordOnlyPassages(
  db: Database,
  seqs: readonly number[],
  modelId: string,
): Set<number> {
  if (seqs.length === 0) return new Set();
  const rows = db.all<{ seq: number }>(
    `SELECT p.seq FROM passages p JOIN documents d ON d.id = p.document_id
     WHERE p.seq IN (SELECT value FROM json_each(?))
       AND (p.embedding IS NULL OR d.embedding_model IS NOT ?)`,
    [JSON.stringify(seqs), modelId],
  );
  return new Set(rows.map((row) => row.seq));
}

/**
 * Fuses ranked lists of `seq`s by reciprocal rank fusion: the best `limit`,
 * best first, with their fused scores.
 *
 * `keywordOnly` are Passages that only the first list, keyword search's, can
 * hold (see `keywordOnlyPassages`): their rank there counts once for each
 * list that isn't empty, as if vector search had ranked them as keyword
 * search did, so having no vector yet neither drops them nor ranks them below
 * Passages both lists hold. While vector search returns nothing (embeddings
 * are off, or the query couldn't be embedded), nothing changes.
 */
export function fuseRankingScores(
  rankings: readonly (readonly number[])[],
  limit: number,
  keywordOnly: ReadonlySet<number> = new Set(),
): ScoredSeq[] {
  const scores = new Map<number, number>();
  const lists = rankings.filter((ranking) => ranking.length > 0).length;
  rankings.forEach((ranking, list) => {
    ranking.forEach((seq, index) => {
      const counted = list === 0 && keywordOnly.has(seq) ? lists : 1;
      scores.set(seq, (scores.get(seq) ?? 0) + counted / (RRF_K + index + 1));
    });
  });
  // Ties keep the order of first appearance: the keyword list's order comes first.
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([seq, score]) => ({ seq, score }));
}

/** Fuses ranked lists of `seq`s by reciprocal rank fusion: the best `limit`, best first. */
export function fuseRankings(
  rankings: readonly (readonly number[])[],
  limit: number,
  keywordOnly?: ReadonlySet<number>,
): number[] {
  return fuseRankingScores(rankings, limit, keywordOnly).map((hit) => hit.seq);
}

interface ResultRow {
  seq: number;
  passage_id: string;
  document_id: string;
  document_name: string;
  page_from: number | null;
  page_to: number | null;
  position: number;
  text: string;
}

/** The live Passages with these `seq`s, in the order given. */
export function passagesBySeq(db: Database, seqs: readonly number[]): PassageSearchResult[] {
  if (seqs.length === 0) return [];
  const rows = db.all<ResultRow>(
    `SELECT p.seq, p.id AS passage_id, p.document_id, d.name AS document_name,
       p.page_from, p.page_to, p.position, p.text
     FROM passages p JOIN documents d ON d.id = p.document_id
     WHERE p.seq IN (SELECT value FROM json_each(?))
       AND p.deleted_at IS NULL AND d.deleted_at IS NULL`,
    [JSON.stringify(seqs)],
  );
  const bySeq = new Map(rows.map((row) => [row.seq, row]));
  return seqs.flatMap((seq) => {
    const row = bySeq.get(seq);
    return row
      ? [
          {
            passageId: row.passage_id,
            documentId: row.document_id,
            documentName: row.document_name,
            pageFrom: row.page_from,
            pageTo: row.page_to,
            position: row.position,
            text: row.text,
          },
        ]
      : [];
  });
}

/** A live Passage as the document-search Tool sees it: with its sliding window and its Document. */
export interface WindowedPassage extends PassageSearchResult {
  seq: number;
  documentKind: DocumentKind;
  /** The version of the Document the Passage was built from. */
  contentHash: string;
  /** Positions of the first and last Passage in this Passage's sliding window. */
  windowFrom: number;
  windowTo: number;
}

interface WindowedRow extends ResultRow {
  document_kind: string;
  content_hash: string | null;
  window_from: number;
  window_to: number;
}

const WINDOWED_COLUMNS = `p.seq, p.id AS passage_id, p.document_id, d.name AS document_name,
  d.kind AS document_kind, coalesce(p.content_hash, d.content_hash) AS content_hash,
  p.page_from, p.page_to, p.position, p.window_from, p.window_to, p.text`;

const toWindowed = (row: WindowedRow): WindowedPassage => ({
  seq: row.seq,
  passageId: row.passage_id,
  documentId: row.document_id,
  documentName: row.document_name,
  documentKind: row.document_kind as DocumentKind,
  contentHash: row.content_hash ?? "",
  pageFrom: row.page_from,
  pageTo: row.page_to,
  position: row.position,
  windowFrom: row.window_from,
  windowTo: row.window_to,
  text: row.text,
});

/** The live Passages with these `seq`s, in the order given, with their windows. */
export function windowedPassagesBySeq(db: Database, seqs: readonly number[]): WindowedPassage[] {
  if (seqs.length === 0) return [];
  const rows = db.all<WindowedRow>(
    `SELECT ${WINDOWED_COLUMNS}
     FROM passages p JOIN documents d ON d.id = p.document_id
     WHERE p.seq IN (SELECT value FROM json_each(?))
       AND p.deleted_at IS NULL AND d.deleted_at IS NULL`,
    [JSON.stringify(seqs)],
  );
  const bySeq = new Map(rows.map((row) => [row.seq, row]));
  return seqs.flatMap((seq) => {
    const row = bySeq.get(seq);
    return row ? [toWindowed(row)] : [];
  });
}

/**
 * The live Passages of one sliding window of a Document, in reading order:
 * those whose own window range takes in `window`. With windows of 3 and step 1,
 * window w holds the Passages at positions w, w + 1 and w + 2.
 */
export function passagesInWindow(
  db: Database,
  documentId: string,
  window: number,
): WindowedPassage[] {
  return db
    .all<WindowedRow>(
      `SELECT ${WINDOWED_COLUMNS}
       FROM passages p JOIN documents d ON d.id = p.document_id
       WHERE p.document_id = ? AND p.deleted_at IS NULL AND d.deleted_at IS NULL
         AND p.window_from <= ? AND p.window_to >= ?
       ORDER BY p.position`,
      [documentId, BigInt(window), BigInt(window)],
    )
    .map(toWindowed);
}
