/**
 * Searching Passages (ADR-0009). Keyword search runs FTS5 over each Passage's
 * segmented words (see ./keywords); vector search scans the in-memory vectors
 * (see ./vectors); hybrid search fuses the two by reciprocal rank fusion. The
 * search Tool (#30) and Search scopes (#36) build on these.
 */
import type { PassageSearchResult } from "../api";
import type { Database } from "../storage";
import { keywordQuery } from "./keywords";

/** Reciprocal rank fusion's k: a Passage at rank r (from 1) in a list scores 1 / (k + r). */
export const RRF_K = 60;

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

/** Fuses ranked lists of `seq`s by reciprocal rank fusion: the best `limit`, best first. */
export function fuseRankings(rankings: readonly (readonly number[])[], limit: number): number[] {
  const scores = new Map<number, number>();
  for (const ranking of rankings) {
    ranking.forEach((seq, index) => {
      scores.set(seq, (scores.get(seq) ?? 0) + 1 / (RRF_K + index + 1));
    });
  }
  // Ties keep the order of first appearance: the keyword list's order comes first.
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([seq]) => seq);
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
