/**
 * Keyword search over Passage text with the FTS5 index. Later tickets build
 * hybrid search (keyword plus vector) on it.
 */

import { hasCjk } from "../../shared/text";
import type { PassageSearchResult } from "../api";
import type { Database, SqlValue } from "../storage";

/** The trigram tokenizer can't match anything shorter than this. */
const TRIGRAM = 3;
/** Keeps a pasted paragraph from becoming a huge query. */
const MAX_TERMS = 64;

export interface KeywordQuery {
  /** Matched through the index, any of them, ranked by BM25. */
  phrases: string[];
  /** Too short for the index (e.g. "AI", "模型"): matched by scanning, only when there are no phrases. */
  shortTerms: string[];
}

const JOINERS = "-_.'’+#&@/";
const SEPARATOR = new RegExp(`[^\\p{L}\\p{N}\\p{M}${escapeForClass(JOINERS)}]+`, "u");
const EDGE_JOINERS = new RegExp(`^[${escapeForClass(JOINERS)}]+|[-_.'’&@/]+$`, "gu");

function escapeForClass(characters: string): string {
  return characters.replace(/[\\\]^-]/g, "\\$&");
}

/**
 * Splits a query into terms at spaces and punctuation, keeping joiners inside
 * words ("GPT-4", "C++"). Every term is matched as a substring: the trigram
 * tokenizer indexes every 3 characters. Chinese has no spaces, so a long CJK
 * term matches through its overlapping 3-character pieces, any of them; the
 * Passages with the most pieces rank first.
 *
 * PROVISIONAL, like the trigram tokenizer (ticket #21).
 */
export function parseKeywordQuery(query: string): KeywordQuery {
  const phrases = new Set<string>();
  const shortTerms = new Set<string>();
  const terms = query
    .normalize("NFKC")
    .split(SEPARATOR)
    .map((term) => term.replace(EDGE_JOINERS, ""))
    .filter(Boolean);
  for (const term of terms) {
    const characters = Array.from(term);
    if (characters.length < TRIGRAM) shortTerms.add(term);
    else if (characters.length === TRIGRAM || !hasCjk(term)) phrases.add(term);
    else {
      for (let start = 0; start + TRIGRAM <= characters.length; start++) {
        phrases.add(characters.slice(start, start + TRIGRAM).join(""));
      }
    }
  }
  return {
    phrases: [...phrases].slice(0, MAX_TERMS),
    shortTerms: [...shortTerms].slice(0, MAX_TERMS),
  };
}

const quote = (phrase: string) => `"${phrase.replaceAll('"', '""')}"`;

const likePattern = (term: string) => `%${term.replace(/[\\%_]/g, "\\$&")}%`;

interface ResultRow {
  passage_id: string;
  document_id: string;
  document_name: string;
  page_from: number | null;
  page_to: number | null;
  position: number;
  text: string;
}

const COLUMNS = `p.id AS passage_id, p.document_id, d.name AS document_name,
  p.page_from, p.page_to, p.position, p.text`;

const toResult = (row: ResultRow): PassageSearchResult => ({
  passageId: row.passage_id,
  documentId: row.document_id,
  documentName: row.document_name,
  pageFrom: row.page_from,
  pageTo: row.page_to,
  position: row.position,
  text: row.text,
});

/** Live Passages of live Documents matching the query, best first. */
export function searchPassages(db: Database, query: string, limit: number): PassageSearchResult[] {
  const { phrases, shortTerms } = parseKeywordQuery(query);
  if (phrases.length > 0) {
    return db
      .all<ResultRow>(
        `SELECT ${COLUMNS}
         FROM passages_fts
         JOIN passages p ON p.seq = passages_fts.rowid
         JOIN documents d ON d.id = p.document_id
         WHERE passages_fts MATCH ? AND p.deleted_at IS NULL AND d.deleted_at IS NULL
         ORDER BY bm25(passages_fts), p.position
         LIMIT ?`,
        [phrases.map(quote).join(" OR "), limit],
      )
      .map(toResult);
  }
  if (shortTerms.length === 0) return [];
  // No index for these: scan, and rank by how many of the terms a Passage has.
  const hits = shortTerms.map(() => `(p.text LIKE ? ESCAPE '\\')`).join(" + ");
  const params: SqlValue[] = shortTerms.map(likePattern);
  return db
    .all<ResultRow>(
      `SELECT * FROM (
         SELECT ${COLUMNS}, d.created_at AS added_at, ${hits} AS hits
         FROM passages p
         JOIN documents d ON d.id = p.document_id
         WHERE p.deleted_at IS NULL AND d.deleted_at IS NULL
       )
       WHERE hits > 0
       ORDER BY hits DESC, added_at DESC, position
       LIMIT ?`,
      [...params, limit],
    )
    .map(toResult);
}
