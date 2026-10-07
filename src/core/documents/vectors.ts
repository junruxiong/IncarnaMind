/**
 * Vector search (ADR-0009): brute-force cosine similarity over the Passage
 * vectors, held in memory. Vectors are stored as float32 BLOBs on the Passage
 * rows; the first search loads them, and the core keeps the copy in step as
 * Passages are embedded and Documents are deleted or processed again. At
 * 100,000 Passages that is about 154 MB and 40 ms a query, and a Search scope
 * only scans its own Documents.
 */
import type { Database } from "../storage";

/** Vectors are L2-normalised, so their dot product is their cosine similarity. */
export interface VectorHit {
  /** The Passage's `seq`. */
  seq: number;
  score: number;
}

interface DocumentVectors {
  seqs: number[];
  /** `count` vectors, one after another, with spare room at the end. */
  data: Float32Array;
  count: number;
}

/** A vector as stored: its float32s, in the platform's byte order (little-endian on every platform the app runs on). */
export const encodeVector = (vector: Float32Array): Uint8Array =>
  new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);

export function decodeVector(blob: Uint8Array): Float32Array {
  // Copied: the blob's bytes needn't be aligned for a Float32Array.
  const bytes = blob.slice();
  return new Float32Array(bytes.buffer, 0, bytes.byteLength / 4);
}

export interface VectorIndex {
  /** A Passage was embedded. */
  add(documentId: string, seq: number, vector: Float32Array): void;
  /** A Document was deleted, or its Passages replaced. */
  removeDocument(documentId: string): void;
  /** The `limit` Passages most similar to `query`, most similar first. Optionally only some Documents' Passages. */
  search(query: Float32Array, limit: number, documentIds?: readonly string[]): VectorHit[];
}

export function createVectorIndex(
  db: Database,
  model: { readonly id: string; readonly dimensions: number },
): VectorIndex {
  const { dimensions } = model;
  /** Loaded on the first search. */
  let documents: Map<string, DocumentVectors> | undefined;

  function append(
    map: Map<string, DocumentVectors>,
    documentId: string,
    seq: number,
    vector: Float32Array,
  ): void {
    if (vector.length !== dimensions) return;
    let entry = map.get(documentId);
    if (!entry) {
      entry = { seqs: [], data: new Float32Array(dimensions * 8), count: 0 };
      map.set(documentId, entry);
    }
    if ((entry.count + 1) * dimensions > entry.data.length) {
      const grown = new Float32Array(entry.data.length * 2);
      grown.set(entry.data);
      entry.data = grown;
    }
    entry.data.set(vector, entry.count * dimensions);
    entry.seqs.push(seq);
    entry.count++;
  }

  function load(): Map<string, DocumentVectors> {
    const map = new Map<string, DocumentVectors>();
    const rows = db.all<{ seq: number; document_id: string; embedding: Uint8Array }>(
      `SELECT p.seq, p.document_id, p.embedding
       FROM passages p JOIN documents d ON d.id = p.document_id
       WHERE p.deleted_at IS NULL AND d.deleted_at IS NULL AND p.embedding IS NOT NULL
         AND d.embedding_model = ?
       ORDER BY p.document_id, p.position`,
      [model.id],
    );
    for (const row of rows) append(map, row.document_id, row.seq, decodeVector(row.embedding));
    return map;
  }

  return {
    add(documentId, seq, vector) {
      if (documents) append(documents, documentId, seq, vector);
    },

    removeDocument(documentId) {
      documents?.delete(documentId);
    },

    search(query, limit, documentIds) {
      documents ??= load();
      // The best hits so far, best first.
      const top: VectorHit[] = [];
      const scan = (entry: DocumentVectors) => {
        const { data, seqs, count } = entry;
        for (let index = 0; index < count; index++) {
          const offset = index * dimensions;
          let score = 0;
          for (let d = 0; d < dimensions; d++)
            score += (data[offset + d] as number) * (query[d] as number);
          if (top.length === limit && score <= (top[limit - 1] as VectorHit).score) continue;
          let at = top.length;
          while (at > 0 && (top[at - 1] as VectorHit).score < score) at--;
          top.splice(at, 0, { seq: seqs[index] as number, score });
          if (top.length > limit) top.pop();
        }
      };
      if (documentIds) {
        for (const id of new Set(documentIds)) {
          const entry = documents.get(id);
          if (entry) scan(entry);
        }
      } else {
        for (const entry of documents.values()) scan(entry);
      }
      return top;
    },
  };
}
