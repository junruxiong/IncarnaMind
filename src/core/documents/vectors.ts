/**
 * Vector search (ADR-0009): brute-force cosine similarity over the Passage
 * vectors, held in memory. Vectors are stored as float32 BLOBs on the Passage
 * rows; the first search loads them, and the core keeps the copy in step as
 * Passages are embedded and Documents are deleted or processed again. At
 * 100,000 Passages of 384 numbers that is about 154 MB and 40 ms a query, and
 * a Search scope only scans its own Documents.
 *
 * Only vectors from the current embedding model are held: those of Documents
 * whose recorded model (documents.embedding_model) is its id. A query is only
 * compared with vectors of its own size, so vectors from different models are
 * never mixed, even while Documents are being embedded again after a switch.
 *
 * The same vectors give each Document's mean, the vector Documents are
 * grouped into Topics by (R2 in docs/designs/library-structure-view.md).
 */
import type { Database } from "../storage";

/** Vectors are L2-normalised, so their dot product is their cosine similarity. */
export interface VectorHit {
  /** The Passage's `seq`. */
  seq: number;
  score: number;
}

interface DocumentVectors {
  /** The size of each vector: the first one's. Vectors of another size are left out. */
  dimensions: number;
  seqs: number[];
  /** `count` vectors, one after another, with spare room at the end. */
  data: Float32Array;
  count: number;
}

/** A vector as stored: its float32s, in the platform's byte order (little-endian on every platform the app runs on). */
export const encodeVector = (vector: Float32Array): Uint8Array =>
  new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);

function decodeVector(blob: Uint8Array): Float32Array {
  // Copied: the blob's bytes needn't be aligned for a Float32Array.
  const bytes = blob.slice();
  return new Float32Array(bytes.buffer, 0, bytes.byteLength / 4);
}

/** Each Document's vector for grouping Documents into Topics (R2 in docs/designs/library-structure-view.md). */
export interface DocumentMeans {
  /** The Documents, in the order of their means. */
  ids: string[];
  dimensions: number;
  /**
   * One L2-normalised mean per Document, one after another
   * (`ids.length * dimensions` numbers), in a buffer of their own, ready to
   * transfer to a worker.
   */
  data: Float32Array;
}

export interface VectorIndex {
  /** A Passage was embedded with the current model. */
  add(documentId: string, seq: number, vector: Float32Array): void;
  /** A Document was deleted, or its Passages replaced, or its vectors are being replaced. */
  removeDocument(documentId: string): void;
  /** The embedding model changed: the vectors held are dropped, and the next search loads the new model's. */
  reset(): void;
  /**
   * The `limit` Passages most similar to `query`, most similar first, among
   * vectors of its size. Optionally only some Documents' Passages.
   */
  search(query: Float32Array, limit: number, documentIds?: readonly string[]): VectorHit[];
  /**
   * Each Document's mean Passage vector, L2-normalised, from the current
   * model's vectors only. Loads the index first, as a first search does, if
   * it isn't loaded yet. Only Documents whose vectors have `dimensions`
   * numbers are included (by default, the size most Documents have);
   * Documents with no vectors yet are left out.
   */
  documentMeans(dimensions?: number): DocumentMeans;
}

/** `model.id` is read at each load, so it can change (see `reset`). */
export function createVectorIndex(db: Database, model: { readonly id: string }): VectorIndex {
  /** Loaded on the first search. */
  let documents: Map<string, DocumentVectors> | undefined;

  function append(
    map: Map<string, DocumentVectors>,
    documentId: string,
    seq: number,
    vector: Float32Array,
  ): void {
    const dimensions = vector.length;
    if (dimensions === 0) return;
    let entry = map.get(documentId);
    if (!entry) {
      entry = { dimensions, seqs: [], data: new Float32Array(dimensions * 8), count: 0 };
      map.set(documentId, entry);
    }
    if (entry.dimensions !== dimensions) return;
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

    reset() {
      documents = undefined;
    },

    search(query, limit, documentIds) {
      documents ??= load();
      const dimensions = query.length;
      // The best hits so far, best first.
      const top: VectorHit[] = [];
      const scan = (entry: DocumentVectors) => {
        if (entry.dimensions !== dimensions) return;
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

    documentMeans(dimensions) {
      documents ??= load();
      const size = dimensions ?? commonSize(documents);
      const ids: string[] = [];
      for (const [id, entry] of documents) {
        if (entry.dimensions === size && entry.count > 0) ids.push(id);
      }
      const data = new Float32Array(ids.length * size);
      const sum = new Float64Array(size);
      let kept = 0;
      for (const id of ids) {
        const entry = documents.get(id) as DocumentVectors;
        sum.fill(0);
        for (let index = 0; index < entry.count; index++) {
          const offset = index * size;
          for (let d = 0; d < size; d++) {
            sum[d] = (sum[d] as number) + (entry.data[offset + d] as number);
          }
        }
        let norm = 0;
        for (let d = 0; d < size; d++) norm += (sum[d] as number) * (sum[d] as number);
        norm = Math.sqrt(norm);
        // Vectors that cancel out have no direction to group by.
        if (!(norm > 0)) continue;
        const offset = kept * size;
        for (let d = 0; d < size; d++) data[offset + d] = (sum[d] as number) / norm;
        ids[kept++] = id;
      }
      ids.length = kept;
      return { ids, dimensions: size, data: data.slice(0, kept * size) };
    },
  };
}

/** The vector size most Documents have (the first seen on a tie), or 0 with none. */
function commonSize(documents: ReadonlyMap<string, DocumentVectors>): number {
  const counts = new Map<number, number>();
  for (const entry of documents.values()) {
    if (entry.count > 0) counts.set(entry.dimensions, (counts.get(entry.dimensions) ?? 0) + 1);
  }
  let best = 0;
  let bestCount = 0;
  for (const [size, count] of counts) {
    if (count > bestCount) {
      best = size;
      bestCount = count;
    }
  }
  return best;
}
