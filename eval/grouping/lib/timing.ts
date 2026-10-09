/**
 * The grouping check's timings, on generated vectors:
 * - k-means at 5,000 Documents (R1: it runs on the grouping worker);
 * - the means on the core's thread at 100,000 Passages (R2, O5): loading the
 *   vector index from SQLite plus computing every Document's mean must take
 *   at most 300 ms, or the named fallback applies: the grouping worker reads
 *   the vectors itself, through its own read-only connection.
 */
import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createVectorIndex, encodeVector } from "../../../src/core/documents/vectors";
import { migrate, openDatabase } from "../../../src/core/storage";
import {
  DEFAULT_SEED,
  groupDocuments,
  type KMeansRun,
  normalised,
  seededRandom,
} from "../../../src/core/topics/grouping";

export const MAIN_THREAD_BAR_MS = 300;

/** Unit vectors around `centres` random directions, with seeded noise: data k-means has to work on. */
export function syntheticVectors(
  count: number,
  dimensions: number,
  centres: number,
  seed = DEFAULT_SEED,
): Float32Array[] {
  const random = seededRandom(seed);
  const gaussian = () => {
    // Box–Muller.
    const u = Math.max(random(), Number.EPSILON);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
  };
  const directions = Array.from({ length: centres }, () =>
    normalised(Array.from({ length: dimensions }, gaussian)),
  ) as Float32Array[];
  return Array.from({ length: count }, (_, index) => {
    const centre = directions[index % centres] as Float32Array;
    const noisy = Array.from(centre, (value) => value + gaussian() * (0.6 / Math.sqrt(dimensions)));
    return normalised(noisy) as Float32Array;
  });
}

export interface KMeansTiming {
  documents: number;
  dimensions: number;
  k: number;
  seconds: number;
  runs: KMeansRun[];
}

/** groupDocuments as the grouping worker would run it: k from the count, 3 seeded runs. */
export function timeKMeans(documents = 5_000, dimensions = 384): KMeansTiming {
  const vectors = syntheticVectors(documents, dimensions, 60);
  const started = performance.now();
  const grouping = groupDocuments(vectors.map((vector, index) => ({ id: String(index), vector })));
  const seconds = (performance.now() - started) / 1000;
  return { documents, dimensions, k: grouping.k, seconds, runs: grouping.runs };
}

export interface MeansTiming {
  passages: number;
  documents: number;
  dimensions: number;
  /** Characters of text per Passage: the rows are read with their text, as the app's are. */
  textLength: number;
  databaseMb: number;
  /** Writing the generated database (not part of the bar). */
  buildSeconds: number;
  /** Loading the index and computing the means, from a fresh index each time: what the bar is about. */
  coldMs: number[];
  /** The means alone, with the index loaded (after a first search, the usual case). */
  warmMs: number[];
  barMs: number;
  /** The median cold time is within the bar. */
  passes: boolean;
}

export function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? (sorted[middle] as number)
    : ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2;
}

/**
 * Writes a database with `passages` Passages of random vectors in a temporary
 * folder, then times the index's documentMeans on it, cold and warm, three
 * times each. The folder is deleted afterwards.
 */
export async function timeMeans({
  passages = 100_000,
  documents = 2_000,
  dimensions = 384,
  textLength = 1_500,
  repeats = 3,
} = {}): Promise<MeansTiming> {
  const dir = await mkdtemp(join(tmpdir(), "incarnamind-eval-means-"));
  const file = join(dir, "means.sqlite");
  const model = { id: "timing-model" };
  try {
    const built = performance.now();
    const db = openDatabase(file);
    migrate(db);
    const now = new Date().toISOString();
    const random = seededRandom(DEFAULT_SEED);
    const filler = "lorem ipsum dolor sit amet consectetur adipiscing elit ".repeat(
      Math.ceil(textLength / 56),
    );
    const vector = new Float32Array(dimensions);
    const perDocument = Math.ceil(passages / documents);
    let seq = 0;
    for (let first = 0; first < documents; first += 100) {
      db.transaction(() => {
        for (let d = first; d < Math.min(documents, first + 100); d++) {
          const id = `document-${d}`;
          db.run(
            `INSERT INTO documents (id, content_hash, name, kind, size, status, created_at, updated_at,
               embedding_model, embedding_dimensions)
             VALUES (?, ?, ?, 'pdf', 1, 'ready', ?, ?, ?, ?)`,
            [id, `hash-${d}`, id, now, now, model.id, BigInt(dimensions)],
          );
          for (let p = 0; p < perDocument && seq < passages; p++) {
            seq++;
            for (let i = 0; i < dimensions; i++) vector[i] = random() - 0.5;
            const unit = normalised(vector) as Float32Array;
            db.run(
              `INSERT INTO passages (seq, id, document_id, position, window_from, window_to, text,
                 created_at, updated_at, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
              [
                BigInt(seq),
                `passage-${seq}`,
                id,
                BigInt(p),
                BigInt(p),
                BigInt(p),
                `${seq} ${filler}`.slice(0, textLength),
                now,
                now,
                encodeVector(unit),
              ],
            );
          }
        }
      });
    }
    db.close();
    const buildSeconds = (performance.now() - built) / 1000;
    const databaseMb = (await stat(file)).size / 1e6;

    const coldMs: number[] = [];
    const warmMs: number[] = [];
    for (let run = 0; run < repeats; run++) {
      // A new connection and a new index: nothing is held in memory yet.
      const reader = openDatabase(file);
      try {
        const index = createVectorIndex(reader, model);
        let started = performance.now();
        const means = index.documentMeans();
        coldMs.push(performance.now() - started);
        if (means.ids.length !== documents) {
          throw new Error(`Expected ${documents} means, got ${means.ids.length}.`);
        }
        started = performance.now();
        index.documentMeans();
        warmMs.push(performance.now() - started);
      } finally {
        reader.close();
      }
    }
    return {
      passages: seq,
      documents,
      dimensions,
      textLength,
      databaseMb,
      buildSeconds,
      coldMs,
      warmMs,
      barMs: MAIN_THREAD_BAR_MS,
      passes: median(coldMs) <= MAIN_THREAD_BAR_MS,
    };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}
