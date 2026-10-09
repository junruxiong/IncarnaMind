/**
 * The vector index's per-Document means (R2 in
 * docs/designs/library-structure-view.md): each Document's vector for
 * grouping is the L2-normalised mean of its Passage vectors from the current
 * embedding model, computed from the vectors the index holds in memory.
 */
import { afterEach, describe, expect, test } from "vitest";
import { createVectorIndex, encodeVector } from "../../src/core/documents/vectors";
import { type Database, migrate, openDatabase } from "../../src/core/storage";

const MODEL = { id: "test-model" };
const NOW = "2026-10-07T09:00:00.000Z";

let db: Database;
let nextPassage = 1;

function open(): Database {
  db = openDatabase(":memory:");
  migrate(db);
  return db;
}

afterEach(() => db?.close());

function addDocument(id: string, model: string | null = MODEL.id, deleted = false): void {
  db.run(
    `INSERT INTO documents (id, content_hash, name, kind, size, status, created_at, updated_at,
       deleted_at, embedding_model)
     VALUES (?, ?, ?, 'text', 1, 'ready', ?, ?, ?, ?)`,
    [id, `hash-${id}`, id, NOW, NOW, deleted ? NOW : null, model],
  );
}

/** A Passage of the Document with this vector (or none), at the next position. */
function addPassage(documentId: string, vector: number[] | null, deleted = false): number {
  const seq = nextPassage++;
  db.run(
    `INSERT INTO passages (seq, id, document_id, position, window_from, window_to, text,
       created_at, updated_at, deleted_at, embedding)
     VALUES (?, ?, ?, ?, 0, 0, 'text', ?, ?, ?, ?)`,
    [
      BigInt(seq),
      `p-${seq}`,
      documentId,
      BigInt(seq),
      NOW,
      NOW,
      deleted ? NOW : null,
      vector ? encodeVector(new Float32Array(vector)) : null,
    ],
  );
  return seq;
}

const unit = (vector: number[]) => {
  const norm = Math.hypot(...vector);
  return vector.map((value) => value / norm);
};

const meansOf = (index: ReturnType<typeof createVectorIndex>, dimensions?: number) => {
  const { ids, dimensions: size, data } = index.documentMeans(dimensions);
  return new Map(ids.map((id, at) => [id, Array.from(data.subarray(at * size, (at + 1) * size))]));
};

const expectClose = (actual: number[] | undefined, expected: number[]) => {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, at) => {
    expect(actual?.[at]).toBeCloseTo(value, 6);
  });
};

describe("VectorIndex.documentMeans", () => {
  test("is each Document's L2-normalised mean of its Passage vectors", () => {
    open();
    addDocument("a");
    addPassage("a", unit([1, 0, 0]));
    addPassage("a", unit([0, 1, 0]));
    addDocument("b");
    addPassage("b", unit([0, 0, 1]));
    addPassage("b", unit([0, 3, 4]));
    addPassage("b", unit([0, 0, 1]));
    const means = meansOf(createVectorIndex(db, MODEL));
    expect([...means.keys()].sort()).toEqual(["a", "b"]);
    expectClose(means.get("a"), unit([1, 1, 0]));
    // (0, 0, 1) + (0, 0.6, 0.8) + (0, 0, 1) = (0, 0.6, 2.8)
    expectClose(means.get("b"), unit([0, 0.6, 2.8]));
  });

  test("returns the means one after another in one array, ready to transfer to a worker", () => {
    open();
    for (const id of ["a", "b", "c"]) {
      addDocument(id);
      addPassage(id, unit([1, id === "b" ? 1 : 0]));
    }
    const means = createVectorIndex(db, MODEL).documentMeans();
    expect(means.dimensions).toBe(2);
    expect(means.data).toBeInstanceOf(Float32Array);
    expect(means.data.length).toBe(means.ids.length * 2);
    expect(means.data.buffer.byteLength).toBe(means.data.byteLength);
  });

  test("uses only the current model's vectors of live Passages and Documents", () => {
    open();
    addDocument("current");
    addPassage("current", unit([1, 0]));
    addPassage("current", unit([0, 1]), true); // a deleted Passage
    addDocument("other-model", "another-model");
    addPassage("other-model", unit([1, 0]));
    addDocument("deleted", MODEL.id, true);
    addPassage("deleted", unit([1, 0]));
    const means = meansOf(createVectorIndex(db, MODEL));
    expect([...means.keys()]).toEqual(["current"]);
    expectClose(means.get("current"), [1, 0]);
  });

  test("leaves out Documents with no vectors yet", () => {
    open();
    addDocument("embedded");
    addPassage("embedded", unit([1, 1]));
    addDocument("waiting");
    addPassage("waiting", null);
    addDocument("empty");
    expect([...meansOf(createVectorIndex(db, MODEL)).keys()]).toEqual(["embedded"]);
  });

  test("only includes Documents whose vectors have the given size, by default the most common one", () => {
    open();
    addDocument("small-1");
    addPassage("small-1", unit([1, 0]));
    addDocument("small-2");
    addPassage("small-2", unit([0, 1]));
    addDocument("large");
    addPassage("large", unit([1, 0, 0]));
    const index = createVectorIndex(db, MODEL);
    expect([...meansOf(index).keys()].sort()).toEqual(["small-1", "small-2"]);
    expect([...meansOf(index, 3).keys()]).toEqual(["large"]);
    expect(meansOf(index, 5).size).toBe(0);
  });

  test("loads the index like a first search, and then follows added and removed vectors", () => {
    open();
    addDocument("a");
    addPassage("a", unit([1, 0]));
    const index = createVectorIndex(db, MODEL);
    expectClose(meansOf(index).get("a"), [1, 0]);
    // Embedded after the index was loaded: the core adds it to the index, as the embedding job does.
    index.add("a", 99, new Float32Array(unit([0, 1])));
    expectClose(meansOf(index).get("a"), unit([1, 1]));
    index.removeDocument("a");
    expect(meansOf(index).size).toBe(0);
  });

  test("gives nothing for an empty library", () => {
    open();
    const means = createVectorIndex(db, MODEL).documentMeans();
    expect(means.ids).toEqual([]);
    expect(means.data.length).toBe(0);
  });
});
