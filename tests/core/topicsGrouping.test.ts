/**
 * The pure grouping functions (src/core/topics/grouping.ts): k from the
 * Document count, spherical k-means with k-means++ starts and seeded runs,
 * dissolving small clusters, the placement threshold with O1's small-Topic
 * rule, and Regroup's matching and freeze rules
 * (docs/designs/library-structure-view.md).
 */
import { describe, expect, test } from "vitest";
import {
  type DocumentVector,
  dot,
  formTopics,
  groupDocuments,
  matchTopics,
  meanDirection,
  normalised,
  percentile,
  place,
  placementTargets,
  regroup,
  seededRandom,
  sphericalKMeans,
  topicCount,
  withoutDirection,
} from "../../src/core/topics/grouping";

const DIMENSIONS = 16;

/** A unit vector along axis `axis`, nudged by seeded noise of size `noise`. */
function near(axis: number, noise: number, random: () => number, dimensions = DIMENSIONS) {
  const vector = new Float32Array(dimensions);
  for (let d = 0; d < dimensions; d++) vector[d] = (random() - 0.5) * 2 * noise;
  vector[axis] = (vector[axis] as number) + 1;
  return normalised(vector) as Float32Array;
}

/** `sizes[i]` Documents around axis i, with ids "c<i>-<n>". */
function clusters(sizes: readonly number[], noise = 0.15, seed = 7): DocumentVector[] {
  const random = seededRandom(seed);
  return sizes.flatMap((size, axis) =>
    Array.from({ length: size }, (_, n) => ({
      id: `c${axis}-${n}`,
      vector: near(axis, noise, random),
    })),
  );
}

const clusterOf = (id: string) => id.split("-")[0];

/** The Topics as sorted lists of ids, sorted, for comparing groupings. */
const memberLists = (topics: readonly { members: readonly string[] }[]) =>
  topics
    .map((topic) => [...topic.members].sort())
    .sort((a, b) => (a[0] ?? "").localeCompare(b[0] ?? ""));

describe("topicCount", () => {
  test("is round(√(N/2)), clamped to 4–40", () => {
    expect(topicCount(0)).toBe(4);
    expect(topicCount(20)).toBe(4);
    expect(topicCount(40)).toBe(4); // √20 = 4.47
    expect(topicCount(41)).toBe(5); // √20.5 = 4.53
    expect(topicCount(200)).toBe(10);
    expect(topicCount(800)).toBe(20);
    expect(topicCount(3200)).toBe(40);
    expect(topicCount(5000)).toBe(40);
  });
});

describe("vectors", () => {
  test("normalised gives a unit vector, or null for a zero vector", () => {
    const unit = normalised([3, 4]) as Float32Array;
    expect(Array.from(unit)).toEqual([0.6000000238418579, 0.800000011920929]);
    expect(normalised([0, 0])).toBeNull();
  });

  test("meanDirection is the normalised mean, null with nothing to average", () => {
    const mean = meanDirection([
      new Float32Array([1, 0]),
      new Float32Array([0, 1]),
    ]) as Float32Array;
    expect(mean[0]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(mean[1]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(meanDirection([])).toBeNull();
    expect(meanDirection([new Float32Array([1, 0]), new Float32Array([-1, 0])])).toBeNull();
  });

  test("withoutDirection removes a direction and normalises what is left (R3's third variant)", () => {
    const vector = normalised([1, 1, 0]) as Float32Array;
    const name = new Float32Array([1, 0, 0]);
    const left = withoutDirection(vector, name) as Float32Array;
    expect(dot(left, name)).toBeCloseTo(0, 6);
    expect(Array.from(left).map((value) => Number(value.toFixed(6)))).toEqual([0, 1, 0]);
    // Nothing is left of a vector along the direction itself.
    expect(withoutDirection(name, name)).toBeNull();
  });

  test("percentile interpolates between the closest ranks", () => {
    expect(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 0.1)).toBeCloseTo(2, 10);
    expect(percentile([10, 0], 0.1)).toBeCloseTo(1, 10);
    expect(percentile([0.4], 0.1)).toBe(0.4);
    expect(percentile([], 0.1)).toBeNaN();
  });

  test("seededRandom repeats its sequence for a seed, and differs between seeds", () => {
    const a = seededRandom(1);
    const b = seededRandom(1);
    const c = seededRandom(2);
    const first = [a(), a(), a()];
    expect([b(), b(), b()]).toEqual(first);
    expect([c(), c(), c()]).not.toEqual(first);
    for (const value of first) expect(value >= 0 && value < 1).toBe(true);
  });
});

describe("sphericalKMeans", () => {
  test("finds well-separated clusters, whatever the seed", () => {
    const documents = clusters([6, 6, 6, 6]);
    for (const seed of [1, 2, 3]) {
      const result = sphericalKMeans(
        documents.map((document) => document.vector),
        4,
        { seed },
      );
      const byCluster = new Map<number, Set<string>>();
      result.assignments.forEach((cluster, index) => {
        const set = byCluster.get(cluster) ?? new Set<string>();
        set.add(clusterOf((documents[index] as DocumentVector).id) as string);
        byCluster.set(cluster, set);
      });
      expect(byCluster.size).toBe(4);
      for (const set of byCluster.values()) expect(set.size).toBe(1);
    }
  });

  test("runs 3 seeded starts and keeps the best; the same seed gives the same result", () => {
    const vectors = clusters([5, 5, 5, 5, 5], 0.6).map((document) => document.vector);
    const first = sphericalKMeans(vectors, 5, { seed: 11 });
    const again = sphericalKMeans(vectors, 5, { seed: 11 });
    expect(first.runs.map((run) => run.seed)).toEqual([11, 12, 13]);
    expect(Array.from(again.assignments)).toEqual(Array.from(first.assignments));
    expect(first.similarity).toBe(Math.max(...first.runs.map((run) => run.similarity)));
    for (const centroid of first.centroids) expect(dot(centroid, centroid)).toBeCloseTo(1, 5);
  });

  test("its similarity is the sum of each vector's similarity to its centroid", () => {
    const vectors = clusters([4, 4, 4], 0.3).map((document) => document.vector);
    const result = sphericalKMeans(vectors, 3, { seed: 5, runs: 1 });
    const sum = vectors.reduce(
      (total, vector, index) =>
        total + dot(vector, result.centroids[result.assignments[index] as number] as Float32Array),
      0,
    );
    expect(result.similarity).toBeCloseTo(sum, 4);
  });

  test("never leaves a cluster empty, even with repeated vectors", () => {
    const same = normalised([1, 0, 0, 0]) as Float32Array;
    const vectors = [same, same, same, same, normalised([0, 1, 0, 0]) as Float32Array];
    const result = sphericalKMeans(vectors, 3, { seed: 1 });
    const used = new Set(result.assignments);
    expect(used.size).toBe(3);
  });

  test("k equal to the number of vectors gives each its own cluster", () => {
    const vectors = clusters([1, 1, 1]).map((document) => document.vector);
    const result = sphericalKMeans(vectors, 3, { seed: 1 });
    expect(new Set(result.assignments).size).toBe(3);
  });
});

describe("placementTargets and place", () => {
  test("a Topic's threshold is its 10th-percentile member similarity to its centroid", () => {
    const documents = clusters([10, 10], 0.4);
    const topic = documents.filter((document) => document.id.startsWith("c0-"));
    const [target] = placementTargets([
      topic.map((document) => document.vector),
      documents
        .filter((document) => document.id.startsWith("c1-"))
        .map((document) => document.vector),
    ]);
    const centroid = meanDirection(topic.map((document) => document.vector)) as Float32Array;
    const similarities = topic.map((document) => dot(document.vector, centroid));
    expect(target?.threshold).toBeCloseTo(percentile(similarities, 0.1), 6);
    expect(Array.from(target?.centroid ?? [])).toEqual(Array.from(centroid));
  });

  test("a Topic with fewer than 5 Documents uses the library-wide 10th percentile (O1)", () => {
    const documents = clusters([10, 4], 0.4);
    const big = documents.filter((document) => document.id.startsWith("c0-")).map((d) => d.vector);
    const small = documents
      .filter((document) => document.id.startsWith("c1-"))
      .map((d) => d.vector);
    const [bigTarget, smallTarget] = placementTargets([big, small]);
    const all = [big, small].flatMap((members) => {
      const centroid = meanDirection(members) as Float32Array;
      return members.map((vector) => dot(vector, centroid));
    });
    expect(smallTarget?.threshold).toBeCloseTo(percentile(all, 0.1), 6);
    expect(bigTarget?.threshold).not.toBeCloseTo(smallTarget?.threshold as number, 6);
    // A one-Document Topic doesn't get a threshold of 1, which nothing could pass.
    const [, single] = placementTargets([big, [small[0] as Float32Array]]);
    expect(single?.threshold).toBeLessThan(0.999);
  });

  test("an empty Topic has no target", () => {
    expect(placementTargets([[]])).toEqual([null]);
  });

  test("a Document joins its nearest Topic only at or above that Topic's threshold", () => {
    const documents = clusters([8, 8], 0.2);
    const targets = placementTargets([
      documents.slice(0, 8).map((document) => document.vector),
      documents.slice(8).map((document) => document.vector),
    ]);
    const random = seededRandom(99);
    const close = place(near(1, 0.05, random), targets);
    expect(close).toMatchObject({ index: 1, placed: true });
    // Halfway between both Topics: nearest to one, but beyond its threshold.
    const between = normalised([1, 1, ...new Array(DIMENSIONS - 2).fill(0)]) as Float32Array;
    const far = place(between, targets);
    expect(far?.placed).toBe(false);
    // Exactly at the threshold counts.
    const target = targets[0] as { centroid: Float32Array; threshold: number };
    expect(place(target.centroid, [{ centroid: target.centroid, threshold: 1 }])?.placed).toBe(
      true,
    );
    expect(place(target.centroid, [null])).toBeNull();
  });
});

describe("groupDocuments", () => {
  test("groups Documents with k from their count, and keeps every cluster of 3 or more", () => {
    const documents = clusters([5, 5, 5, 5]);
    const grouping = groupDocuments(documents, { seed: 1 });
    expect(grouping.k).toBe(4); // topicCount(20)
    expect(grouping.ungrouped).toEqual([]);
    expect(memberLists(grouping.topics)).toEqual(
      ["c0", "c1", "c2", "c3"].map((prefix) =>
        documents
          .filter((document) => clusterOf(document.id) === prefix)
          .map((d) => d.id)
          .sort(),
      ),
    );
    for (const topic of grouping.topics)
      expect(dot(topic.centroid, topic.centroid)).toBeCloseTo(1, 5);
  });

  test("minTopics raises k (the check uses at least as many clusters as subjects)", () => {
    const documents = clusters([4, 4, 4, 4, 4, 4]);
    expect(groupDocuments(documents, { seed: 1 }).k).toBe(4);
    const grouping = groupDocuments(documents, { seed: 1, minTopics: 6 });
    expect(grouping.k).toBe(6);
    expect(grouping.topics).toHaveLength(6);
  });

  test("never asks for more clusters than Documents", () => {
    expect(groupDocuments(clusters([1, 1]), { seed: 1 }).k).toBe(2);
    expect(groupDocuments([], { seed: 1 })).toMatchObject({ k: 0, topics: [], ungrouped: [] });
  });

  test("dissolves clusters under 3 Documents: each joins its nearest Topic if it passes the threshold, else goes ungrouped", () => {
    const random = seededRandom(3);
    // Two big clusters on axes 0 and 1; a pair on axis 2 (far from both); a pair close to axis 0.
    const documents: DocumentVector[] = [
      ...clusters([8, 8], 0.15),
      { id: "far-0", vector: near(2, 0.05, random) },
      { id: "far-1", vector: near(2, 0.05, random) },
    ];
    const grouping = groupDocuments(documents, { seed: 1, minTopics: 4 });
    expect(grouping.k).toBe(4);
    // The far pair is a cluster of 2: dissolved, and too far from both Topics to join them.
    expect(grouping.ungrouped).toEqual(expect.arrayContaining(["far-0", "far-1"]));
    expect(grouping.topics.every((topic) => topic.members.length >= 3)).toBe(true);
    const grouped = grouping.topics.flatMap((topic) => topic.members);
    expect(new Set([...grouped, ...grouping.ungrouped]).size).toBe(documents.length);
  });

  test("is deterministic for a seed", () => {
    const documents = clusters([6, 5, 7, 4], 0.5);
    const a = groupDocuments(documents, { seed: 9 });
    const b = groupDocuments(documents, { seed: 9 });
    expect(memberLists(a.topics)).toEqual(memberLists(b.topics));
    expect(a.ungrouped).toEqual(b.ungrouped);
  });
});

describe("matchTopics", () => {
  test("matches one-to-one by highest Jaccard overlap, only above 0.5", () => {
    const old = [
      ["a", "b", "c", "d"],
      ["e", "f", "g", "h"],
      ["i", "j"],
    ];
    const clusters = [
      ["e", "f", "g"], // 3/4 with old 1
      ["a", "b", "c", "d", "x"], // 4/5 with old 0
      ["i", "k"], // 1/3 with old 2: not above 0.5
    ];
    expect(matchTopics(old, clusters)).toEqual([1, 0, -1]);
  });

  test("an overlap of exactly 0.5 doesn't match", () => {
    expect(matchTopics([["a", "b"]], [["a", "c", "b", "d"]])).toEqual([-1]);
    expect(matchTopics([["a", "b"]], [["a", "b", "c"]])).toEqual([0]);
  });

  test("each old Topic is used once: the higher overlap wins", () => {
    // Both clusters overlap old 0 above 0.5; the second overlaps it more.
    const old = [["a", "b", "c", "d", "e", "f"]];
    const clusters = [
      ["a", "b", "c", "d"], // 4/6
      ["a", "b", "c", "d", "e"], // 5/6
    ];
    expect(matchTopics(old, clusters)).toEqual([-1, 0]);
  });

  test("ties go to the old Topic created earlier", () => {
    // The cluster overlaps both old Topics by 3/4.
    const old = [
      ["a", "b", "c"],
      ["a", "b", "c", "d"],
    ];
    expect(matchTopics(old, [["a", "b", "c", "d"]])).toEqual([1]);
    const tied = [
      ["a", "b", "c", "x"],
      ["a", "b", "c", "y"],
    ];
    expect(matchTopics(tied, [["a", "b", "c"]])).toEqual([0]);
  });
});

describe("regroup", () => {
  const vectorsOf = (documents: readonly DocumentVector[]) =>
    new Map(documents.map((document) => [document.id, document.vector]));

  function counter(prefix = "new") {
    let next = 1;
    return () => `${prefix}-${next++}`;
  }

  test("frozen Topics keep their Documents, and Documents the User moved keep their Topic", () => {
    const documents = clusters([6, 6, 6, 6, 6]);
    const ids = (prefix: string) =>
      documents.filter((document) => clusterOf(document.id) === prefix).map((d) => d.id);
    // The User renamed t-0 (frozen) and moved c3-0 into it; t-1 holds a misplaced c2 Document.
    const result = regroup({
      vectors: vectorsOf(documents),
      topics: [
        { id: "t-0", members: [...ids("c0"), "c3-0"], frozen: true },
        { id: "t-1", members: [...ids("c1"), "c2-0"], frozen: false },
        { id: "t-2", members: ids("c2").slice(1), frozen: false },
        { id: "t-3", members: ids("c3").slice(1), frozen: false },
        { id: "t-4", members: ids("c4"), frozen: false },
      ],
      ungrouped: [],
      moved: new Set(["c3-0"]),
      newTopicId: counter(),
      options: { seed: 1 },
    });
    const t0 = result.topics.find((topic) => topic.id === "t-0");
    expect(t0).toMatchObject({ kept: true, frozen: true });
    expect([...(t0?.members ?? [])].sort()).toEqual([...ids("c0"), "c3-0"].sort());
    // The others were re-clustered: c2-0 is back with its own kind, and matching kept the ids.
    const t1 = result.topics.find((topic) => topic.id === "t-1");
    const t2 = result.topics.find((topic) => topic.id === "t-2");
    expect([...(t1?.members ?? [])].sort()).toEqual(ids("c1").sort());
    expect([...(t2?.members ?? [])].sort()).toEqual(ids("c2").sort());
    expect(t1?.kept && t2?.kept).toBe(true);
    expect(result.topics.map((topic) => topic.id).sort()).toEqual([
      "t-0",
      "t-1",
      "t-2",
      "t-3",
      "t-4",
    ]);
    expect(result.removed).toEqual([]);
    expect(result.reclustered).toBe(true);
  });

  test("a Topic holding a Document the User moved counts as frozen", () => {
    const documents = clusters([6, 6, 6]);
    const result = regroup({
      vectors: vectorsOf(documents),
      topics: [
        {
          id: "t-0",
          members: documents
            .filter((d) => d.id.startsWith("c0-"))
            .map((d) => d.id)
            .concat("c1-0"),
          frozen: false,
        },
        {
          id: "t-1",
          members: documents
            .filter((d) => /^c[12]-/.test(d.id) && d.id !== "c1-0")
            .map((d) => d.id),
          frozen: false,
        },
      ],
      ungrouped: [],
      moved: new Set(["c1-0"]),
      newTopicId: counter(),
      options: { seed: 1 },
    });
    const t0 = result.topics.find((topic) => topic.id === "t-0");
    expect(t0?.frozen).toBe(true);
    expect(t0?.members).toContain("c1-0");
  });

  test("Documents in Not grouped yet are first placed into frozen Topics by the threshold", () => {
    const random = seededRandom(21);
    const documents = clusters([8, 8, 8]);
    const late: DocumentVector = { id: "late-c0", vector: near(0, 0.05, random) };
    const result = regroup({
      vectors: vectorsOf([...documents, late]),
      topics: [
        {
          id: "t-0",
          members: documents.filter((d) => d.id.startsWith("c0-")).map((d) => d.id),
          frozen: true,
        },
        {
          id: "t-1",
          members: documents.filter((d) => !d.id.startsWith("c0-")).map((d) => d.id),
          frozen: false,
        },
      ],
      ungrouped: ["late-c0", "no-vector"],
      newTopicId: counter(),
      options: { seed: 1 },
    });
    expect(result.topics.find((topic) => topic.id === "t-0")?.members).toContain("late-c0");
    // A Document without a vector stays in Not grouped yet.
    expect(result.ungrouped).toContain("no-vector");
    expect(result.ungrouped).not.toContain("late-c0");
  });

  test("unmatched clusters get new ids, unmatched old Topics are removed", () => {
    const documents = clusters([6, 6, 6, 6]);
    const all = documents.map((document) => document.id);
    // One old Topic held everything: no new cluster overlaps it by more than 0.5.
    const result = regroup({
      vectors: vectorsOf(documents),
      topics: [{ id: "old", members: all, frozen: false }],
      ungrouped: [],
      newTopicId: counter(),
      options: { seed: 1 },
    });
    expect(result.removed).toEqual(["old"]);
    expect(result.topics.map((topic) => topic.id).sort()).toEqual([
      "new-1",
      "new-2",
      "new-3",
      "new-4",
    ]);
    expect(result.topics.every((topic) => !topic.kept)).toBe(true);
    expect(memberLists(result.topics)).toEqual(
      ["c0", "c1", "c2", "c3"].map((prefix) => all.filter((id) => clusterOf(id) === prefix).sort()),
    );
  });

  test("with fewer than 12 Documents to re-cluster, it only places them into existing Topics (O1)", () => {
    const random = seededRandom(31);
    const documents = clusters([8, 8, 8]);
    const loose: DocumentVector[] = [
      { id: "loose-c1", vector: near(1, 0.05, random) },
      { id: "loose-far", vector: near(5, 0.05, random) },
    ];
    const ids = (prefix: string) =>
      documents.filter((d) => d.id.startsWith(prefix)).map((d) => d.id);
    const result = regroup({
      vectors: vectorsOf([...documents, ...loose]),
      topics: [
        { id: "t-0", members: ids("c0-"), frozen: true },
        { id: "t-1", members: ids("c1-"), frozen: true },
        // The only open Topic: 8 Documents plus 2 loose ones make a pool of 10.
        { id: "t-2", members: ids("c2-"), frozen: false },
      ],
      ungrouped: ["loose-c1", "loose-far"],
      newTopicId: counter(),
      options: { seed: 1 },
    });
    expect(result.reclustered).toBe(false);
    expect(result.topics.map((topic) => topic.id)).toEqual(["t-0", "t-1", "t-2"]);
    expect(result.topics.find((topic) => topic.id === "t-1")?.members).toContain("loose-c1");
    expect([...(result.topics.find((topic) => topic.id === "t-2")?.members ?? [])].sort()).toEqual(
      ids("c2-").sort(),
    );
    expect(result.ungrouped).toEqual(["loose-far"]);
  });

  test("with a small pool, open Topics keep every member, even those beyond the threshold", () => {
    // Noisy enough that some members are below their Topic's 10th percentile.
    const documents = clusters([8, 8], 0.8);
    const ids = (prefix: string) =>
      documents.filter((d) => d.id.startsWith(prefix)).map((d) => d.id);
    const result = regroup({
      vectors: vectorsOf(documents),
      topics: [
        { id: "t-0", members: ids("c0-"), frozen: true },
        { id: "t-1", members: ids("c1-"), frozen: false },
      ],
      ungrouped: [],
      newTopicId: counter(),
      options: { seed: 1 },
    });
    expect(result.reclustered).toBe(false);
    expect(result.topics.find((topic) => topic.id === "t-1")?.members).toEqual(ids("c1-"));
    expect(result.ungrouped).toEqual([]);
  });
});

describe("formTopics", () => {
  test("keeps clusters of 3 or more; a smaller one's Documents join their nearest Topic or go ungrouped", () => {
    const random = seededRandom(41);
    const documents: DocumentVector[] = [
      ...clusters([8, 8], 0.2),
      { id: "near-c0", vector: near(0, 0.02, random) },
      { id: "far", vector: near(5, 0.02, random) },
    ];
    const index = (id: string) => documents.findIndex((document) => document.id === id);
    const c0 = documents.filter((d) => d.id.startsWith("c0-")).map((d) => index(d.id));
    const c1 = documents.filter((d) => d.id.startsWith("c1-")).map((d) => index(d.id));
    const { topics, ungrouped } = formTopics(documents, [c0, c1, [index("near-c0"), index("far")]]);
    expect(topics).toHaveLength(2);
    expect(topics[0]?.members).toEqual([...c0.map((i) => documents[i]?.id), "near-c0"]);
    expect(topics[1]?.members).toEqual(c1.map((i) => documents[i]?.id));
    expect(ungrouped).toEqual(["far"]);
    // The centroid includes the Document that joined.
    const members = [
      ...c0.map((i) => (documents[i] as DocumentVector).vector),
      (documents[index("near-c0")] as DocumentVector).vector,
    ];
    expect(Array.from(topics[0]?.centroid ?? [])).toEqual(Array.from(meanDirection(members) ?? []));
  });
});
