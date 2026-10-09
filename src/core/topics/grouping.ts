/**
 * Grouping Documents into Topics (docs/designs/library-structure-view.md,
 * "How Topics are made" and "Keeping it current"; the decision ledger's R1,
 * R2 and O1). Pure functions over L2-normalised Document vectors: no
 * storage, no threads and no model calls, so the grouping worker and the
 * check before building (eval/grouping, #51) run the same code.
 *
 * - k = round(√(N/2)), clamped to 4–40.
 * - Spherical k-means (cosine similarity), started with k-means++, best of 3
 *   seeded runs.
 * - A cluster under 3 Documents is dissolved: each of its Documents joins its
 *   nearest Topic if it passes the placement threshold, or is left ungrouped
 *   ("Not grouped yet").
 * - The placement threshold of a Topic is its members' 10th-percentile
 *   similarity to its centroid; a Topic with fewer than 5 Documents uses the
 *   library-wide 10th percentile instead (O1).
 * - Regroup leaves frozen Topics alone, re-clusters the rest, and matches the
 *   new clusters to the old Topics one-to-one by Jaccard overlap above 0.5.
 */

export const MIN_TOPICS = 4;
export const MAX_TOPICS = 40;
/** Smaller clusters are dissolved. */
export const MIN_TOPIC_SIZE = 3;
/** The placement threshold is this percentile of the members' similarities to their centroid. */
export const PLACEMENT_PERCENTILE = 0.1;
/** A Topic with fewer Documents uses the library-wide threshold (O1). */
export const SMALL_TOPIC_SIZE = 5;
/** A Regroup with fewer Documents to re-cluster only places them (O1). */
export const MIN_REGROUP_POOL = 12;
export const DEFAULT_SEED = 1;
export const DEFAULT_RUNS = 3;
const MAX_ITERATIONS = 100;

/** How many Topics N Documents are grouped into: round(√(N/2)), clamped to 4–40. */
export function topicCount(documents: number): number {
  return Math.min(MAX_TOPICS, Math.max(MIN_TOPICS, Math.round(Math.sqrt(documents / 2))));
}

/** A small seeded generator (mulberry32): the same seed gives the same numbers, from 0 up to 1. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) sum += (a[index] as number) * (b[index] as number);
  return sum;
}

/** The vector scaled to length 1, or null if it has no length. */
export function normalised(vector: ArrayLike<number>): Float32Array | null {
  const norm = Math.sqrt(dot(vector, vector));
  if (!(norm > 0)) return null;
  const result = new Float32Array(vector.length);
  for (let index = 0; index < vector.length; index++) {
    result[index] = (vector[index] as number) / norm;
  }
  return result;
}

/** The direction of the vectors' mean (their normalised mean), or null if there is none. */
export function meanDirection(vectors: readonly ArrayLike<number>[]): Float32Array | null {
  const first = vectors[0];
  if (!first) return null;
  const sum = new Float64Array(first.length);
  for (const vector of vectors) {
    for (let index = 0; index < sum.length; index++) {
      sum[index] = (sum[index] as number) + (vector[index] as number);
    }
  }
  return normalised(sum);
}

/**
 * The vector with the component along a unit `direction` removed, then
 * normalised, or null if nothing is left. The check's third Document vector
 * (R3): a Document's mean with its name's direction taken out.
 */
export function withoutDirection(
  vector: ArrayLike<number>,
  direction: ArrayLike<number>,
): Float32Array | null {
  const along = dot(vector, direction);
  const rest = new Float64Array(vector.length);
  for (let index = 0; index < rest.length; index++) {
    rest[index] = (vector[index] as number) - along * (direction[index] as number);
  }
  // Rounding leaves a tiny remainder of a vector along the direction itself.
  if (Math.sqrt(dot(rest, rest)) < 1e-6) return null;
  return normalised(rest);
}

/** The `p` quantile (0 to 1) of the values, interpolated between the closest ranks; NaN for none. */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const at = p * (sorted.length - 1);
  const below = Math.floor(at);
  const above = Math.ceil(at);
  const low = sorted[below] as number;
  return low + ((sorted[above] as number) - low) * (at - below);
}

export interface KMeansOptions {
  /** Run r starts from seed + r. */
  seed?: number;
  runs?: number;
  maxIterations?: number;
}

export interface KMeansRun {
  seed: number;
  iterations: number;
  /** The sum of each vector's cosine similarity to its centroid: what the best run maximises. */
  similarity: number;
}

export interface KMeansResult {
  k: number;
  /** The cluster of each vector, from 0 to k - 1. */
  assignments: Int32Array;
  /** Unit vectors, one per cluster. */
  centroids: Float32Array[];
  /** The best run's similarity. */
  similarity: number;
  runs: KMeansRun[];
}

/** The vectors one after another, for tight loops. */
function flatten(vectors: readonly ArrayLike<number>[], dimensions: number): Float32Array {
  const data = new Float32Array(vectors.length * dimensions);
  vectors.forEach((vector, index) => {
    data.set(vector, index * dimensions);
  });
  return data;
}

function rowDot(
  data: Float32Array,
  row: number,
  centres: Float64Array | Float32Array,
  centre: number,
  dimensions: number,
): number {
  const a = row * dimensions;
  const b = centre * dimensions;
  let sum = 0;
  for (let d = 0; d < dimensions; d++) sum += (data[a + d] as number) * (centres[b + d] as number);
  return sum;
}

/** k-means++ starts: each next centre is a vector drawn with probability ∝ (1 − its best similarity)². */
function plusPlusStarts(
  data: Float32Array,
  n: number,
  dimensions: number,
  k: number,
  random: () => number,
): Float64Array {
  const centres = new Float64Array(k * dimensions);
  const chosen = new Set<number>();
  const best = new Float64Array(n).fill(Number.NEGATIVE_INFINITY);
  const take = (row: number, centre: number) => {
    chosen.add(row);
    for (let d = 0; d < dimensions; d++) {
      centres[centre * dimensions + d] = data[row * dimensions + d] as number;
    }
    for (let i = 0; i < n; i++) {
      const similarity = rowDot(data, i, centres, centre, dimensions);
      if (similarity > (best[i] as number)) best[i] = similarity;
    }
  };
  take(Math.min(n - 1, Math.floor(random() * n)), 0);
  const weights = new Float64Array(n);
  for (let centre = 1; centre < k; centre++) {
    let total = 0;
    for (let i = 0; i < n; i++) {
      const distance = Math.max(0, 1 - (best[i] as number));
      weights[i] = distance * distance;
      total += weights[i] as number;
    }
    let row = -1;
    if (total > 0) {
      let target = random() * total;
      for (let i = 0; i < n; i++) {
        target -= weights[i] as number;
        if (target < 0 && (weights[i] as number) > 0) {
          row = i;
          break;
        }
      }
      // Rounding can leave the target just above zero: the last vector with weight.
      if (row === -1)
        for (let i = n - 1; i >= 0 && row === -1; i--) if ((weights[i] as number) > 0) row = i;
    } else {
      // Every vector equals a centre already: take one not taken yet.
      for (let i = 0; i < n && row === -1; i++) if (!chosen.has(i)) row = i;
    }
    take(row, centre);
  }
  return centres;
}

/** Each vector to its most similar centre; a tie keeps the current one. Returns how many changed. */
function assign(
  data: Float32Array,
  n: number,
  dimensions: number,
  centres: Float64Array,
  k: number,
  assignments: Int32Array,
  similarities: Float64Array,
): number {
  let changed = 0;
  for (let i = 0; i < n; i++) {
    const current = assignments[i] as number;
    let bestCentre = current;
    let bestSimilarity =
      current >= 0 ? rowDot(data, i, centres, current, dimensions) : Number.NEGATIVE_INFINITY;
    for (let centre = 0; centre < k; centre++) {
      if (centre === current) continue;
      const similarity = rowDot(data, i, centres, centre, dimensions);
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestCentre = centre;
      }
    }
    similarities[i] = bestSimilarity;
    if (bestCentre !== current) {
      assignments[i] = bestCentre;
      changed++;
    }
  }
  return changed;
}

/**
 * Centres as the normalised sums of their vectors. An empty cluster takes the
 * vector least similar to its own centre, from a cluster that can spare one.
 */
function update(
  data: Float32Array,
  n: number,
  dimensions: number,
  centres: Float64Array,
  k: number,
  assignments: Int32Array,
  similarities: Float64Array,
): void {
  const counts = new Int32Array(k);
  const countOf = (i: number) => counts[assignments[i] as number] as number;
  for (let i = 0; i < n; i++) counts[assignments[i] as number] = countOf(i) + 1;
  for (let centre = 0; centre < k; centre++) {
    if ((counts[centre] as number) > 0) continue;
    let row = -1;
    for (let i = 0; i < n; i++) {
      if (countOf(i) < 2) continue;
      if (row === -1 || (similarities[i] as number) < (similarities[row] as number)) row = i;
    }
    if (row === -1) continue;
    counts[assignments[row] as number] = countOf(row) - 1;
    assignments[row] = centre;
    counts[centre] = 1;
    similarities[row] = Number.POSITIVE_INFINITY;
  }
  const sums = new Float64Array(k * dimensions);
  for (let i = 0; i < n; i++) {
    const offset = (assignments[i] as number) * dimensions;
    const row = i * dimensions;
    for (let d = 0; d < dimensions; d++) {
      sums[offset + d] = (sums[offset + d] as number) + (data[row + d] as number);
    }
  }
  for (let centre = 0; centre < k; centre++) {
    const offset = centre * dimensions;
    let norm = 0;
    for (let d = 0; d < dimensions; d++) norm += (sums[offset + d] as number) ** 2;
    norm = Math.sqrt(norm);
    // A cluster that stays empty (fewer distinct vectors than k) keeps its centre.
    if (!(norm > 0)) continue;
    for (let d = 0; d < dimensions; d++) centres[offset + d] = (sums[offset + d] as number) / norm;
  }
}

/**
 * Spherical k-means on unit vectors: k-means++ starts, Lloyd iterations until
 * no vector changes cluster, and the best of `runs` seeded runs (the one with
 * the highest total similarity; the earlier run on a tie). Deterministic for
 * a seed.
 */
export function sphericalKMeans(
  vectors: readonly ArrayLike<number>[],
  k: number,
  options: KMeansOptions = {},
): KMeansResult {
  const n = vectors.length;
  const dimensions = vectors[0]?.length ?? 0;
  if (n === 0 || k < 1) {
    return { k: 0, assignments: new Int32Array(0), centroids: [], similarity: 0, runs: [] };
  }
  const clusters = Math.min(k, n);
  const seed = options.seed ?? DEFAULT_SEED;
  const runCount = Math.max(1, options.runs ?? DEFAULT_RUNS);
  const maxIterations = options.maxIterations ?? MAX_ITERATIONS;
  const data = flatten(vectors, dimensions);
  const runs: KMeansRun[] = [];
  let best: { assignments: Int32Array; centres: Float64Array; similarity: number } | undefined;

  for (let run = 0; run < runCount; run++) {
    const random = seededRandom(seed + run);
    const centres = plusPlusStarts(data, n, dimensions, clusters, random);
    const assignments = new Int32Array(n).fill(-1);
    const similarities = new Float64Array(n);
    assign(data, n, dimensions, centres, clusters, assignments, similarities);
    let iterations = 0;
    let converged = false;
    while (iterations < maxIterations) {
      iterations++;
      update(data, n, dimensions, centres, clusters, assignments, similarities);
      if (assign(data, n, dimensions, centres, clusters, assignments, similarities) === 0) {
        converged = true;
        break;
      }
    }
    // Stopped by the iteration limit: the centres follow the last assignments.
    if (!converged) update(data, n, dimensions, centres, clusters, assignments, similarities);
    let similarity = 0;
    for (let i = 0; i < n; i++) {
      similarity += rowDot(data, i, centres, assignments[i] as number, dimensions);
    }
    runs.push({ seed: seed + run, iterations, similarity });
    if (!best || similarity > best.similarity) best = { assignments, centres, similarity };
  }

  const chosen = best as { assignments: Int32Array; centres: Float64Array; similarity: number };
  const centroids = Array.from({ length: clusters }, (_, centre) =>
    Float32Array.from(chosen.centres.subarray(centre * dimensions, (centre + 1) * dimensions)),
  );
  return {
    k: clusters,
    assignments: chosen.assignments,
    centroids,
    similarity: chosen.similarity,
    runs,
  };
}

/** Where a Document can be placed: a Topic's centroid and its placement threshold. */
export interface PlacementTarget {
  centroid: Float32Array;
  threshold: number;
}

/**
 * One target per Topic, from its members' vectors: its centroid (their
 * normalised mean) and threshold (their 10th-percentile similarity to it, or
 * the library-wide one for a Topic of fewer than 5 Documents). Null for a
 * Topic with no members, which nothing can join.
 */
export function placementTargets(
  topics: readonly (readonly ArrayLike<number>[])[],
): (PlacementTarget | null)[] {
  const centroids = topics.map((members) => meanDirection(members));
  const similarities = topics.map((members, index) => {
    const centroid = centroids[index];
    return centroid ? members.map((vector) => dot(vector, centroid)) : [];
  });
  const libraryWide = percentile(similarities.flat(), PLACEMENT_PERCENTILE);
  return topics.map((members, index) => {
    const centroid = centroids[index];
    if (!centroid) return null;
    const threshold =
      members.length < SMALL_TOPIC_SIZE
        ? libraryWide
        : percentile(similarities[index] as number[], PLACEMENT_PERCENTILE);
    return { centroid, threshold };
  });
}

export interface Placement {
  /** The nearest target's index. */
  index: number;
  similarity: number;
  /** Whether the similarity reaches that target's threshold: if not, the Document isn't grouped. */
  placed: boolean;
}

/** The nearest target (by cosine similarity), and whether the vector may join it; null with no targets. */
export function place(
  vector: ArrayLike<number>,
  targets: readonly (PlacementTarget | null)[],
): Placement | null {
  let index = -1;
  let similarity = Number.NEGATIVE_INFINITY;
  targets.forEach((target, at) => {
    if (!target) return;
    const value = dot(vector, target.centroid);
    if (value > similarity) {
      similarity = value;
      index = at;
    }
  });
  if (index === -1) return null;
  const target = targets[index] as PlacementTarget;
  return { index, similarity, placed: similarity >= target.threshold };
}

/** A Document's vector: L2-normalised. */
export interface DocumentVector {
  id: string;
  vector: Float32Array;
}

export interface Topic {
  /** Document ids, in the order the Documents were given. */
  members: string[];
  /** The members' normalised mean. */
  centroid: Float32Array;
}

export interface GroupingOptions extends KMeansOptions {
  /** At least this many clusters, above `topicCount` (the check uses the number of subjects). */
  minTopics?: number;
}

export interface Grouping {
  k: number;
  topics: Topic[];
  /** Documents of dissolved clusters beyond the placement threshold: "Not grouped yet". */
  ungrouped: string[];
  runs: KMeansRun[];
}

/**
 * Topics from clusters (lists of indices into `documents`): those of 3 or
 * more Documents are kept, and each Document of a smaller one joins its
 * nearest kept Topic if it passes that Topic's placement threshold, or is
 * left ungrouped. Thresholds come from the kept clusters before anything
 * joins them, so the order of the dissolved Documents doesn't matter.
 */
export function formTopics(
  documents: readonly DocumentVector[],
  clusters: readonly (readonly number[])[],
): { topics: Topic[]; ungrouped: string[] } {
  const vectorOf = (index: number) => (documents[index] as DocumentVector).vector;
  const kept = clusters.filter((cluster) => cluster.length >= MIN_TOPIC_SIZE).map((c) => [...c]);
  const dissolved = clusters
    .filter((cluster) => cluster.length > 0 && cluster.length < MIN_TOPIC_SIZE)
    .flat()
    .sort((a, b) => a - b);
  const targets = placementTargets(kept.map((cluster) => cluster.map(vectorOf)));
  const ungrouped: string[] = [];
  for (const index of dissolved) {
    const placement = place(vectorOf(index), targets);
    if (placement?.placed) (kept[placement.index] as number[]).push(index);
    else ungrouped.push((documents[index] as DocumentVector).id);
  }
  const topics = kept.map((cluster) => {
    const members = [...cluster].sort((a, b) => a - b);
    return {
      members: members.map((index) => (documents[index] as DocumentVector).id),
      centroid: meanDirection(members.map(vectorOf)) as Float32Array,
    };
  });
  return { topics, ungrouped };
}

/** Groups Documents into Topics: step 1 of "How Topics are made". */
export function groupDocuments(
  documents: readonly DocumentVector[],
  options: GroupingOptions = {},
): Grouping {
  const n = documents.length;
  if (n === 0) return { k: 0, topics: [], ungrouped: [], runs: [] };
  const k = Math.min(n, Math.max(topicCount(n), options.minTopics ?? 0));
  const result = sphericalKMeans(
    documents.map((document) => document.vector),
    k,
    options,
  );
  const clusters: number[][] = Array.from({ length: result.k }, () => []);
  result.assignments.forEach((cluster, index) => {
    (clusters[cluster] as number[]).push(index);
  });
  return { k: result.k, ...formTopics(documents, clusters), runs: result.runs };
}

/**
 * New clusters matched one-to-one to old Topics ("Matching child Topics after
 * a Regroup"): pairs are taken in order of highest Jaccard overlap, a pair
 * only if its overlap is strictly greater than 0.5, and each cluster and old
 * Topic is used once. On equal overlaps the old Topic created earlier (first
 * in `old`) wins. Returns, for each cluster, its old Topic's index or -1.
 */
export function matchTopics(
  old: readonly (readonly string[])[],
  clusters: readonly (readonly string[])[],
): number[] {
  const oldSets = old.map((members) => new Set(members));
  const pairs: { cluster: number; old: number; overlap: number }[] = [];
  clusters.forEach((members, cluster) => {
    const set = new Set(members);
    oldSets.forEach((oldSet, index) => {
      let shared = 0;
      for (const id of set) if (oldSet.has(id)) shared++;
      const union = set.size + oldSet.size - shared;
      const overlap = union === 0 ? 0 : shared / union;
      if (overlap > 0.5) pairs.push({ cluster, old: index, overlap });
    });
  });
  pairs.sort((a, b) => b.overlap - a.overlap || a.old - b.old || a.cluster - b.cluster);
  const matched = clusters.map(() => -1);
  const used = new Set<number>();
  for (const pair of pairs) {
    if (matched[pair.cluster] !== -1 || used.has(pair.old)) continue;
    matched[pair.cluster] = pair.old;
    used.add(pair.old);
  }
  return matched;
}

/** A Topic as it is before a Regroup. */
export interface CurrentTopic {
  id: string;
  members: readonly string[];
  /** Renamed or created by the User, or a Document was moved into it: left out of re-clustering. */
  frozen: boolean;
}

export interface RegroupInput {
  /** Every Document that has a vector, L2-normalised. Others can't be grouped. */
  vectors: ReadonlyMap<string, Float32Array>;
  /** The current Topics, oldest first. */
  topics: readonly CurrentTopic[];
  /** The Documents in "Not grouped yet". */
  ungrouped: readonly string[];
  /** Documents the User moved: they keep their Topic, which counts as frozen. */
  moved?: ReadonlySet<string>;
  /** An id for each Topic a Regroup makes. */
  newTopicId: () => string;
  options?: GroupingOptions;
}

export interface RegroupedTopic {
  id: string;
  members: string[];
  /** An old Topic's id (frozen or matched): it keeps its name, with no naming call. */
  kept: boolean;
  frozen: boolean;
}

export interface Regrouping {
  /** Frozen Topics first, then the other kept ones, in their old order, then new ones. */
  topics: RegroupedTopic[];
  /** Old Topics no new cluster matched. */
  removed: string[];
  ungrouped: string[];
  /** False when the pool was too small to re-cluster (O1): it was only placed. */
  reclustered: boolean;
  /** The k the pool was clustered with, or 0. */
  k: number;
}

/**
 * Regroup ("What a Regroup re-clusters"):
 * 1. Documents in "Not grouped yet" that have vectors are placed into frozen
 *    Topics, by the placement threshold.
 * 2. The rest of them, with every Document of a Topic that isn't frozen, make
 *    the pool. With 12 or more, the pool is grouped (k from its size), Documents
 *    of dissolved clusters may join a frozen Topic too, and the new clusters are
 *    matched to the old open Topics. With fewer (O1), nothing is re-clustered:
 *    open Topics keep their members and the ungrouped are placed into any Topic.
 * Members without a vector, such as a Document being processed again, can't
 * be re-clustered: from an open Topic they go to "Not grouped yet".
 */
export function regroup(input: RegroupInput): Regrouping {
  const { vectors, topics, newTopicId } = input;
  const moved = input.moved ?? new Set<string>();
  const options = input.options ?? {};
  const vectorOf = (id: string) => vectors.get(id) as Float32Array;
  const withVectors = (ids: readonly string[]) => ids.filter((id) => vectors.has(id));
  const isFrozen = topics.map((topic) => topic.frozen || topic.members.some((id) => moved.has(id)));
  const members = topics.map((topic) => [...topic.members]);
  const ungrouped: string[] = [];

  // 1. Into frozen Topics, with thresholds from the library as it is.
  const targetsNow = () => placementTargets(members.map((list) => withVectors(list).map(vectorOf)));
  const frozenTargets = targetsNow().map((target, index) => (isFrozen[index] ? target : null));
  const loose: string[] = [];
  for (const id of input.ungrouped) {
    if (!vectors.has(id)) {
      ungrouped.push(id);
      continue;
    }
    const placement = place(vectorOf(id), frozenTargets);
    if (placement?.placed) (members[placement.index] as string[]).push(id);
    else loose.push(id);
  }

  // 2. The pool.
  const open = topics.map((_, index) => index).filter((index) => !isFrozen[index]);
  const pool: string[] = [];
  for (const index of open) {
    for (const id of members[index] as string[]) {
      if (vectors.has(id)) pool.push(id);
      else ungrouped.push(id);
    }
  }
  pool.push(...loose);

  const frozenTopics = (): RegroupedTopic[] =>
    topics.flatMap((topic, index) =>
      isFrozen[index]
        ? [{ id: topic.id, members: members[index] as string[], kept: true, frozen: true }]
        : [],
    );

  if (pool.length < MIN_REGROUP_POOL) {
    const targets = targetsNow();
    for (const id of loose) {
      const placement = place(vectorOf(id), targets);
      if (placement?.placed) (members[placement.index] as string[]).push(id);
      else ungrouped.push(id);
    }
    const kept: RegroupedTopic[] = [];
    const removed: string[] = [];
    for (const index of open) {
      const topic = topics[index] as CurrentTopic;
      const list = withVectors(members[index] as string[]);
      if (list.length === 0) removed.push(topic.id);
      else kept.push({ id: topic.id, members: list, kept: true, frozen: false });
    }
    return { topics: [...frozenTopics(), ...kept], removed, ungrouped, reclustered: false, k: 0 };
  }

  const grouping = groupDocuments(
    pool.map((id) => ({ id, vector: vectorOf(id) })),
    options,
  );
  // Documents of dissolved clusters may still fit a frozen Topic.
  const frozenNow = targetsNow().map((target, index) => (isFrozen[index] ? target : null));
  for (const id of grouping.ungrouped) {
    const placement = place(vectorOf(id), frozenNow);
    if (placement?.placed) (members[placement.index] as string[]).push(id);
    else ungrouped.push(id);
  }
  const clusters = grouping.topics.map((topic) => topic.members);
  const matches = matchTopics(
    open.map((index) => (topics[index] as CurrentTopic).members),
    clusters,
  );
  const matchedTopics: (RegroupedTopic | undefined)[] = open.map(() => undefined);
  const created: RegroupedTopic[] = [];
  clusters.forEach((cluster, at) => {
    const match = matches[at] as number;
    if (match === -1) {
      created.push({ id: newTopicId(), members: cluster, kept: false, frozen: false });
    } else {
      const old = topics[open[match] as number] as CurrentTopic;
      matchedTopics[match] = { id: old.id, members: cluster, kept: true, frozen: false };
    }
  });
  const removed = open.flatMap((index, at) =>
    matchedTopics[at] ? [] : [(topics[index] as CurrentTopic).id],
  );
  return {
    topics: [
      ...frozenTopics(),
      ...matchedTopics.filter((topic): topic is RegroupedTopic => topic !== undefined),
      ...created,
    ],
    removed,
    ungrouped,
    reclustered: true,
    k: grouping.k,
  };
}
