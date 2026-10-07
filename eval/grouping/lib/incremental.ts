/**
 * The grouping check's incremental cases (O8a): Documents added after the
 * grouping are placed by the threshold, and the User's corrections followed
 * by a Regroup are kept ("Keeping it current, and the User's corrections").
 * Pure: it works on the Documents' vectors, by key.
 */
import {
  DEFAULT_SEED,
  groupDocuments,
  place,
  placementTargets,
  regroup,
} from "../../../src/core/topics/grouping";
import { type CasesScore, scoreCases, topicsByKey } from "./scoring";
import type { GroupingSet, SetDocument } from "./set";

export type Arrival = "with its subject" | "not grouped yet" | "with another subject";

export interface LateArrivalResult {
  key: string;
  /** Whether a Document on its subject was grouped first: then it should join one, else stay out. */
  expected: "joins its subject" | "stays out";
  outcome: Arrival;
  correct: boolean;
  /** Its similarity to the nearest Topic's centroid, and that Topic's threshold. */
  similarity: number | null;
  threshold: number | null;
}

export interface CorrectionCheck {
  description: string;
  kept: boolean;
}

export interface IncrementalResult {
  initial: { documents: number; k: number; topics: number; ungrouped: number };
  lateArrivals: LateArrivalResult[];
  /** What the User did, in order. */
  corrections: string[];
  regroup: {
    pool: number;
    reclustered: boolean;
    k: number;
    /** Old Topics kept: frozen, or matched by overlap. */
    kept: number;
    created: number;
    removed: number;
    ungrouped: number;
  };
  /** Whether each of the User's changes survived the Regroup. */
  checks: CorrectionCheck[];
  /** The pairs and decks after the Regroup, for information. */
  afterRegroup: { choosing: CasesScore; heldOut: CasesScore };
}

interface TopicState {
  id: string;
  name: string | null;
  members: string[];
  frozen: boolean;
  /** Why it is frozen, for the report. */
  why: string[];
  userCreated: boolean;
}

const subjectCount = (documents: readonly SetDocument[]) =>
  new Set(documents.map((document) => document.subject)).size;

export function runIncremental(
  set: GroupingSet,
  vectors: ReadonlyMap<string, Float32Array>,
  options: { seed?: number } = {},
): IncrementalResult {
  const seed = options.seed ?? DEFAULT_SEED;
  const byKey = new Map(set.documents.map((document) => [document.key, document]));
  const vectorOf = (key: string) => vectors.get(key) as Float32Array;
  const late = set.lateArrivals.filter((key) => vectors.has(key));
  const lateSet = new Set(late);
  const initial = set.documents.filter(
    (document) => vectors.has(document.key) && !lateSet.has(document.key),
  );

  // The first grouping, without the late arrivals.
  const grouping = groupDocuments(
    initial.map((document) => ({ id: document.key, vector: vectorOf(document.key) })),
    { seed, minTopics: subjectCount(initial) },
  );
  const topics: TopicState[] = grouping.topics.map((topic, index) => ({
    id: `topic-${index + 1}`,
    name: null,
    members: [...topic.members],
    frozen: false,
    why: [],
    userCreated: false,
  }));
  const ungrouped = [...grouping.ungrouped];

  // Late arrivals, each placed by the threshold of the Topics as first grouped.
  const targets = placementTargets(topics.map((topic) => topic.members.map(vectorOf)));
  const lateArrivals = late.map((key): LateArrivalResult => {
    const subject = byKey.get(key)?.subject;
    const expected = initial.some((document) => document.subject === subject)
      ? "joins its subject"
      : "stays out";
    const placement = place(vectorOf(key), targets);
    let outcome: Arrival = "not grouped yet";
    if (placement?.placed) {
      const topic = topics[placement.index] as TopicState;
      outcome = topic.members.some((member) => byKey.get(member)?.subject === subject)
        ? "with its subject"
        : "with another subject";
      topic.members.push(key);
    } else {
      ungrouped.push(key);
    }
    return {
      key,
      expected,
      outcome,
      correct:
        expected === "joins its subject"
          ? outcome === "with its subject"
          : outcome === "not grouped yet",
      similarity: placement?.similarity ?? null,
      threshold: placement ? (targets[placement.index]?.threshold ?? null) : null,
    };
  });

  // The User's corrections.
  const corrections: string[] = [];
  const moved = new Map<string, string>();
  const topicOf = (key: string) => topics.find((topic) => topic.members.includes(key));
  const freeze = (topic: TopicState, why: string) => {
    topic.frozen = true;
    if (!topic.why.includes(why)) topic.why.push(why);
  };
  const takeOut = (key: string) => {
    const from = topicOf(key);
    if (from) from.members.splice(from.members.indexOf(key), 1);
    const at = ungrouped.indexOf(key);
    if (at !== -1) ungrouped.splice(at, 1);
  };
  const move = (key: string, target: TopicState) => {
    const from = topicOf(key);
    takeOut(key);
    target.members.push(key);
    moved.set(key, target.id);
    freeze(target, "a Document was moved into it");
    corrections.push(`Moved ${key} from ${from?.id ?? "Not grouped yet"} into ${target.id}.`);
  };

  // There is always a rename: if its Document isn't grouped, the largest Topic is renamed.
  const holding = topicOf(set.corrections.rename);
  const renamed =
    holding ??
    topics.reduce<TopicState | undefined>(
      (largest, topic) =>
        !largest || topic.members.length > largest.members.length ? topic : largest,
      undefined,
    );
  if (renamed) {
    renamed.name = "Renamed by the User";
    freeze(renamed, "renamed");
    corrections.push(
      holding
        ? `Renamed ${renamed.id}, the Topic holding ${set.corrections.rename}.`
        : `Renamed ${renamed.id}, the largest Topic (${set.corrections.rename} isn't in a Topic).`,
    );
  } else {
    corrections.push("No rename: there are no Topics.");
  }

  let moves = 0;
  for (const [en, zh] of [...set.choosing.pairs, ...set.heldOut.pairs]) {
    if (moves >= set.corrections.moves) break;
    if (!vectors.has(en) || !vectors.has(zh)) continue;
    const target = topicOf(en);
    if (!target || topicOf(zh) === target) continue;
    move(zh, target);
    moves++;
  }
  if (moves === 0) {
    const [key, beside] = set.corrections.lastResortMove;
    const target = topicOf(beside);
    if (target && topicOf(key) !== target && vectors.has(key)) move(key, target);
  }

  const created: TopicState = {
    id: "user-topic-1",
    name: set.corrections.create.name,
    members: [],
    frozen: true,
    why: ["made by the User"],
    userCreated: true,
  };
  for (const key of set.corrections.create.documents) {
    if (!vectors.has(key)) continue;
    takeOut(key);
    created.members.push(key);
    moved.set(key, created.id);
  }
  topics.push(created);
  corrections.push(
    `Made "${created.name}" (${created.id}) by moving ${created.members.join(" and ")} into it.`,
  );
  // An emptied Topic the User didn't make is removed (O1).
  for (let index = topics.length - 1; index >= 0; index--) {
    const topic = topics[index] as TopicState;
    if (topic.members.length === 0 && !topic.userCreated) {
      topics.splice(index, 1);
      corrections.push(`${topic.id} was left empty and removed.`);
    }
  }

  // Regroup.
  const before = topics.map((topic) => ({ ...topic, members: [...topic.members] }));
  const poolDocuments = [
    ...topics.filter((topic) => !topic.frozen).flatMap((topic) => topic.members),
    ...ungrouped.filter((key) => vectors.has(key)),
  ].flatMap((key) => byKey.get(key) ?? []);
  let next = 1;
  const result = regroup({
    vectors,
    topics: topics.map((topic) => ({ id: topic.id, members: topic.members, frozen: topic.frozen })),
    ungrouped,
    moved: new Set(moved.keys()),
    newTopicId: () => `regrouped-${next++}`,
    options: { seed, minTopics: subjectCount(poolDocuments) },
  });

  const after = new Map(result.topics.map((topic) => [topic.id, topic]));
  const checks: CorrectionCheck[] = [];
  for (const topic of before.filter((each) => each.frozen)) {
    const now = after.get(topic.id);
    const kept = !!now && topic.members.every((key) => now.members.includes(key));
    const label = topic.name ? `"${topic.name}" (${topic.id})` : topic.id;
    checks.push({
      description: `${label}, frozen because ${topic.why.join(" and ")}, keeps its id and its ${topic.members.length} Documents`,
      kept,
    });
  }
  for (const [key, target] of moved) {
    const now = result.topics.find((topic) => topic.members.includes(key));
    checks.push({
      description: `${key} stays in ${target}, where the User moved it`,
      kept: now?.id === target,
    });
  }

  const afterTopicOf = topicsByKey(result.topics, result.ungrouped);
  return {
    initial: {
      documents: initial.length,
      k: grouping.k,
      topics: grouping.topics.length,
      ungrouped: grouping.ungrouped.length,
    },
    lateArrivals,
    corrections,
    regroup: {
      pool: poolDocuments.length,
      reclustered: result.reclustered,
      k: result.k,
      kept: result.topics.filter((topic) => topic.kept).length,
      created: result.topics.filter((topic) => !topic.kept).length,
      removed: result.removed.length,
      ungrouped: result.ungrouped.length,
    },
    checks,
    afterRegroup: {
      choosing: scoreCases(set, set.choosing, afterTopicOf),
      heldOut: scoreCases(set, set.heldOut, afterTopicOf),
    },
  };
}
