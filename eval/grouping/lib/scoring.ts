/**
 * How the grouping check scores a grouping against its fixture set
 * (docs/designs/library-structure-view.md, "Check before building the
 * grouping"): the English–Chinese pairs that land in one Topic, the decks and
 * spreadsheets that land with a Document on their subject, Documents whose
 * names share a prefix pulled together (R3), and which way of grouping wins
 * on the choosing set (O8a).
 */
import type { Cases, GroupingSet, SetDocument } from "./set";

/** Each Document's Topic, by key: an index, or null for "Not grouped yet". Missing: no vector. */
export type TopicOf = ReadonlyMap<string, number | null>;

/** The Topic of each Document key, from Topics listing their members' keys. */
export function topicsByKey(
  topics: readonly { members: readonly string[] }[],
  ungrouped: readonly string[] = [],
): Map<string, number | null> {
  const map = new Map<string, number | null>();
  topics.forEach((topic, index) => {
    for (const key of topic.members) map.set(key, index);
  });
  for (const key of ungrouped) map.set(key, null);
  return map;
}

export interface CaseResult {
  /** The Documents' keys. */
  documents: string[];
  hit: boolean;
  /** For a miss: what happened instead. */
  why: string | null;
}

export interface Tally {
  hits: number;
  total: number;
}

export interface CasesScore {
  pairs: Tally & { cases: CaseResult[] };
  decks: Tally & { cases: CaseResult[] };
  /** Pairs and decks together. */
  hits: number;
  total: number;
}

const tally = (cases: CaseResult[]) => ({
  hits: cases.filter((each) => each.hit).length,
  total: cases.length,
  cases,
});

function subjectsOf(set: GroupingSet): Map<string, SetDocument> {
  return new Map(set.documents.map((document) => [document.key, document]));
}

/** The subjects in a Topic, most Documents first, e.g. "climate ×2, mars". */
function describeTopic(
  set: GroupingSet,
  topicOf: TopicOf,
  topic: number,
  leaveOut: string,
): string {
  const counts = new Map<string, number>();
  for (const document of set.documents) {
    if (document.key === leaveOut || topicOf.get(document.key) !== topic) continue;
    counts.set(document.subject, (counts.get(document.subject) ?? 0) + 1);
  }
  if (counts.size === 0) return "alone";
  return [...counts]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([subject, count]) => (count > 1 ? `${subject} ×${count}` : subject))
    .join(", ");
}

/** Scores one part of the set (the choosing or the held-out cases). */
export function scoreCases(set: GroupingSet, cases: Cases, topicOf: TopicOf): CasesScore {
  const documents = subjectsOf(set);
  const where = (key: string) => topicOf.get(key);
  const pairs = cases.pairs.map(([en, zh]): CaseResult => {
    const a = where(en);
    const b = where(zh);
    const missing = [en, zh].filter((key) => where(key) === undefined);
    if (missing.length)
      return { documents: [en, zh], hit: false, why: `no vector: ${missing.join(", ")}` };
    if (a === null || b === null) {
      const out = [en, zh].filter((key) => where(key) === null);
      return { documents: [en, zh], hit: false, why: `not grouped yet: ${out.join(", ")}` };
    }
    if (a === b) return { documents: [en, zh], hit: true, why: null };
    return {
      documents: [en, zh],
      hit: false,
      why: `apart: ${en} with ${describeTopic(set, topicOf, a as number, en)}; ${zh} with ${describeTopic(set, topicOf, b as number, zh)}`,
    };
  });
  const decks = cases.decks.map((key): CaseResult => {
    const topic = where(key);
    if (topic === undefined) return { documents: [key], hit: false, why: "no vector" };
    if (topic === null) return { documents: [key], hit: false, why: "not grouped yet" };
    const subject = documents.get(key)?.subject;
    const withSubject = set.documents.some(
      (other) => other.key !== key && other.subject === subject && where(other.key) === topic,
    );
    return withSubject
      ? { documents: [key], hit: true, why: null }
      : { documents: [key], hit: false, why: `with ${describeTopic(set, topicOf, topic, key)}` };
  });
  const pairTally = tally(pairs);
  const deckTally = tally(decks);
  return {
    pairs: pairTally,
    decks: deckTally,
    hits: pairTally.hits + deckTally.hits,
    total: pairTally.total + deckTally.total,
  };
}

export interface NameRowScore {
  label: string;
  /** The Documents whose names match. */
  documents: string[];
  /** Pairs of them on different subjects. */
  pairs: number;
  /** How many of those pairs share a Topic: pulled together by the name, or by something else. */
  together: number;
}

/** For each shared-name row: how often Documents on different subjects end up together. */
export function scoreNameRows(
  set: GroupingSet,
  names: ReadonlyMap<string, string>,
  topicOf: TopicOf,
): NameRowScore[] {
  return set.sharedNames.map((row) => {
    const documents = set.documents.filter((document) =>
      row.pattern.test(names.get(document.key) ?? ""),
    );
    let pairs = 0;
    let together = 0;
    documents.forEach((a, index) => {
      for (const b of documents.slice(index + 1)) {
        if (a.subject === b.subject) continue;
        pairs++;
        const topic = topicOf.get(a.key);
        if (topic !== null && topic !== undefined && topic === topicOf.get(b.key)) together++;
      }
    });
    return {
      label: row.label,
      documents: documents.map((document) => document.key),
      pairs,
      together,
    };
  });
}

export interface GroupingSummary {
  topics: number;
  grouped: number;
  ungrouped: number;
  /** The share of grouped Documents in a Topic whose most common subject is theirs. */
  purity: number;
  /** Each Topic's subjects, e.g. "climate ×2, mars". */
  contents: string[];
}

export function summariseGrouping(set: GroupingSet, topicOf: TopicOf): GroupingSummary {
  const byTopic = new Map<number, string[]>();
  let ungrouped = 0;
  for (const document of set.documents) {
    const topic = topicOf.get(document.key);
    if (topic === undefined) continue;
    if (topic === null) {
      ungrouped++;
      continue;
    }
    byTopic.set(topic, [...(byTopic.get(topic) ?? []), document.subject]);
  }
  let majority = 0;
  let grouped = 0;
  const contents: string[] = [];
  for (const topic of [...byTopic.keys()].sort((a, b) => a - b)) {
    const subjects = byTopic.get(topic) as string[];
    const counts = new Map<string, number>();
    for (const subject of subjects) counts.set(subject, (counts.get(subject) ?? 0) + 1);
    majority += Math.max(...counts.values());
    grouped += subjects.length;
    contents.push(describeTopic(set, topicOf, topic, ""));
  }
  return {
    topics: byTopic.size,
    grouped,
    ungrouped,
    purity: grouped === 0 ? 0 : majority / grouped,
    contents,
  };
}

/** Within this many cases of the best, accuracy counts as a near-tie: one case is not evidence (ADR-0009). */
export const NEAR_TIE_CASES = 1;

export interface Candidate {
  id: string;
  /** Hits on the choosing set. */
  hits: number;
  total: number;
  /** What decides a near-tie, lower first: extra cost for the vector variants, seconds for the classifiers. */
  rank: number;
}

/**
 * The winner: the most hits on the choosing set; among candidates within
 * one case of it, the lowest rank (the cheapest variant, or the fastest
 * classifier). Earlier candidates win exact ties.
 */
export function choose(
  candidates: readonly Candidate[],
  rankName: string,
): { id: string; reason: string } {
  if (candidates.length === 0) throw new Error("Nothing to choose from.");
  const best = Math.max(...candidates.map((candidate) => candidate.hits));
  const near = candidates.filter((candidate) => candidate.hits >= best - NEAR_TIE_CASES);
  const winner = near.reduce((a, b) => (b.rank < a.rank ? b : a));
  const top = candidates.find((candidate) => candidate.hits === best) as Candidate;
  const reason =
    near.length === 1
      ? `the most hits on the choosing set (${winner.hits} of ${winner.total})`
      : winner.id === top.id
        ? `the most hits on the choosing set (${winner.hits} of ${winner.total}) and the lowest ${rankName} of those within ${NEAR_TIE_CASES} case`
        : `within ${NEAR_TIE_CASES} case of the most hits (${winner.hits} against ${best} of ${winner.total}), with the lowest ${rankName}`;
  return { id: winner.id, reason };
}

/** The pass bars, on the held-out set: at least 4 of 5 pairs, and 3 of 5 decks or spreadsheets. */
export const BARS = { pairs: 0.8, decks: 0.6 } as const;

export const required = (share: number, total: number) => Math.ceil(share * total - 1e-9);

/** Each bar the held-out score misses, as a sentence. */
export function barFailures(heldOut: CasesScore): string[] {
  const failures: string[] = [];
  const pairs = required(BARS.pairs, heldOut.pairs.total);
  if (heldOut.pairs.hits < pairs) {
    failures.push(
      `Held-out pairs: ${heldOut.pairs.hits} of ${heldOut.pairs.total} land in one Topic; the bar is ${pairs} of ${heldOut.pairs.total}.`,
    );
  }
  const decks = required(BARS.decks, heldOut.decks.total);
  if (heldOut.decks.hits < decks) {
    failures.push(
      `Held-out decks and spreadsheets: ${heldOut.decks.hits} of ${heldOut.decks.total} land with a Document on their subject; the bar is ${decks} of ${heldOut.decks.total}.`,
    );
  }
  return failures;
}
