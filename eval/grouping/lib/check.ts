/**
 * The grouping check's decisions, apart from its input and output, so that
 * `npm test` covers them without the model: scoring each Document vector,
 * choosing the variant on the choosing set, the held-out bars, the report's
 * recommendation, and the founder's sheet.
 */
import { DEFAULT_SEED, groupDocuments } from "../../../src/core/topics/grouping";
import type { Assignment, ClassifierRun } from "./classifiers";
import { SAMPLE_SIZE, type SheetRow, shuffled, topicPreview } from "./founder";
import type { IncrementalResult } from "./incremental";
import type {
  ClassifierResult,
  Classifiers,
  FixturesReport,
  FounderReport,
  RunInfo,
  VariantResult,
} from "./report";
import {
  BARS,
  barFailures,
  choose,
  required,
  scoreCases,
  scoreNameRows,
  summariseGrouping,
  topicsByKey,
} from "./scoring";
import type { GroupingSet } from "./set";
import { median } from "./timing";
import { VARIANTS, type Variant, type VariantId, type VariantVectors } from "./variants";

export const variantOf = (id: VariantId): Variant =>
  VARIANTS.find((variant) => variant.id === id) as Variant;

/** Groups the set's Documents with one variant's vectors and scores every case. */
export function evaluateVariant(
  set: GroupingSet,
  built: VariantVectors,
  names: ReadonlyMap<string, string>,
  minTopics: number,
): VariantResult {
  const documents = set.documents
    .filter((document) => built.vectors.has(document.key))
    .map((document) => ({
      id: document.key,
      vector: built.vectors.get(document.key) as Float32Array,
    }));
  const started = performance.now();
  const grouping = groupDocuments(documents, { seed: DEFAULT_SEED, minTopics });
  const seconds = (performance.now() - started) / 1000;
  const topicOf = topicsByKey(grouping.topics, grouping.ungrouped);
  const variant = variantOf(built.id);
  return {
    id: built.id,
    label: variant.label,
    cost: variant.cost,
    documents: documents.length,
    extraEmbeddings: built.embeddings,
    buildSeconds: built.seconds,
    k: grouping.k,
    seconds,
    runs: grouping.runs,
    summary: summariseGrouping(set, topicOf),
    choosing: scoreCases(set, set.choosing, topicOf),
    heldOut: scoreCases(set, set.heldOut, topicOf),
    nameRows: scoreNameRows(set, names, topicOf),
  };
}

/** The variant with the most choosing hits; a near-tie goes to the cheaper one (R3: "cost decides near-ties"). */
export function chooseVariant(variants: readonly VariantResult[]): {
  id: VariantId;
  reason: string;
} {
  const chosen = choose(
    variants.map((variant) => ({
      id: variant.id,
      hits: variant.choosing.hits,
      total: variant.choosing.total,
      rank: variantOf(variant.id).rank,
    })),
    "extra cost",
  );
  return { id: chosen.id as VariantId, reason: chosen.reason };
}

/** A classifier's assignments as a grouping, scored like the variants. A failed Document counts as not grouped. */
export function scoreClassifier(
  set: GroupingSet,
  keys: readonly string[],
  run: ClassifierRun,
): ClassifierResult {
  const topicOf = new Map(
    keys.map((key) => [key, (run.assignments.get(key) as Assignment | undefined)?.topic ?? null]),
  );
  return {
    name: run.name,
    local: run.local,
    seconds: run.seconds,
    perDocumentMs: (run.seconds * 1000) / Math.max(1, keys.length),
    failed: run.failed,
    firstError: run.firstError,
    summary: summariseGrouping(set, topicOf),
    choosing: scoreCases(set, set.choosing, topicOf),
    heldOut: scoreCases(set, set.heldOut, topicOf),
  };
}

/** The classifiers ranked: accuracy on the choosing set first, then speed within one case (R0). */
export function rankClassifiers(
  runs: readonly ClassifierResult[],
): { name: string; reason: string } | null {
  if (runs.length === 0) return null;
  const best = choose(
    runs.map((run) => ({
      id: run.name,
      hits: run.choosing.hits,
      total: run.choosing.total,
      rank: run.seconds,
    })),
    "time",
  );
  return { name: best.id, reason: best.reason };
}

/** Each Document's part in the check, for the report's fixture table. */
function roleOf(set: GroupingSet, key: string): string {
  const roles: string[] = [];
  for (const [name, cases] of [
    ["choosing", set.choosing],
    ["held-out", set.heldOut],
  ] as const) {
    if (cases.pairs.some((pair) => pair.includes(key))) roles.push(`${name} pair`);
    if (cases.decks.includes(key)) {
      roles.push(`${name} ${set.documents.find((document) => document.key === key)?.form}`);
    }
  }
  if (set.lateArrivals.includes(key)) roles.push("late arrival");
  return roles.join("; ");
}

export interface FixturesParts {
  set: GroupingSet;
  run: RunInfo;
  names: ReadonlyMap<string, string>;
  passages: ReadonlyMap<string, number>;
  processing: { passages: number; seconds: number };
  variants: VariantResult[];
  incremental: IncrementalResult;
  classifiers: Classifiers;
  timing: FixturesReport["timing"];
}

/** The report: the chosen variant, its held-out bars, what failed, and the recommendation. */
export function fixturesReport(parts: FixturesParts): FixturesReport {
  const { set, variants, incremental } = parts;
  const chosen = chooseVariant(variants);
  const winner = variants.find((variant) => variant.id === chosen.id) as VariantResult;
  const failures = [
    ...barFailures(winner.heldOut),
    ...incremental.checks
      .filter((check) => !check.kept)
      .map((check) => `The Regroup lost the User's change: ${check.description}.`),
  ];
  const report: Omit<FixturesReport, "recommendation"> = {
    kind: "fixtures",
    result: failures.length === 0 ? "pass" : "fail",
    failures,
    run: parts.run,
    set: {
      source: set.source,
      subjects: Object.keys(set.subjects).length,
      documents: set.documents.map((document) => ({
        key: document.key,
        name: parts.names.get(document.key) ?? document.key,
        subject: document.subject,
        language: document.language,
        form: document.form,
        passages: parts.passages.get(document.key) ?? 0,
        role: roleOf(set, document.key),
      })),
    },
    processing: parts.processing,
    variants,
    chosen,
    bars: {
      pairs: {
        hits: winner.heldOut.pairs.hits,
        total: winner.heldOut.pairs.total,
        needed: required(BARS.pairs, winner.heldOut.pairs.total),
      },
      decks: {
        hits: winner.heldOut.decks.hits,
        total: winner.heldOut.decks.total,
        needed: required(BARS.decks, winner.heldOut.decks.total),
      },
    },
    incremental,
    classifiers: parts.classifiers,
    timing: parts.timing,
  };
  return { ...report, recommendation: recommendation(report) };
}

const lower = (text: string) => text.charAt(0).toLowerCase() + text.slice(1);

/** What the run says to do next, in a few lines. */
export function recommendation(report: Omit<FixturesReport, "recommendation">): string[] {
  const winner = report.variants.find(
    (variant) => variant.id === report.chosen.id,
  ) as VariantResult;
  const { pairs, decks } = report.bars;
  const passes = barFailures(winner.heldOut).length === 0;
  const lines = [
    passes
      ? `Build the grouping as designed, with ${lower(winner.label)}: on the held-out set ${pairs.hits} of ${pairs.total} pairs land in one Topic (bar ${pairs.needed}) and ${decks.hits} of ${decks.total} decks or spreadsheets with their subject (bar ${decks.needed}). The founder's sample is the last bar.`
      : `The held-out bars aren't met with ${lower(winner.label)} (pairs ${pairs.hits} of ${pairs.total}, bar ${pairs.needed}; decks and spreadsheets ${decks.hits} of ${decks.total}, bar ${decks.needed}). As the design says, revise its grouping section to a fallback (cluster Passage vectors and give each Document its majority cluster; short summaries; or R0's classifier fallback) and review it again before the build.`,
    `Document vectors: ${lower(winner.label)}, by ${report.chosen.reason}. On the held-out set the others score ${report.variants
      .filter((variant) => variant.id !== winner.id)
      .map(
        (variant) =>
          `${variant.heldOut.hits} of ${variant.heldOut.total} (${lower(variant.label)})`,
      )
      .join(" and ")}.`,
  ];
  const rows = report.variants.map((variant) => ({ variant, row: variant.nameRows[0] }));
  const first = rows[0]?.row;
  if (first) {
    lines.push(
      `Shared names (R3): of the ${first.pairs} pairs of ${first.label} Documents on different subjects, ${rows.map(({ variant, row }) => `${row?.together ?? 0} land in one Topic with ${lower(variant.label)}`).join(", ")}.`,
    );
  }
  const { incremental } = report;
  const right = incremental.lateArrivals.filter((arrival) => arrival.correct).length;
  const kept = incremental.checks.filter((check) => check.kept).length;
  lines.push(
    `Incremental cases: ${right} of ${incremental.lateArrivals.length} late arrivals placed as expected by the threshold; ${kept} of ${incremental.checks.length} of the User's changes kept through the Regroup.`,
  );
  if ("skipped" in report.timing) {
    lines.push(`Timing: not measured (${report.timing.skipped}).`);
  } else {
    const { kMeans, means } = report.timing;
    const cold = Math.round(median(means.coldMs));
    lines.push(
      `k-means at ${kMeans.documents.toLocaleString("en")} Documents takes ${kMeans.seconds.toFixed(1)} s on one thread: it belongs on the grouping worker (R1).`,
      means.passes
        ? `R2 holds: loading the index and computing the means at ${means.passages.toLocaleString("en")} Passages blocks the core's thread for ${cold} ms (median), within the ${means.barMs} ms bar. Keep computing the means there.`
        : `Take O5's named fallback, the grouping worker reading the vectors itself through its own read-only connection (WAL): loading the index and computing the means at ${means.passages.toLocaleString("en")} Passages blocks the core's thread for ${cold} ms (median), over the ${means.barMs} ms bar. The means alone, once the index is loaded, take ${Math.round(median(means.warmMs))} ms.`,
    );
  }
  lines.push(
    "skipped" in report.classifiers
      ? `The classifier fallback (R0) wasn't measured: ${report.classifiers.skipped}`
      : `The classifier fallback (R0): ${report.classifiers.ranking?.name ?? "none"} ranks first, by ${report.classifiers.ranking?.reason}.`,
    "The founder's 30-Document sample is still to do: it needs the User's own library (steps below).",
  );
  return lines;
}

export interface FounderSample {
  rows: SheetRow[];
  letters: FounderReport["letters"];
  sample: number;
}

/**
 * The founder's sheet: each variant groups the library (k from its size),
 * gets a letter in a seeded order, and 30 random Documents are shown with
 * their Topic in each grouping.
 */
export function founderSample(
  built: readonly VariantVectors[],
  names: ReadonlyMap<string, string>,
  seed: number,
): FounderSample {
  const keys = [...names.keys()].filter((key) => built.every((each) => each.vectors.has(key)));
  const letters = shuffled(built, seed).map((each, index) => ({
    letter: String.fromCharCode(65 + index),
    built: each,
    grouping: groupDocuments(
      keys.map((key) => ({ id: key, vector: each.vectors.get(key) as Float32Array })),
      { seed: DEFAULT_SEED },
    ),
  }));
  const sample = shuffled(keys, seed).slice(0, SAMPLE_SIZE);
  const rows = sample.flatMap((key) =>
    letters.map(({ letter, built: each, grouping }): SheetRow => {
      const topic = grouping.topics.find((candidate) => candidate.members.includes(key));
      return {
        document: names.get(key) ?? key,
        file: key,
        grouping: letter,
        topic: topic
          ? topicPreview(
              topic.members.map((member) => ({
                key: member,
                name: names.get(member) ?? member,
                vector: each.vectors.get(member) as Float32Array,
              })),
              topic.centroid,
              key,
            )
          : "Not grouped yet",
      };
    }),
  );
  return {
    rows,
    sample: sample.length,
    letters: Object.fromEntries(
      letters.map(({ letter, built: each, grouping }) => [
        letter,
        {
          id: each.id,
          label: variantOf(each.id).label,
          k: grouping.k,
          topics: grouping.topics.length,
          ungrouped: grouping.ungrouped.length,
          buildSeconds: each.seconds,
        },
      ]),
    ),
  };
}
