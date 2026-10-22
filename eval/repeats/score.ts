/**
 * What the repeats come to, per set and model: each sentence's outcomes, how
 * steady its verdict is, the checks that disagree with the one before, and
 * every result that isn't the verdict its Documents give, in the check's own
 * words.
 */
import type { Expected, RepeatSentence } from "./cases";
import type { Outcome, RepeatResult } from "./repeat";

export interface RepeatRow extends RepeatResult {
  set: string;
  model: string;
  key: string;
  kind: RepeatSentence["kind"];
  expected: Expected;
}

const OUTCOMES: readonly Outcome[] = [
  "supported",
  "partly",
  "nothing",
  "differs",
  "contradicted",
  "not-checked",
  "no-mark",
];

/** The verdict behind an outcome: the three cards of an unsupported sentence are one verdict. */
export const verdictOf = (outcome: Outcome) =>
  outcome === "partly" || outcome === "differs" || outcome === "nothing" ? "unsupported" : outcome;

export function meetsExpected(row: Pick<RepeatRow, "outcome" | "expected">): boolean {
  const verdict = verdictOf(row.outcome);
  if (row.expected === "supported") return verdict === "supported";
  if (row.expected === "unsupported") return verdict === "unsupported";
  return verdict === "unsupported" || verdict === "contradicted";
}

export interface SentenceSummary {
  key: string;
  kind: RepeatSentence["kind"];
  expected: Expected;
  repeats: number;
  outcomes: Record<Outcome, number>;
  /** How many repeats gave its most frequent verdict. */
  steady: number;
}

export interface GroupSummary {
  set: string;
  model: string;
  repeats: number;
  sentences: SentenceSummary[];
  /** Supported, for a sentence its Documents don't support. */
  falseSupported: number;
  /** "Nothing in your Documents says this", for a sentence its Documents state. */
  nothingOnStated: number;
  /** Repeats whose verdicts differ from the one before on any sentence, of those that follow one. */
  disagreeing: number;
  followed: number;
  /** Every result that isn't the verdict its Documents give. */
  misses: RepeatRow[];
}

export function summarise(rows: readonly RepeatRow[]): GroupSummary[] {
  const groups = new Map<string, RepeatRow[]>();
  for (const row of rows) {
    const key = `${row.set}\n${row.model}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return [...groups.values()].map((group) => {
    const first = group[0] as RepeatRow;
    const repeats = [...new Set(group.map((row) => row.repeat))].sort((a, b) => a - b);
    const keys = [...new Set(group.map((row) => row.key))];
    const sentences = keys.map((key): SentenceSummary => {
      const own = group.filter((row) => row.key === key);
      const outcomes = Object.fromEntries(OUTCOMES.map((outcome) => [outcome, 0])) as Record<
        Outcome,
        number
      >;
      const verdicts = new Map<string, number>();
      for (const row of own) {
        outcomes[row.outcome]++;
        const verdict = verdictOf(row.outcome);
        verdicts.set(verdict, (verdicts.get(verdict) ?? 0) + 1);
      }
      const sample = own[0] as RepeatRow;
      return {
        key,
        kind: sample.kind,
        expected: sample.expected,
        repeats: own.length,
        outcomes,
        steady: Math.max(...verdicts.values()),
      };
    });
    const verdictsIn = (repeat: number) =>
      keys
        .map((key) => {
          const row = group.find((each) => each.repeat === repeat && each.key === key);
          return row ? verdictOf(row.outcome) : "missing";
        })
        .join(" ");
    const pairs = repeats
      .slice(1)
      .map((repeat, index): [number, number] => [repeats[index] as number, repeat]);
    return {
      set: first.set,
      model: first.model,
      repeats: repeats.length,
      sentences,
      falseSupported: group.filter(
        (row) => row.expected !== "supported" && row.outcome === "supported",
      ).length,
      nothingOnStated: group.filter(
        (row) => row.expected === "supported" && row.outcome === "nothing",
      ).length,
      disagreeing: pairs.filter(([before, after]) => verdictsIn(before) !== verdictsIn(after))
        .length,
      followed: pairs.length,
      misses: group.filter((row) => !meetsExpected(row)),
    };
  });
}

const cell = (text: string | null) => (text ?? "").replace(/\|/g, "\\|").replace(/\n/g, " ");

export function report(
  summaries: readonly GroupSummary[],
  run: { commit: string; spent: number; repeats: number },
): string {
  return [
    "# Repeated checks of an unchanged Mind",
    "",
    `Commit ${run.commit}, ${run.repeats} repeats a Mind, $${run.spent.toFixed(2)} at the app's own count.`,
    "",
    ...summaries.flatMap((group) => [
      `## ${group.set} · ${group.model} · ${group.repeats} checks`,
      "",
      "| Sentence | Kind | Expected | Supported | Partly | Nothing | Differs | Contradicted | Not checked | Same verdict |",
      "|---|---|---|---|---|---|---|---|---|---|",
      ...group.sentences.map(
        (each) =>
          `| ${each.key} | ${each.kind} | ${each.expected} | ${each.outcomes.supported} | ${each.outcomes.partly} | ${each.outcomes.nothing} | ${each.outcomes.differs} | ${each.outcomes.contradicted} | ${each.outcomes["not-checked"] + each.outcomes["no-mark"]} | ${each.steady} of ${each.repeats} |`,
      ),
      "",
      `- False Supported: ${group.falseSupported}`,
      `- "Nothing in your Documents says this" on a sentence they state: ${group.nothingOnStated}`,
      `- Checks whose verdicts differ from the one before: ${group.disagreeing} of ${group.followed}`,
      "",
      ...(group.misses.length > 0
        ? [
            "| Check | Sentence | Card | What the check said | Covered | Not covered | Quote |",
            "|---|---|---|---|---|---|---|",
            ...group.misses.map(
              (row) =>
                `| ${row.repeat} | ${cell(row.sentence)} | ${row.outcome} | ${cell(row.failure ?? row.says)} | ${cell(row.covered)} | ${cell(row.uncovered)} | ${cell(row.quote)} |`,
            ),
            "",
          ]
        : []),
    ]),
  ].join("\n");
}
