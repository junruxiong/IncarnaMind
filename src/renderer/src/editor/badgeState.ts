import type { HealthPart } from "./healthCounts";

/** The parts whose margin mark the badge shows, in its order. */
export const BADGE_MARKS = ["contradicted", "unsupported", "notFound", "notChecked"] as const;
export type BadgeMark = (typeof BADGE_MARKS)[number] & HealthPart;

export interface BadgeInput {
  /** How many sentences or Citations each marked part has. */
  marks: Record<BadgeMark, number>;
  /** Sentences that make a claim no check has looked at. */
  toCheck: number;
  /** "Checking 12 of 24", or a check's own words, while one runs. */
  running: boolean;
  /** A question before a run billed per token waits. */
  asking: boolean;
  /** A check the User started couldn't run or failed. */
  problem: boolean;
  /** What a check the User started checked, until they next type. */
  checked: number | null;
}

export interface Badge {
  marks: { part: BadgeMark; count: number }[];
  toCheck: number;
  running: boolean;
  checked: number | null;
}

/**
 * What the check badge shows, or null for nothing. "Needs re-check" and
 * "with no Citation" are in its card but never show it on their own.
 */
export function badgeOf(input: BadgeInput): Badge | null {
  const marks = BADGE_MARKS.flatMap((part) =>
    input.marks[part] > 0 ? [{ part, count: input.marks[part] }] : [],
  );
  const shown =
    marks.length > 0 ||
    input.toCheck > 0 ||
    input.running ||
    input.asking ||
    input.problem ||
    input.checked !== null;
  if (!shown) return null;
  return { marks, toCheck: input.toCheck, running: input.running, checked: input.checked };
}
