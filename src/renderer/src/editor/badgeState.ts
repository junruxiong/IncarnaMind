import type { HealthPart } from "./healthCounts";

export const BADGE_MARKS = ["contradicted", "unsupported", "notFound", "notChecked"] as const;
export type BadgeMark = (typeof BADGE_MARKS)[number] & HealthPart;

export interface BadgeInput {
  /** How many sentences or Citations each part of the Mind's health has. */
  marks: Readonly<Partial<Record<HealthPart, number>>>;
  /** Sentences that make a claim no check has looked at. */
  toCheck: number;
  running: boolean;
  asking: boolean;
  /** A check the User started couldn't run or failed. */
  problem: boolean;
  /** What a check the User started checked, until they next type. */
  checked: number | null;
  /** The quiet first open (#856): only a run, a question or a problem shows it, never the marks alone. */
  quiet?: boolean;
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
  const marks = input.quiet
    ? []
    : BADGE_MARKS.flatMap((part) => {
        const count = input.marks[part] ?? 0;
        return count > 0 ? [{ part, count }] : [];
      });
  const toCheck = input.quiet ? 0 : input.toCheck;
  const shown =
    marks.length > 0 ||
    toCheck > 0 ||
    input.running ||
    input.asking ||
    input.problem ||
    input.checked !== null;
  if (!shown) return null;
  return { marks, toCheck, running: input.running, checked: input.checked };
}
