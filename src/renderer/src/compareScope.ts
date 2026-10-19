import type { Currency, RunEstimate } from "../../core/api";

/** A source's Documents come ticked only when there are this few: a newcomer's one button never spends on a whole Library. */
export const TICKED_AT_START_MAX = 5;

/** A run over more Documents than this asks first. */
export const ASK_ABOVE_DOCUMENTS = 20;

/**
 * A run whose estimate is above this asks first, in each currency's own
 * figures. A table of 4 Documents on 4 Questions costs about 30 cents on a
 * cloud model and is plainly meant, so it goes straight through.
 */
export const ASK_ABOVE_COST: Readonly<Record<Currency, number>> = { USD: 0.5, CNY: 3.5 };

/** The Documents a source starts with ticked: all of a few, none of many. */
export const startingChecks = <T>(offered: readonly T[]): readonly T[] =>
  offered.length <= TICKED_AT_START_MAX ? offered : [];

/** Whether the run asks before it starts: `estimate` is null for a plan, this computer, or a cost not known. */
export function asksFirst(documents: number, estimate: RunEstimate | null): boolean {
  if (documents > ASK_ABOVE_DOCUMENTS) return true;
  return (
    estimate?.billing === "per-token" &&
    estimate.cost !== null &&
    estimate.cost.amount > ASK_ABOVE_COST[estimate.cost.currency]
  );
}
