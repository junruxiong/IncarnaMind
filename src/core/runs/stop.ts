/**
 * The one signal that a run must stop before its work is done (#83): the
 * monthly cap would be passed, or a plan's limit is reached. Waiting can't
 * help the next item, so a run of many items stops cleanly at the first,
 * rather than failing item by item, and its unfinished items say "not
 * checked" with the reason. Tasks (#126) resume from it later.
 */
import type { ProviderError, RunStop } from "../api";
import { classifyProviderError } from "../providers/providerErrors";

/** The stop a failure means, or null for a failure that leaves the next item free to try. */
export function runStopOf(error: ProviderError): RunStop | null {
  if (error.kind === "cost-cap") return { reason: "cost-cap" };
  if (error.kind === "plan-limit")
    return { reason: "plan-limit", resetsAt: error.resetsAt ?? null };
  return null;
}

/** What became of one item of a run. */
export type ItemOutcome<T> =
  | { status: "done"; value: T }
  | { status: "failed"; error: ProviderError }
  | { status: "not-checked"; stop: RunStop };

/**
 * Runs a run's items in order (a comparison table's cells, verify's
 * sentences…). A failure fails its item and the run goes on; a stop ends it,
 * and that item and every one after it are not checked, for the stop's reason.
 */
export async function runItems<I, T>(
  items: readonly I[],
  each: (item: I) => Promise<T>,
): Promise<{ outcomes: ItemOutcome<T>[]; stop: RunStop | null }> {
  const outcomes: ItemOutcome<T>[] = [];
  let stop: RunStop | null = null;
  for (const item of items) {
    if (stop) {
      outcomes.push({ status: "not-checked", stop });
      continue;
    }
    try {
      outcomes.push({ status: "done", value: await each(item) });
    } catch (thrown) {
      const error = classifyProviderError(thrown);
      stop = runStopOf(error);
      outcomes.push(stop ? { status: "not-checked", stop } : { status: "failed", error });
    }
  }
  return { outcomes, stop };
}
