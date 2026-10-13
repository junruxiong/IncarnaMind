import type { Money, RunEstimate, RunStop } from "../../core/api";
import type { Language } from "../../core/language";
import { formatDateTime } from "../../shared/dates";
import type { MessageKey, MessageParams } from "../../shared/i18n";

type Translate = (key: MessageKey, params?: MessageParams) => string;

/**
 * An amount in its currency: cents, and a small amount to its first figures
 * rather than "$0.00", so one short Answer still shows ("$0.0013").
 */
export function formatMoney({ amount, currency }: Money, language: Language): string {
  const small = amount > 0 && amount < 0.01;
  return new Intl.NumberFormat(language, {
    style: "currency",
    currency,
    minimumFractionDigits: amount === 0 ? 0 : 2,
    maximumFractionDigits: small ? 4 : 2,
  }).format(amount);
}

/** Amounts in one or two currencies: "$3.20", "$3.20 and ¥1.40"; null for none. */
export function formatSpent(
  spent: readonly Money[],
  t: Translate,
  language: Language,
): string | null {
  const [first, second] = spent.map((each) => formatMoney(each, language));
  if (!first) return null;
  return second ? t("cost.and", { first, second }) : first;
}

/** When a plan's limit resets: the time today, else the day and time. */
export function formatResetTime(iso: string, language: Language, now = new Date()): string {
  return formatDateTime(iso, language, { now });
}

/** The line a run stopped at the cap or a plan's limit shows: "Plan limit reached · Try again after 14:30". */
export function runStopText(
  stop: RunStop,
  t: Translate,
  language: Language,
  now = new Date(),
): string {
  if (stop.reason === "cost-cap") return t("run.stopped.cost-cap");
  const line = t("run.stopped.plan-limit");
  return stop.resetsAt
    ? `${line} · ${t("run.stopped.tryAfter", { time: formatResetTime(stop.resetsAt, language, now) })}`
    : line;
}

/** What each unfinished item of a stopped run says: "not checked: cost cap". */
export const notCheckedKey = (stop: RunStop): MessageKey =>
  stop.reason === "cost-cap" ? "run.notChecked.cost-cap" : "run.notChecked.plan-limit";

/**
 * The line before a large run on a model billed per token, shown and never
 * asked: "About 24 calls, 310K tokens, $0.40 · $12.40 of $20 used this month".
 * Null for a plan or a model on this computer, which show nothing.
 */
export function estimateText(
  estimate: RunEstimate,
  t: Translate,
  language: Language,
): string | null {
  if (estimate.billing !== "per-token") return null;
  const tokens = new Intl.NumberFormat(language, { notation: "compact" }).format(estimate.tokens);
  const run = estimate.cost
    ? t("cost.estimate", {
        calls: estimate.calls,
        tokens,
        cost: formatMoney(estimate.cost, language),
      })
    : t("cost.estimate.unpriced", { calls: estimate.calls, tokens });
  if (!estimate.cap || !estimate.spentTowardCap) return run;
  return `${run} · ${t("cost.estimate.month", {
    spent: formatMoney(estimate.spentTowardCap, language),
    cap: formatMoney(estimate.cap, language),
  })}`;
}
