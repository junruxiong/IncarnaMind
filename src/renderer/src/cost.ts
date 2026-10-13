import type { Money, RunEstimate, RunStop } from "../../core/api";
import type { Language } from "../../core/language";
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
  const at = new Date(iso);
  const sameDay = at.toDateString() === now.toDateString();
  return new Intl.DateTimeFormat(language, {
    ...(sameDay ? {} : { month: "short", day: "numeric" }),
    hour: "numeric",
    minute: "2-digit",
  }).format(at);
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
    ? joinMeta([
        line,
        t("run.stopped.tryAfter", { time: formatResetTime(stop.resetsAt, language, now) }),
      ])
    : line;
}

/** What each unfinished item of a stopped run says: "not checked: cost cap". */
export const notCheckedKey = (stop: RunStop): MessageKey =>
  stop.reason === "cost-cap" ? "run.notChecked.cost-cap" : "run.notChecked.plan-limit";

/** Parts of a meta line on one line, joined by middle dots; `MetaLine` is for lines that can wrap. */
export const joinMeta = (parts: readonly string[]): string => parts.filter(Boolean).join(" · ");

/** The estimate as parts for a `MetaLine`, and the tooltip that adds the tokens. */
export interface EstimateLine {
  parts: string[];
  title: string;
}

/**
 * The line before a large run on a model billed per token, shown and never
 * asked: "About $0.40 · 24 model calls · $12.40 of $20.00 used this month",
 * with the tokens in its tooltip (price first, as DESIGN.md's plan card writes
 * it). With no known price: "About 24 model calls · 310K tokens". Null for a
 * plan or a model on this computer, which show nothing.
 */
export function estimateLine(
  estimate: RunEstimate,
  t: Translate,
  language: Language,
): EstimateLine | null {
  if (estimate.billing !== "per-token") return null;
  const tokens = t("cost.estimate.tokens", {
    tokens: new Intl.NumberFormat(language, { notation: "compact" }).format(estimate.tokens),
  });
  const { calls } = estimate;
  const month =
    estimate.cap && estimate.spentTowardCap
      ? [
          t("cost.estimate.month", {
            spent: formatMoney(estimate.spentTowardCap, language),
            cap: formatMoney(estimate.cap, language),
          }),
        ]
      : [];
  const run = estimate.cost
    ? [
        t("cost.estimate.price", { cost: formatMoney(estimate.cost, language) }),
        t(calls === 1 ? "cost.estimate.calls.one" : "cost.estimate.calls", { calls }),
      ]
    : [
        t(calls === 1 ? "cost.estimate.callsAbout.one" : "cost.estimate.callsAbout", { calls }),
        tokens,
      ];
  const parts = [...run, ...month];
  return { parts, title: joinMeta(estimate.cost ? [...run, tokens, ...month] : parts) };
}
