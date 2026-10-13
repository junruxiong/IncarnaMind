/**
 * The last check for contradictions a new version of a Document started in a
 * Mind (#112), in the Mind's Yjs settings like its model (`mindModel.ts`), so
 * the line under its title can say what it found and cost, wherever the Mind
 * is open. Not content: an older version leaves it as it is.
 */
import type * as Y from "yjs";
import {
  type ContradictionWatchRun,
  type Currency,
  MIND_SETTINGS_FIELD,
  type Money,
  type RunStop,
} from "../core/api";

const KEY = "contradictionWatch";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const count = (value: unknown) =>
  typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : 0;

const CURRENCIES: readonly Currency[] = ["USD", "CNY"];

function moneyOf(value: unknown): Money | null {
  if (!isRecord(value) || typeof value.amount !== "number") return null;
  const currency = CURRENCIES.find((each) => each === value.currency);
  return currency ? { currency, amount: value.amount } : null;
}

function stopOf(value: unknown): RunStop | null {
  if (!isRecord(value)) return null;
  if (value.reason === "cost-cap") return { reason: "cost-cap" };
  if (value.reason === "plan-limit") {
    return {
      reason: "plan-limit",
      resetsAt: typeof value.resetsAt === "string" ? value.resetsAt : null,
    };
  }
  return null;
}

/** The last run a new version started, or null for none (or one this version can't read). */
export function readContradictionWatch(settings: Y.Map<unknown>): ContradictionWatchRun | null {
  const value = settings.get(KEY);
  if (!isRecord(value) || typeof value.at !== "string") return null;
  return {
    at: value.at,
    documentName: typeof value.documentName === "string" ? value.documentName : "",
    claims: count(value.claims),
    checked: count(value.checked),
    contradicted: count(value.contradicted),
    cost: moneyOf(value.cost),
    stop: stopOf(value.stop),
  };
}

export const contradictionWatchOf = (doc: Y.Doc): ContradictionWatchRun | null =>
  readContradictionWatch(doc.getMap(MIND_SETTINGS_FIELD));

export function writeContradictionWatch(settings: Y.Map<unknown>, run: ContradictionWatchRun) {
  settings.set(KEY, {
    ...run,
    cost: run.cost && { ...run.cost },
    stop: run.stop && { ...run.stop },
  });
}

/** Calls `listener` whenever the Mind's last run may have changed. Returns how to stop. */
export function observeContradictionWatch(doc: Y.Doc, listener: () => void): () => void {
  const settings = doc.getMap(MIND_SETTINGS_FIELD);
  settings.observe(listener);
  return () => settings.unobserve(listener);
}
