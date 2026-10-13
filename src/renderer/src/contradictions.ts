/**
 * Contradiction watch (#112), the window's side: "Check for contradictions"
 * (⌘K, the Mind's menu) checks the Mind's cited claims against the other
 * Documents in its Folder. On a model billed per token the line under the
 * Mind's title asks first, with the estimate; otherwise it starts at once.
 * While it runs, the line says so with Stop, as #96's checks do; then it says
 * what it found, or why there was nothing to check.
 */
import { create } from "zustand";
import type { ContradictionCheckOutcome, ContradictionCheckPlan } from "../../core/api";
import {
  type CheckProblem,
  setCheckProblem,
  setCheckRunning,
  setCheckSmallModel,
  useChecks,
} from "./checks";
import { core } from "./core";
import { estimateText } from "./cost";
import { languageNow, translateNow } from "./i18n";
import { useAppStore } from "./store";

interface ContradictionsState {
  /** By Mind: the question asked before a run billed per token, with its estimate. */
  asking: Readonly<Record<string, { claims: number; estimate: string }>>;
  /** By Mind: what the last run on demand found, until closed. */
  found: Readonly<Record<string, ContradictionCheckOutcome>>;
}

export const useContradictions = create<ContradictionsState>()(() => ({ asking: {}, found: {} }));

function setFor<K extends keyof ContradictionsState>(
  key: K,
  mindId: string,
  value: ContradictionsState[K][string] | null,
) {
  useContradictions.setState((state) => {
    const next = { ...state[key] } as Record<string, ContradictionsState[K][string]>;
    if (value === null) delete next[mindId];
    else next[mindId] = value;
    return { [key]: next } as Partial<ContradictionsState>;
  });
}

const problemOf = (plan: Exclude<ContradictionCheckPlan, { kind: "ready" }>): CheckProblem =>
  plan.kind === "nothing" ? { kind: "no-cited" } : plan;

export const dismissContradictions = (mindId: string) => {
  setFor("asking", mindId, null);
  setFor("found", mindId, null);
};

/** Checks the Mind's cited claims for contradictions now. */
export async function checkContradictions(mindId: string): Promise<void> {
  if (useChecks.getState().running[mindId]) return;
  dismissContradictions(mindId);
  const requestId = crypto.randomUUID();
  setCheckRunning(mindId, { requestId, action: "contradiction-watch" });
  try {
    const result = await core.checkContradictions({ requestId, mindId });
    if (result.kind === "checked") {
      const { claims, checked, contradicted, cost, stop } = result;
      setFor("found", mindId, { claims, checked, contradicted, cost, stop });
      setCheckSmallModel(mindId, result.smallModel);
    } else if (result.kind !== "cancelled") {
      setCheckProblem(mindId, result.kind === "nothing" ? { kind: "no-cited" } : result);
    }
  } catch (error) {
    setCheckProblem(mindId, {
      kind: "failed",
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    setCheckRunning(mindId, null);
  }
}

/**
 * "Check for contradictions": shows the Mind, then asks first with the
 * estimate on a model billed per token, or checks at once.
 */
export async function startContradictionCheck(mindId: string): Promise<void> {
  const running = useChecks.getState().running[mindId];
  await useAppStore.getState().openMind(mindId);
  if (running) return;
  const plan = await core.planContradictionCheck(mindId);
  if (plan.kind !== "ready") {
    dismissContradictions(mindId);
    setCheckProblem(mindId, problemOf(plan));
    return;
  }
  const estimate = await core
    .estimateRun({ items: plan.claims, ...(plan.model && { model: plan.model }) })
    .then((result) => estimateText(result, translateNow, languageNow()))
    .catch(() => null);
  if (estimate === null) return checkContradictions(mindId);
  setFor("found", mindId, null);
  setFor("asking", mindId, { claims: plan.claims, estimate });
}
