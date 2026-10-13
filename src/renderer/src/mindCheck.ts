/**
 * Check this Mind (#737), the window's side: "Check this Mind" (⌘K, the
 * Mind's menu) checks every sentence that makes a claim; the health line's
 * "Check all" and "Check again" check only what changed. When the model is
 * to name the claims and is billed per token, the line under the Mind's title
 * asks first, with the estimate; otherwise it starts at once. While it runs,
 * the health line counts the claims checked, with Stop.
 */
import { useEffect, useState } from "react";
import type * as Y from "yjs";
import { create } from "zustand";
import type { CheckMindMode, MindCheckRun } from "../../core/api";
import { mindCheckOf, observeMindCheck } from "../../shared/mindCheck";
import {
  dismissCheckProblem,
  setCheckProblem,
  setCheckRunning,
  setCheckSmallModel,
  useChecks,
} from "./checks";
import { core } from "./core";
import { estimateText } from "./cost";
import { languageNow, translateNow } from "./i18n";
import { useAppStore } from "./store";

interface MindCheckState {
  /** By Mind: the question asked before a run billed per token, with its estimate. */
  asking: Readonly<
    Record<string, { mode: CheckMindMode; sentences: number; inFolder: boolean; estimate: string }>
  >;
}

export const useMindCheck = create<MindCheckState>()(() => ({ asking: {} }));

export const dismissMindCheck = (mindId: string) =>
  useMindCheck.setState((state) => {
    const asking = { ...state.asking };
    delete asking[mindId];
    return { asking };
  });

/** The Mind's last whole-Mind check, as it changes. */
export function useMindCheckRun(doc: Y.Doc): MindCheckRun | null {
  const [run, setRun] = useState(() => mindCheckOf(doc));
  useEffect(() => {
    setRun(mindCheckOf(doc));
    return observeMindCheck(doc, () => setRun(mindCheckOf(doc)));
  }, [doc]);
  return run;
}

/** Checks the Mind now, as `mode` says. */
export async function checkMind(mindId: string, mode: CheckMindMode): Promise<void> {
  if (useChecks.getState().running[mindId]) return;
  dismissMindCheck(mindId);
  dismissCheckProblem(mindId);
  const requestId = crypto.randomUUID();
  setCheckRunning(mindId, { requestId, action: "whole-mind" });
  try {
    const result = await core.checkMind({ requestId, mindId, mode });
    if (result.kind === "checked") setCheckSmallModel(mindId, result.smallModel);
    else if (result.kind === "failed") {
      setCheckProblem(mindId, { kind: "find-failed", error: result.error });
    } else if (result.kind === "nothing") setCheckProblem(mindId, { kind: "no-claims" });
    else if (result.kind !== "cancelled") setCheckProblem(mindId, result);
  } catch (error) {
    setCheckProblem(mindId, { kind: "failed", error });
  } finally {
    setCheckRunning(mindId, null);
  }
}

/**
 * Shows the Mind, then asks first with the estimate when the model names
 * the claims and is billed per token, or checks at once.
 */
export async function startMindCheck(mindId: string, mode: CheckMindMode): Promise<void> {
  const { openMind, minds } = useAppStore.getState();
  await openMind(mindId);
  if (useChecks.getState().running[mindId]) return;
  const plan = await core.planMindCheck(mindId, mode);
  if (plan.kind !== "ready") {
    dismissMindCheck(mindId);
    setCheckProblem(mindId, plan.kind === "nothing" ? { kind: "no-claims" } : plan);
    return;
  }
  if (!plan.findsClaims) return checkMind(mindId, mode);
  const estimate = await core
    .estimateRun({ items: plan.sentences, ...(plan.model && { model: plan.model }) })
    .then((result) => estimateText(result, translateNow, languageNow()))
    .catch(() => null);
  if (estimate === null) return checkMind(mindId, mode);
  const inFolder = (minds.find((each) => each.id === mindId)?.folderId ?? null) !== null;
  dismissCheckProblem(mindId);
  useMindCheck.setState((state) => ({
    asking: { ...state.asking, [mindId]: { mode, sentences: plan.sentences, inFolder, estimate } },
  }));
}
