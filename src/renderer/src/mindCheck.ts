/**
 * Check this Mind (#737), the window's side: "Check this Mind" (⌘K, the
 * Mind's menu) checks every sentence that makes a claim; the health line's
 * "Check all" and "Check again" check only what changed. When the model is
 * to name the claims, it does that first, with Stop, and the line under the
 * Mind's title then asks with their number and, billed per token, the
 * estimate (#973); a run that only checks again asks so when it is large,
 * and always for the Documents added since (#993). While it runs, the
 * health line counts the claims checked, with Stop.
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
import { type EstimateLine, estimateLine } from "./cost";
import { languageNow, translateNow } from "./i18n";
import { useAppStore } from "./store";

interface MindCheckState {
  /** By Mind: the question asked before a run billed per token, with its estimate. */
  asking: Readonly<Record<string, Asking>>;
}

interface Asking {
  mode: CheckMindMode;
  allDocuments: boolean;
  /** The claims the model named and the checks to make again, or, when only checking again, the checks. */
  count: number;
  claims: boolean;
  /** Checked against the Mind's Folder: the question names it. */
  inFolder: boolean;
  /** Null for a plan or a model on this computer, or when `estimateRun` failed. */
  estimate: EstimateLine | null;
}

/** A run that only checks again asks first, with the estimate, when it checks this many or more. */
const ASK_FROM = 10;

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

/** Checks the Mind now, as `mode` says: against its Folder, or every Document. */
export async function checkMind(
  mindId: string,
  mode: CheckMindMode,
  { allDocuments = false }: { allDocuments?: boolean } = {},
): Promise<void> {
  if (useChecks.getState().running[mindId]) return;
  dismissMindCheck(mindId);
  dismissCheckProblem(mindId);
  const requestId = crypto.randomUUID();
  setCheckRunning(mindId, { requestId, action: "whole-mind" });
  try {
    const result = await core.checkMind({ requestId, mindId, mode, allDocuments });
    if (result.kind === "checked") setCheckSmallModel(mindId, result.smallModel);
    else if (result.kind === "failed") {
      setCheckProblem(mindId, { kind: "find-failed", error: result.error, mode, allDocuments });
    } else if (result.kind === "nothing") setCheckProblem(mindId, { kind: "no-claims" });
    else if (result.kind !== "cancelled") setCheckProblem(mindId, result);
  } catch (error) {
    setCheckProblem(mindId, { kind: "failed", error });
  } finally {
    setCheckRunning(mindId, null);
  }
}

async function findClaims(
  mindId: string,
  mode: CheckMindMode,
  allDocuments: boolean,
): Promise<number | null> {
  const requestId = crypto.randomUUID();
  setCheckRunning(mindId, { requestId, action: "whole-mind" });
  try {
    const found = await core.findMindClaims({ requestId, mindId, mode, allDocuments });
    if (found.kind === "found") return found.claims;
    if (found.kind === "failed") {
      setCheckProblem(mindId, { kind: "find-failed", error: found.error, mode, allDocuments });
    } else if (found.kind === "nothing") setCheckProblem(mindId, { kind: "no-claims" });
    else if (found.kind !== "cancelled") setCheckProblem(mindId, found);
  } catch (error) {
    setCheckProblem(mindId, { kind: "failed", error });
  } finally {
    setCheckRunning(mindId, null);
  }
  return null;
}

/**
 * Shows the Mind, has the model name the claims, then asks with their number
 * and the estimate; a run that only checks again checks at once when small,
 * or when nothing is billed per token.
 */
export async function startMindCheck(
  mindId: string,
  mode: CheckMindMode,
  { allDocuments = false }: { allDocuments?: boolean } = {},
): Promise<void> {
  const { openMind, minds } = useAppStore.getState();
  await openMind(mindId);
  if (useChecks.getState().running[mindId]) return;
  const plan = await core.planMindCheck(mindId, mode, allDocuments);
  if (plan.kind !== "ready") {
    dismissMindCheck(mindId);
    setCheckProblem(mindId, plan.kind === "nothing" ? { kind: "no-claims" } : plan);
    return;
  }
  if (mode !== "added" && !plan.findsClaims && plan.sentences < ASK_FROM)
    return checkMind(mindId, mode, { allDocuments });
  dismissMindCheck(mindId);
  dismissCheckProblem(mindId);
  const count = plan.findsClaims ? await findClaims(mindId, mode, allDocuments) : plan.sentences;
  if (count === null) return;
  const estimate = await core
    .estimateRun({ items: count, each: "check", ...(plan.model && { model: plan.model }) })
    .then((result) => estimateLine(result, translateNow, languageNow()))
    .catch((error: unknown) => {
      console.warn(`Couldn't estimate checking Mind ${mindId}; it asks without one.`, error);
      return null;
    });
  if (estimate === null && !plan.findsClaims) return checkMind(mindId, mode, { allDocuments });
  const inFolder =
    !allDocuments && (minds.find((each) => each.id === mindId)?.folderId ?? null) !== null;
  useMindCheck.setState((state) => ({
    asking: {
      ...state.asking,
      [mindId]: { mode, allDocuments, count, claims: plan.findsClaims, inFolder, estimate },
    },
  }));
}
