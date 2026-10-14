import type { GettingStarted } from "../../core/api";
import type { AppState } from "./store";

/**
 * The one place for nudges, at the bottom of the sidebar (DESIGN.md, First
 * run): one card at a time, the first of these whose condition holds, and
 * each goes by itself once its condition is met. A new nudge is one more entry,
 * in the place its order gives it.
 */
export type NudgeId = "get-started" | "usage-data";

type NudgeState = Pick<AppState, "settings" | "privacy" | "ownAnswerArrived" | "usageNotice">;

/** Whether the "Get started" checklist is shown: started, not hidden, and a step still to do. */
export const gettingStartedShown = (progress: GettingStarted | undefined): boolean =>
  progress !== undefined &&
  progress.started &&
  !progress.hidden &&
  !(progress.citationChecked && progress.indexed && progress.askedOwn);

/** Whether usage data has something to say: a release asks after the first own Answer, a test build says so from the first run. */
export function usageDataShown(state: NudgeState): boolean {
  const usageData = state.privacy?.usageData;
  if (!usageData?.available || usageData.localMode || state.usageNotice === "answered")
    return false;
  if (usageData.testerBuild) return !usageData.asked || state.usageNotice === "kept";
  const ownAnswered =
    state.ownAnswerArrived || state.settings?.device.gettingStarted.askedOwn === true;
  return !usageData.asked && ownAnswered;
}

const ORDER: readonly { id: NudgeId; shown(state: NudgeState): boolean }[] = [
  {
    id: "get-started",
    shown: (state) => gettingStartedShown(state.settings?.device.gettingStarted),
  },
  { id: "usage-data", shown: usageDataShown },
];

/** The nudge to show now, if any. */
export const chooseNudge = (state: NudgeState): NudgeId | null =>
  ORDER.find((nudge) => nudge.shown(state))?.id ?? null;
