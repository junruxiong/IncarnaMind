import type { GettingStarted } from "../../core/api";
import type { AppState } from "./store";

export type NudgeId = "get-started" | "usage-data";

type NudgeState = Pick<AppState, "settings" | "privacy" | "ownAnswerArrived" | "usageNotice">;

export const gettingStartedShown = (progress: GettingStarted | undefined): boolean =>
  progress !== undefined &&
  progress.started &&
  !progress.hidden &&
  !(progress.citationChecked && progress.indexed && progress.askedOwn);

export function usageDataShown(state: NudgeState): boolean {
  const usageData = state.privacy?.usageData;
  if (!usageData?.available || usageData.localMode || state.usageNotice === "answered")
    return false;
  if (usageData.testerBuild) return !usageData.asked || state.usageNotice === "kept";
  const ownAnswered =
    state.ownAnswerArrived || state.settings?.device.gettingStarted.askedOwn === true;
  return !usageData.asked && ownAnswered;
}

// A test build sends from its first launch, so its notice must come before Get started.
const ORDER: readonly { id: NudgeId; shown(state: NudgeState): boolean }[] = [
  {
    id: "usage-data",
    shown: (state) => state.privacy?.usageData.testerBuild === true && usageDataShown(state),
  },
  {
    id: "get-started",
    shown: (state) => gettingStartedShown(state.settings?.device.gettingStarted),
  },
  { id: "usage-data", shown: usageDataShown },
];

export const chooseNudge = (state: NudgeState): NudgeId | null =>
  ORDER.find((nudge) => nudge.shown(state))?.id ?? null;
