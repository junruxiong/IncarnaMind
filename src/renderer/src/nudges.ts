import type { AppState } from "./store";

export type NudgeId = "usage-data";

type NudgeState = Pick<AppState, "settings" | "privacy" | "ownAnswerArrived" | "usageNotice">;

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
  { id: "usage-data", shown: usageDataShown },
];

export const chooseNudge = (state: NudgeState): NudgeId | null =>
  ORDER.find((nudge) => nudge.shown(state))?.id ?? null;
