import { chooseNudge } from "../nudges";
import { useAppStore } from "../store";
import { UsageDataCard } from "./UsageDataCard";

export function NudgeSlot() {
  const nudge = useAppStore(chooseNudge);
  return nudge === "usage-data" ? <UsageDataCard /> : null;
}
