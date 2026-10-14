import { chooseNudge } from "../nudges";
import { useAppStore } from "../store";
import { GetStartedCard } from "./GettingStarted";
import { UsageDataCard } from "./UsageDataCard";

export function NudgeSlot() {
  const nudge = useAppStore(chooseNudge);
  if (nudge === "get-started") return <GetStartedCard />;
  if (nudge === "usage-data") return <UsageDataCard />;
  return null;
}
