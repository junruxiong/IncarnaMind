import { chooseNudge } from "../nudges";
import { useAppStore } from "../store";
import { GetStartedCard } from "./GettingStarted";
import { UsageDataCard } from "./UsageDataCard";

/** The sidebar's one place for nudges: the card `chooseNudge` picks, nothing else. */
export function NudgeSlot() {
  const nudge = useAppStore(chooseNudge);
  if (nudge === "get-started") return <GetStartedCard />;
  if (nudge === "usage-data") return <UsageDataCard />;
  return null;
}
