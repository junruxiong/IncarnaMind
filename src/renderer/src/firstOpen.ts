import { stepsShownOf } from "./nudges";
import { useAppStore } from "./store";

// Quiet while the steps card shows, done or not: each part of the shell comes on its own trigger,
// never all at once when the last step ticks (#1027).
export const useFirstOpen = (): boolean =>
  useAppStore((state) => stepsShownOf(state.settings?.device.gettingStarted));
