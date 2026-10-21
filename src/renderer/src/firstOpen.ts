import { stepsShownOf } from "./nudges";
import { useAppStore } from "./store";

// Quiet while the steps card shows, done or not, and only while its example is there: each part of
// the shell comes on its own trigger (#1027), and a Mind deleted on its own can't leave it quiet.
export const useFirstOpen = (): boolean =>
  useAppStore(
    (state) =>
      Boolean(state.examples?.mindId) && stepsShownOf(state.settings?.device.gettingStarted),
  );
