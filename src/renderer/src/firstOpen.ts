import { firstOpenOf } from "./nudges";
import { useAppStore } from "./store";

// Only while its example is there: a Mind deleted on its own can't leave the shell quiet for good.
export const useFirstOpen = (): boolean =>
  useAppStore(
    (state) =>
      Boolean(state.examples?.mindId) && firstOpenOf(state.settings?.device.gettingStarted),
  );
