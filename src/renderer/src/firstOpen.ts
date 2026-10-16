import { gettingStartedShown } from "./nudges";
import { useAppStore } from "./store";

export const useFirstOpen = (): boolean =>
  useAppStore((state) => gettingStartedShown(state.settings?.device.gettingStarted));
