import { firstOpenOf } from "./nudges";
import { useAppStore } from "./store";

export const useFirstOpen = (): boolean =>
  useAppStore((state) => firstOpenOf(state.settings?.device.gettingStarted));
