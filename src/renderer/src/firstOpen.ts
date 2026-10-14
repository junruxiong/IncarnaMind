import type { GettingStarted } from "../../core/api";
import { useAppStore } from "./store";

export const firstOpenOf = (progress: GettingStarted | undefined): boolean =>
  progress?.started === true &&
  !progress.hidden &&
  !(progress.citationChecked && progress.indexed && progress.askedOwn);

export const useFirstOpen = (): boolean =>
  useAppStore((state) => firstOpenOf(state.settings?.device.gettingStarted));
