import type { GettingStarted } from "../../core/api";
import { useAppStore } from "./store";

/** The first open: Get started has begun, and its three steps aren't all done or hidden. */
export const firstOpenOf = (progress: GettingStarted | undefined): boolean =>
  progress !== undefined &&
  progress.started &&
  !progress.hidden &&
  !(progress.citationChecked && progress.indexed && progress.askedOwn);

export const useFirstOpen = (): boolean =>
  useAppStore((state) => firstOpenOf(state.settings?.device.gettingStarted));
