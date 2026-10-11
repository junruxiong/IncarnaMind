import { useSyncExternalStore } from "react";
import { MERGE_BELOW } from "./layoutSlice";

const subscribeToResize = (onChange: () => void) => {
  window.addEventListener("resize", onChange);
  return () => window.removeEventListener("resize", onChange);
};

/** The window's width, followed as it is resized. */
export const useWindowWidth = () =>
  useSyncExternalStore(subscribeToResize, () => window.innerWidth);

/** Whether the window is too narrow for groups side by side: they merge into one strip. */
export const useGroupsMerged = () =>
  useSyncExternalStore(subscribeToResize, () => window.innerWidth < MERGE_BELOW);
