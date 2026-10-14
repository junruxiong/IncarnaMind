import type { GettingStarted } from "../core/api";

export const firstOpenOf = (progress: GettingStarted | undefined): boolean =>
  progress?.started === true &&
  !progress.hidden &&
  !(progress.citationChecked && progress.indexed && progress.askedOwn);
