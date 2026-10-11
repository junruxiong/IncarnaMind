import type { StyledRun } from "../api";

/** A run as the engine returns it: in a citation, `item` is the index of the reference it writes. */
export interface EngineRun extends StyledRun {
  item?: number;
}
