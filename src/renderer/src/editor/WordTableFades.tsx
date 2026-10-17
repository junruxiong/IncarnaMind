import type { RefObject } from "react";
import { useTableFades } from "../components/useTableFades";

/** In a Mind with the Word look, a table frame that scrolls has its cut edges faded. */
export function WordTableFades({ frame }: { frame: RefObject<HTMLElement | null> }) {
  useTableFades(frame, null);
  return null;
}
