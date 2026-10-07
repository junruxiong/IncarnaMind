import { createContext } from "react";

/** Where a PDF's view was: the page in view, and its zoom (null: fitted to the width). */
export interface ViewPlace {
  page: number;
  zoom: number | null;
}

/**
 * Keeps a view's place while its Document's file is read again (a new
 * version, or the file back after it went), so the view comes back where it
 * was rather than at the top. Only for the same open request: opening the
 * Document again, or another one, starts afresh. See `ViewerPanel`.
 */
export interface PlaceMemory {
  /** The place kept for this open request, if any. */
  recall(): ViewPlace | null;
  remember(place: ViewPlace): void;
}

export const PlaceMemoryContext = createContext<PlaceMemory>({
  recall: () => null,
  remember: () => undefined,
});
