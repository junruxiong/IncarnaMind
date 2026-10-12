import { createContext, useContext, useEffect } from "react";
import type { PageArea } from "../../../shared/citedPage";

/** Where a view drew the cited Units: its elements, or a range of the page. */
export type CitedArea = Range | readonly Element[];

/**
 * Given by the hidden window that draws a cited page to save it (#133): a
 * view says where it drew the cited area once it has, or why it can't draw
 * its file. The app's own viewer gives none, so nothing changes there.
 */
export interface CitedAreaCapture {
  /** What of the cited Units to show: a sheet's cited rows, or all of them. */
  area: PageArea;
  drawn(area: () => CitedArea | null): void;
  failed(reason: string): void;
}

export const CitedAreaContext = createContext<CitedAreaCapture | null>(null);

/** Says where the cited area is, once `ready`, while a cited page is drawn to save it. */
export function useCitedArea(ready: boolean, area: (asked: PageArea) => CitedArea | null): void {
  const capture = useContext(CitedAreaContext);
  useEffect(() => {
    if (capture && ready) capture.drawn(() => area(capture.area));
  }, [capture, ready, area]);
}

/** Says why the file can't be drawn (null: it can), while a cited page is drawn to save it. */
export function useCitedAreaFailure(reason: string | null): void {
  const capture = useContext(CitedAreaContext);
  useEffect(() => {
    if (capture && reason !== null) capture.failed(reason);
  }, [capture, reason]);
}
