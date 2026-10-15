import { type RefObject, useCallback, useLayoutEffect, useRef } from "react";
import { slideSidebar } from "./motion";

/**
 * Moves the sidebar with the card's edge as it hides, shows or peeks
 * (`slideSidebar`). Call what it returns just before the change, with whether
 * the sidebar shows after it; the motion plays once the change is laid out,
 * before it is painted. Everything in the workbench may move with the edge:
 * the card, and an assistant docked left of it or below it; the tab band
 * stretches to the card's right edge.
 */
export function useSidebarSlide(
  workbench: RefObject<HTMLElement | null>,
): (shows: boolean) => void {
  const play = useRef<((bands: readonly HTMLElement[]) => void) | null>(null);
  useLayoutEffect(() => {
    const playing = play.current;
    play.current = null;
    if (!playing) return;
    // After the render this one sets off, as the groups merge or part on the card's new width,
    // so what that render draws moves too.
    queueMicrotask(() =>
      playing(Array.from(workbench.current?.querySelectorAll<HTMLElement>(".tab-strip") ?? [])),
    );
  });
  return useCallback(
    (shows: boolean) => {
      const sidebar = document.getElementById("sidebar");
      if (!sidebar || !workbench.current) return;
      const beside = Array.from(workbench.current.children).filter(
        (child): child is HTMLElement => child instanceof HTMLElement,
      );
      play.current = slideSidebar(sidebar, beside, shows);
    },
    [workbench],
  );
}
