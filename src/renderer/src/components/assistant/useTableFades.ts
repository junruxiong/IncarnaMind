import { type RefObject, useEffect } from "react";

/**
 * A table wider than its Answer scrolls sideways in its frame (`.tableWrapper`):
 * each side that still has table beyond it is marked, and the styles fade it, so
 * the cut edge reads as more to come (#612).
 */
export function useTableFades(frame: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const root = frame.current;
    if (!root) return;
    const mark = () => {
      for (const wrapper of root.querySelectorAll<HTMLElement>(".tableWrapper")) {
        const start = wrapper.scrollLeft;
        const end = wrapper.scrollWidth - wrapper.clientWidth - start;
        wrapper.toggleAttribute("data-more-before", start > 1);
        wrapper.toggleAttribute("data-more-after", end > 1);
      }
    };
    mark();
    // Scrolling doesn't bubble: it is heard on the way down.
    root.addEventListener("scroll", mark, true);
    const resized = new ResizeObserver(mark);
    resized.observe(root);
    const changed = new MutationObserver(mark);
    changed.observe(root, { childList: true, subtree: true });
    return () => {
      root.removeEventListener("scroll", mark, true);
      resized.disconnect();
      changed.disconnect();
    };
  }, [frame]);
}
