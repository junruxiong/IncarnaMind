import { type RefObject, useLayoutEffect } from "react";

/** What the page hides, one more at each step: Added first, then Format, then the Tags' chips. */
const STEPS = ["", "added", "added format", "added format tags"] as const;

/** A name wider than this is cut at any width, so it doesn't make the columns give way. */
const WORTH_ROOM = 320;

const isCut = (name: HTMLElement) =>
  name.scrollWidth > name.clientWidth + 1 && name.scrollWidth <= WORTH_ROOM;

/**
 * Hides the columns past a Document's name, Added first, then Format, then the Tags' chips, only as
 * far as it takes for no name to be cut (`data-hide` on the page, read by the columns' classes).
 */
export function useFittingColumns(page: RefObject<HTMLElement | null>): void {
  useLayoutEffect(() => {
    const element = page.current;
    if (!element) return;
    const fit = () => {
      const names = [...element.querySelectorAll<HTMLElement>("[data-fit-name]")];
      for (const step of STEPS) {
        element.dataset.hide = step;
        if (!names.some(isCut)) return;
      }
    };
    fit();
    let frame = 0;
    const later = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(fit);
    };
    const resized = new ResizeObserver(later);
    resized.observe(element);
    for (const child of element.children) resized.observe(child);
    // Rows come and go, and are renamed, without the page changing size.
    const changed = new MutationObserver(later);
    changed.observe(element, { childList: true, subtree: true, characterData: true });
    return () => {
      cancelAnimationFrame(frame);
      resized.disconnect();
      changed.disconnect();
    };
  }, [page]);
}
