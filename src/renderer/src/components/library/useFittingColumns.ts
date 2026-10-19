import { type RefObject, useLayoutEffect } from "react";

/**
 * What the page gives up, one more at each step: Added, then Format, then the Tags' room (the
 * words the columns' classes read), and last the chips themselves.
 */
const STEPS = [
  "",
  "added",
  "added format",
  "added format narrow",
  "added format narrower",
  "added format narrowest",
  "added format chips",
] as const;

/** A name that would take more than this is asked for no more: one long name doesn't clear the columns. */
const WORTH_ROOM = 320;

/** What a name wider than that is asked for: enough to read its start. */
const LEAST_ROOM = 160;

const isCut = (name: HTMLElement) =>
  name.clientWidth + 1 < (name.scrollWidth <= WORTH_ROOM ? name.scrollWidth : LEAST_ROOM);

/**
 * Gives up the columns past a Document's name, Added first, then Format, then some of the Tags'
 * room, then their chips, only as far as it takes for no name to be cut (`data-hide` on the page,
 * read by the columns' classes), so every row's columns keep the same edges.
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
