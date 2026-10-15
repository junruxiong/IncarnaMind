import { type RefObject, useLayoutEffect } from "react";

/** Below this, prose reflows and tables scroll in their frames (DESIGN.md, the Word look). */
export const LEAST_WORD_ZOOM = 0.8;

/**
 * The content's `--word-pt` takes the zoom; the frame's stays whole, so its
 * max-width is still the file's text width to measure the zoom against.
 */
export const wordLookZoomCss = (scope: string) => [
  `${scope} { --word-unit: var(--word-pt); }`,
  `${scope} .mind-editor { --word-pt: calc(var(--word-unit) * var(--word-zoom, 1)); }`,
];

/** Zooms the Word look drawn by `css` (null for none) to fit a frame narrower than the file's text width. */
export function useWordLookZoom(frame: RefObject<HTMLElement | null>, css: string | null): void {
  useLayoutEffect(() => {
    const element = frame.current;
    if (!element || css === null) return;
    const toFit = () => {
      const full = Number.parseFloat(getComputedStyle(element).maxWidth);
      const zoom =
        full > 0
          ? Math.min(
              1,
              Math.max(LEAST_WORD_ZOOM, Math.floor((element.clientWidth / full) * 1000) / 1000),
            )
          : 1;
      element.style.setProperty("--word-zoom", String(zoom));
    };
    toFit();
    const observer = new ResizeObserver(toFit);
    observer.observe(element);
    return () => {
      observer.disconnect();
      element.style.removeProperty("--word-zoom");
    };
  }, [frame, css]);
}
