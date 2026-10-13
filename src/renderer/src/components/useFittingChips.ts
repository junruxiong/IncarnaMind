import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

/**
 * How many of a toolbar's Tag chips the room holds, at most `most`. The row of
 * chips sits in a room that shrinks with the window, after a name that gives
 * way first: chips that don't fit are hidden, last first, and fold into "+N".
 * The row's chips and its "+N" are the `[data-tag-chip]` and `[data-tag-more]`
 * in it; `total` is how many Tags the Document has.
 */
export function useFittingChips(total: number, most: number) {
  const row = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(most);

  // After every render, so a rename or another language is measured too.
  useLayoutEffect(() => {
    if (row.current) setShown(foldToFit(row.current, total, most));
  });

  useEffect(() => {
    const room = row.current?.parentElement;
    if (!room) return;
    const observer = new ResizeObserver(() => {
      if (row.current) {
        const fits = foldToFit(row.current, total, most);
        flushSync(() => setShown(fits));
      }
    });
    observer.observe(room);
    return () => observer.disconnect();
  }, [total, most]);

  return { row, shown: Math.min(shown, total, most) };
}

/**
 * Hides chips, last first, until the row ends inside its parent, then "+N" too
 * if even it doesn't fit; returns how many chips are left.
 */
function foldToFit(row: HTMLElement, total: number, most: number): number {
  const room = row.parentElement;
  const chips = [...row.querySelectorAll<HTMLElement>("[data-tag-chip]")];
  const more = row.querySelector<HTMLElement>("[data-tag-more]");
  const overflows = () =>
    room !== null && row.getBoundingClientRect().right - room.getBoundingClientRect().right > 0.01;
  for (let shown = Math.min(chips.length, most); ; shown--) {
    chips.forEach((chip, index) => {
      chip.style.display = index < shown ? "" : "none";
    });
    if (more) more.style.display = shown < total ? "" : "none";
    if (!overflows()) return shown;
    if (shown === 0) {
      if (more) more.style.display = "none";
      return 0;
    }
  }
}
