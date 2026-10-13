import { type RefObject, useEffect } from "react";

/** How near the end counts as being at it. */
const AT_END_PX = 24;

/**
 * Keeps what the User was looking at in view when the thread changes size: the
 * assistant moving to another dock or floating, unfolding, a resize of it or of
 * the window (#612). A thread at its end stays at its end, so the latest
 * Answer's last lines and its actions are not left under the composer; one
 * scrolled up keeps the turn at its top where it was.
 */
export function useKeepInView(thread: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const scroller = thread.current;
    if (!scroller) return;
    let atEnd = true;
    let anchor: { turn: Element; offset: number } | null = null;
    // Our own scroll is not the User's: the layout may move on before its event arrives.
    let ours = false;

    const remember = () => {
      if (ours) {
        ours = false;
        return;
      }
      if (scroller.clientHeight === 0) return;
      atEnd = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= AT_END_PX;
      const top = scroller.getBoundingClientRect().top;
      const turn = [...scroller.querySelectorAll("[data-answer-id], .assistant-question")].find(
        (each) => each.getBoundingClientRect().bottom > top + 1,
      );
      anchor = turn ? { turn, offset: turn.getBoundingClientRect().top - top } : null;
    };
    const restore = () => {
      if (scroller.clientHeight === 0) return;
      const was = scroller.scrollTop;
      if (atEnd) {
        scroller.scrollTop = scroller.scrollHeight;
      } else if (anchor?.turn.isConnected) {
        const top = scroller.getBoundingClientRect().top;
        scroller.scrollTop += anchor.turn.getBoundingClientRect().top - top - anchor.offset;
      }
      if (scroller.scrollTop !== was) {
        ours = true;
        requestAnimationFrame(() => {
          ours = false;
        });
      }
    };

    remember();
    scroller.addEventListener("scroll", remember, { passive: true });
    const resized = new ResizeObserver(restore);
    resized.observe(scroller);
    if (scroller.firstElementChild) resized.observe(scroller.firstElementChild);
    return () => {
      scroller.removeEventListener("scroll", remember);
      resized.disconnect();
    };
  }, [thread]);
}
