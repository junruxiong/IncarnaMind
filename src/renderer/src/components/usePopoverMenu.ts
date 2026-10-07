import { type KeyboardEvent, type ToggleEvent, useEffect, useId, useRef, useState } from "react";

const GAP_PX = 4;
const EDGE_PX = 8;
/** Menu items of every kind: plain, radio and checkbox. */
const ITEMS = '[role^="menuitem"]';

const itemsOf = (menu: HTMLElement | null) =>
  Array.from(menu?.querySelectorAll<HTMLElement>(ITEMS) ?? []);

/** Below the button, or above it if the window is too short. */
function place(button: HTMLElement | null, menu: HTMLElement | null): void {
  const anchor = button?.getBoundingClientRect();
  if (!anchor || !menu) return;
  menu.style.left = `${Math.max(EDGE_PX, anchor.left)}px`;
  const below = anchor.bottom + GAP_PX;
  const height = menu.offsetHeight;
  menu.style.top =
    below + height > window.innerHeight - EDGE_PX
      ? `${Math.max(EDGE_PX, anchor.top - GAP_PX - height)}px`
      : `${below}px`;
}

/**
 * A menu that opens from a button as a popover, so it sits above everything
 * and a click outside or Esc closes it. It shows below the button, or above
 * it if the window is too short; opening focuses the checked item (or the
 * first), and the arrow keys, Home and End move between items. The menu may
 * render its items only while `open`.
 *
 * Spread `buttonProps` on the button and `menuProps` on the menu's element,
 * which also takes `role="menu"`, an `aria-label` and `menuClass` (./ui).
 */
export function usePopoverMenu() {
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);

  // Once open and rendered: placed again, now its height is known, and focused.
  useEffect(() => {
    if (!open) return;
    place(button.current, menu.current);
    const all = itemsOf(menu.current);
    (all.find((each) => each.getAttribute("aria-checked") === "true") ?? all[0])?.focus();
  }, [open]);

  const moveFocus = (event: KeyboardEvent) => {
    const all = itemsOf(menu.current);
    const index = all.indexOf(document.activeElement as HTMLElement);
    const next =
      event.key === "ArrowDown"
        ? all[(index + 1) % all.length]
        : event.key === "ArrowUp"
          ? all[(index - 1 + all.length) % all.length]
          : event.key === "Home"
            ? all[0]
            : event.key === "End"
              ? all.at(-1)
              : undefined;
    if (!next) return;
    event.preventDefault();
    next.focus();
  };

  return {
    open,
    close: () => menu.current?.hidePopover(),
    buttonProps: {
      ref: button,
      popoverTarget: id,
      "aria-haspopup": "menu" as const,
      "aria-expanded": open,
    },
    menuProps: {
      ref: menu,
      id,
      popover: "auto" as const,
      // Before it shows, so a menu that renders its items only while open has them when it paints.
      onBeforeToggle: (event: ToggleEvent<HTMLDivElement>) => {
        if (event.newState === "open") place(button.current, menu.current);
        setOpen(event.newState === "open");
      },
      onKeyDown: moveFocus,
    },
  };
}
