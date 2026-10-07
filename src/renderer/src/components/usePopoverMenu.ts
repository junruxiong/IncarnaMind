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

/** The checked item, or the first. */
function focusFirst(menu: HTMLElement | null): void {
  const all = itemsOf(menu);
  (all.find((each) => each.getAttribute("aria-checked") === "true") ?? all[0])?.focus();
}

/**
 * A menu that opens from a button as a popover, so it sits above everything
 * and a click outside or Esc closes it (Esc closes only the menu, not the
 * Document viewer too). It shows below the button, or above it if the window
 * is too short; opening focuses the checked item (or the first), and the
 * arrow keys, Home and End move between items. The menu may render its items
 * only while `open`.
 *
 * Spread `buttonProps` on the button and `menuProps` on the menu's element,
 * which also takes `role="menu"`, an `aria-label` and `menuClass` (./ui).
 */
export function usePopoverMenu() {
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);

  // Once rendered with its items: placed again and focused, if it shows by
  // then. A popover that hasn't opened yet has no height, and takes no focus;
  // its `toggle` event (below) does both once it has.
  useEffect(() => {
    if (!open || !menu.current?.matches(":popover-open")) return;
    place(button.current, menu.current);
    focusFirst(menu.current);
  }, [open]);

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      // Handled here, so the Document viewer's Esc doesn't close it too.
      event.preventDefault();
      menu.current?.hidePopover();
      button.current?.focus();
      return;
    }
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
      // Now it shows: its real height decides above or below, and it can take the focus.
      onToggle: (event: ToggleEvent<HTMLDivElement>) => {
        if (event.newState !== "open") return;
        place(button.current, menu.current);
        focusFirst(menu.current);
      },
      onKeyDown,
    },
  };
}
