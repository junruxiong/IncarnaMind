import {
  type KeyboardEvent,
  type ToggleEvent,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

const GAP_PX = 4;
const EDGE_PX = 8;
/** Menu items of every kind: plain, radio and checkbox. */
const ITEMS = '[role^="menuitem"]';

const itemsOf = (menu: HTMLElement | null) =>
  Array.from(menu?.querySelectorAll<HTMLElement>(ITEMS) ?? []);

/**
 * Below the button, or above it if the window is too short. A popover that
 * isn't showing yet isn't laid out (it has no size), so it is laid out for a
 * moment to measure it as it will show.
 */
function place(button: HTMLElement | null, menu: HTMLElement | null): void {
  const anchor = button?.getBoundingClientRect();
  if (!anchor || !menu) return;
  const hidden = !menu.matches(":popover-open");
  if (hidden) menu.style.display = "block";
  const width = menu.offsetWidth;
  const height = menu.offsetHeight;
  if (hidden) menu.style.removeProperty("display");
  menu.style.left = `${Math.max(EDGE_PX, Math.min(anchor.left, window.innerWidth - width - EDGE_PX))}px`;
  const below = anchor.bottom + GAP_PX;
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

  // Once rendered with its items, and before it is drawn: placed. A click
  // renders it just before the popover shows, so it shows where it belongs;
  // its `toggle` event comes only after a frame is drawn, too late to place it.
  // It takes the focus once it shows: here if it already does, or else on that
  // `toggle` event (below), as a popover that hasn't shown can't be focused.
  useLayoutEffect(() => {
    if (!open) return;
    place(button.current, menu.current);
    if (menu.current?.matches(":popover-open")) focusFirst(menu.current);
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
        setOpen(event.newState === "open");
      },
      // Now it shows, where it was placed: it can take the focus.
      onToggle: (event: ToggleEvent<HTMLDivElement>) => {
        if (event.newState === "open") focusFirst(menu.current);
      },
      onKeyDown,
    },
  };
}
