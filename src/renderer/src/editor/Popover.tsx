import {
  autoUpdate,
  computePosition,
  flip,
  offset,
  type Placement,
  shift,
  type VirtualElement,
} from "@floating-ui/dom";
import {
  type HTMLAttributes,
  type ReactNode,
  type RefObject,
  useLayoutEffect,
  useRef,
} from "react";

interface PopoverProps extends HTMLAttributes<HTMLDivElement> {
  /** What the popover sits next to. It follows it as the Mind scrolls. */
  anchor: Element | VirtualElement;
  placement?: Placement;
  /**
   * Called on Esc, wherever the focus is, or on a press outside both the
   * popover and `anchor`.
   */
  onDismiss(): void;
  /** What gets the focus once the popover shows, e.g. a menu's first item. */
  initialFocus?: RefObject<HTMLElement | null>;
  children: ReactNode;
}

/** A native popover in the top layer, next to `anchor`, for the editor's small menus and fields. */
export function Popover({
  anchor,
  placement = "bottom-start",
  onDismiss,
  initialFocus,
  children,
  className = "",
  role = "dialog",
  onKeyDown,
  ...props
}: PopoverProps) {
  const ref = useRef<HTMLDivElement>(null);
  const dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;
  const focusFirst = useRef(initialFocus);
  focusFirst.current = initialFocus;

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    element.showPopover();
    let shown = false;
    const place = () => {
      void computePosition(anchor, element, {
        placement,
        strategy: "fixed",
        middleware: [offset(4), flip(), shift({ padding: 8 })],
      }).then(({ x, y }) => {
        Object.assign(element.style, { left: `${x}px`, top: `${y}px`, visibility: "visible" });
        // Hidden until placed, so it can only take the focus now.
        if (!shown) focusFirst.current?.current?.focus();
        shown = true;
      });
    };
    const stopFollowing = autoUpdate(anchor, element, place);
    const dismissOnPressOutside = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node) || element.contains(target)) return;
      if (anchor instanceof Element && anchor.contains(target)) return;
      dismiss.current();
    };
    // Esc while the focus is elsewhere, e.g. still in the editor or on the
    // button that opened it. Captured, and marked handled, before the
    // Document viewer's own Esc sees it; inside, `onKeyDown` handles it.
    const dismissOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      if (event.target instanceof Node && element.contains(event.target)) return;
      event.preventDefault();
      dismiss.current();
    };
    document.addEventListener("pointerdown", dismissOnPressOutside, true);
    document.addEventListener("keydown", dismissOnEscape, true);
    return () => {
      stopFollowing();
      document.removeEventListener("pointerdown", dismissOnPressOutside, true);
      document.removeEventListener("keydown", dismissOnEscape, true);
      if (element.matches(":popover-open")) element.hidePopover();
    };
  }, [anchor, placement]);

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: it has a role, a dialog's unless the caller gives another (e.g. menu).
    <div
      ref={ref}
      popover="manual"
      role={role}
      {...props}
      onKeyDown={(event) => {
        onKeyDown?.(event);
        if (event.key === "Escape" && !event.defaultPrevented) {
          // Handled here, so Esc doesn't also close the Document viewer.
          event.preventDefault();
          dismiss.current();
        }
      }}
      className={`editor-popover ${className}`}
    >
      {children}
    </div>
  );
}
