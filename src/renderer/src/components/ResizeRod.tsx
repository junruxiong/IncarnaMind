import { type KeyboardEvent, type PointerEvent, useRef } from "react";

interface ResizeRodProps {
  label: string;
  testId?: string;
  /** Current size of the pane this rod resizes: its width, or its height on the "y" axis. */
  size: number;
  min: number;
  max: number;
  /** "x" (the default): a vertical rod between side-by-side panes; "y": a horizontal rod above or below one. */
  axis?: "x" | "y";
  /** 1 if the pane is before the rod (left or above: dragging right or down grows it), -1 if it is after it. */
  direction: 1 | -1;
  /** Called while dragging. */
  onPreview(size: number): void;
  /** Called once with the final size. */
  onCommit(size: number): void;
  onReset?(): void;
  /** Starts below the 44px band the panes' headers share (`.pane-divider--below-band`). */
  belowBand?: boolean;
  /** The 8px gap between the sidebar and the card: no rule, only a grip while pointed at. */
  gap?: boolean;
  /** Dragging this far past `min` and letting go calls this (the pane folds away) instead of resizing. */
  onCollapse?(): void;
  className?: string;
}

/** How far past its minimum a drag must go to fold the pane away. */
const COLLAPSE_PAST_MIN = 40;

/**
 * The divider between two panes: a 1px rule that shows a three-dot grip only
 * while pointed at, dragged or focused (see `.pane-divider`). Draggable, and
 * keyboard-operable with the arrow keys.
 */
export function ResizeRod({
  label,
  testId,
  size,
  min,
  max,
  axis = "x",
  direction,
  onPreview,
  onCommit,
  onReset,
  belowBand = false,
  gap = false,
  onCollapse,
  className,
}: ResizeRodProps) {
  const drag = useRef<{
    start: number;
    startSize: number;
    latest: number;
    collapse: boolean;
  } | null>(null);
  const clamp = (value: number) => Math.round(Math.min(Math.max(value, min), Math.max(min, max)));
  const along = (event: PointerEvent) => (axis === "x" ? event.clientX : event.clientY);

  const startDrag = (event: PointerEvent<HTMLHRElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { start: along(event), startSize: size, latest: size, collapse: false };
  };
  const moveDrag = (event: PointerEvent<HTMLHRElement>) => {
    const current = drag.current;
    if (!current) return;
    const raw = current.startSize + direction * (along(event) - current.start);
    current.collapse = onCollapse !== undefined && raw < min - COLLAPSE_PAST_MIN;
    current.latest = clamp(raw);
    onPreview(current.latest);
  };
  const endDrag = () => {
    const current = drag.current;
    drag.current = null;
    if (current?.collapse) {
      // The size it had is kept, for when it comes back.
      onPreview(current.startSize);
      onCollapse?.();
    } else if (current && current.latest !== current.startSize) onCommit(current.latest);
  };
  const cancelDrag = () => {
    const current = drag.current;
    drag.current = null;
    if (current) onPreview(current.startSize);
  };
  const [less, more] = axis === "x" ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
  const resizeWithKeys = (event: KeyboardEvent<HTMLHRElement>) => {
    const step = event.shiftKey ? 50 : 10;
    const delta = event.key === more ? step : event.key === less ? -step : 0;
    if (delta === 0) return;
    event.preventDefault();
    const next = clamp(size + direction * delta);
    onPreview(next);
    onCommit(next);
  };

  const look = gap
    ? "pane-divider pane-divider--gap"
    : belowBand
      ? "pane-divider pane-divider--below-band"
      : "pane-divider";
  return (
    <hr
      data-testid={testId}
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
      aria-label={label}
      aria-valuenow={size}
      aria-valuemin={min}
      aria-valuemax={max}
      // A focusable separator is the ARIA window-splitter pattern: arrow keys resize.
      tabIndex={0}
      onPointerDown={startDrag}
      onPointerMove={moveDrag}
      onPointerUp={endDrag}
      onPointerCancel={cancelDrag}
      onDoubleClick={onReset}
      onKeyDown={resizeWithKeys}
      className={[look, axis === "y" && "pane-divider--y", className].filter(Boolean).join(" ")}
    />
  );
}
