import { type KeyboardEvent, type PointerEvent, useRef } from "react";

interface ResizeRodProps {
  label: string;
  testId?: string;
  /** Current width of the pane this rod resizes. */
  width: number;
  min: number;
  max: number;
  /** 1 if the pane is left of the rod (dragging right grows it), -1 if it is right of the rod. */
  direction: 1 | -1;
  /** Called while dragging. */
  onPreview(width: number): void;
  /** Called once with the final width. */
  onCommit(width: number): void;
  /** Starts below the 44px band the panes' headers share (`.pane-divider--below-band`). */
  belowBand?: boolean;
  /** The 8px gap between the sidebar and the card: no rule, only a grip while pointed at. */
  gap?: boolean;
  /** Dragging this far past `min` and letting go calls this (the pane folds away) instead of resizing. */
  onCollapse?(): void;
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
  width,
  min,
  max,
  direction,
  onPreview,
  onCommit,
  belowBand = false,
  gap = false,
  onCollapse,
}: ResizeRodProps) {
  const drag = useRef<{
    startX: number;
    startWidth: number;
    latest: number;
    collapse: boolean;
  } | null>(null);
  const clamp = (value: number) => Math.round(Math.min(Math.max(value, min), Math.max(min, max)));

  const startDrag = (event: PointerEvent<HTMLHRElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { startX: event.clientX, startWidth: width, latest: width, collapse: false };
  };
  const moveDrag = (event: PointerEvent<HTMLHRElement>) => {
    const current = drag.current;
    if (!current) return;
    const raw = current.startWidth + direction * (event.clientX - current.startX);
    current.collapse = onCollapse !== undefined && raw < min - COLLAPSE_PAST_MIN;
    current.latest = clamp(raw);
    onPreview(current.latest);
  };
  const endDrag = () => {
    const current = drag.current;
    drag.current = null;
    if (current?.collapse) {
      // The width it had is kept, for when it comes back.
      onPreview(current.startWidth);
      onCollapse?.();
    } else if (current && current.latest !== current.startWidth) onCommit(current.latest);
  };
  const cancelDrag = () => {
    const current = drag.current;
    drag.current = null;
    if (current) onPreview(current.startWidth);
  };
  const resizeWithKeys = (event: KeyboardEvent<HTMLHRElement>) => {
    const step = event.shiftKey ? 50 : 10;
    const delta = event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0;
    if (delta === 0) return;
    event.preventDefault();
    const next = clamp(width + direction * delta);
    onPreview(next);
    onCommit(next);
  };

  return (
    <hr
      data-testid={testId}
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={width}
      aria-valuemin={min}
      aria-valuemax={max}
      // A focusable separator is the ARIA window-splitter pattern: arrow keys resize.
      tabIndex={0}
      onPointerDown={startDrag}
      onPointerMove={moveDrag}
      onPointerUp={endDrag}
      onPointerCancel={cancelDrag}
      onKeyDown={resizeWithKeys}
      className={
        gap
          ? "pane-divider pane-divider--gap"
          : belowBand
            ? "pane-divider pane-divider--below-band"
            : "pane-divider"
      }
    />
  );
}
