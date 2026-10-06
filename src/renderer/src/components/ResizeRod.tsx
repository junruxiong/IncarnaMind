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
}

/** The old app's 3px pane divider with a three-dot grip, draggable and keyboard-operable. */
export function ResizeRod({
  label,
  testId,
  width,
  min,
  max,
  direction,
  onPreview,
  onCommit,
}: ResizeRodProps) {
  const drag = useRef<{ startX: number; startWidth: number; latest: number } | null>(null);
  const clamp = (value: number) => Math.round(Math.min(Math.max(value, min), Math.max(min, max)));

  const startDrag = (event: PointerEvent<HTMLHRElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { startX: event.clientX, startWidth: width, latest: width };
  };
  const moveDrag = (event: PointerEvent<HTMLHRElement>) => {
    const current = drag.current;
    if (!current) return;
    current.latest = clamp(current.startWidth + direction * (event.clientX - current.startX));
    onPreview(current.latest);
  };
  const endDrag = () => {
    const current = drag.current;
    drag.current = null;
    if (current && current.latest !== current.startWidth) onCommit(current.latest);
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
      className="three-dots flex h-auto w-[3px] shrink-0 cursor-col-resize items-center justify-center border-0 bg-gray-200 outline-none focus-visible:bg-gray-300"
    />
  );
}
