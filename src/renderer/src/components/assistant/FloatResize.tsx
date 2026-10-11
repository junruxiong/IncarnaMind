import { type KeyboardEvent, type PointerEvent, useRef } from "react";
import type { AssistantFloat } from "../../../../core/api";
import { FLOAT_MIN, floatAt, type Rect, WINDOW_EDGE } from "../../assistantPlace";
import { useT } from "../../i18n";

/** The edges and corners a floating box resizes from. */
const HANDLES = ["n", "s", "e", "w", "ne", "nw", "se", "sw"] as const;
type Handle = (typeof HANDLES)[number];

const windowSize = () => ({ width: window.innerWidth, height: window.innerHeight });

/** `rect` with the edges `handle` names moved by (dx, dy), between its minimum and the window. */
function resized(rect: Rect, handle: string, dx: number, dy: number): Rect {
  const within = (value: number, min: number, max: number) =>
    Math.round(Math.min(Math.max(value, min), max));
  const right = rect.left + rect.width;
  const bottom = rect.top + rect.height;
  const next = { ...rect };
  if (handle.includes("e")) {
    next.width = within(
      rect.width + dx,
      FLOAT_MIN.width,
      window.innerWidth - WINDOW_EDGE - rect.left,
    );
  }
  if (handle.includes("w")) {
    next.width = within(rect.width - dx, FLOAT_MIN.width, right - WINDOW_EDGE);
    next.left = right - next.width;
  }
  if (handle.includes("s")) {
    next.height = within(
      rect.height + dy,
      FLOAT_MIN.height,
      window.innerHeight - WINDOW_EDGE - rect.top,
    );
  }
  if (handle.includes("n")) {
    next.height = within(rect.height - dy, FLOAT_MIN.height, bottom - WINDOW_EDGE);
    next.top = bottom - next.height;
  }
  return next;
}

/**
 * A floating assistant resizes from any edge or corner, as a window does; its
 * bottom-right corner also takes the arrow keys (← → its width, ↑ ↓ its
 * height). A double-click on any of them restores its default size.
 */
export function FloatResize({
  rect,
  onPreview,
  onCommit,
  onReset,
}: {
  /** Where the box is drawn now. */
  rect: Rect;
  onPreview(float: AssistantFloat): void;
  onCommit(float: AssistantFloat): void;
  onReset(): void;
}) {
  const t = useT();
  const drag = useRef<{ handle: string; x: number; y: number; start: Rect; latest: Rect } | null>(
    null,
  );

  const start = (handle: Handle) => (event: PointerEvent<HTMLElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { handle, x: event.clientX, y: event.clientY, start: rect, latest: rect };
  };
  const move = (event: PointerEvent<HTMLElement>) => {
    const current = drag.current;
    if (!current) return;
    current.latest = resized(
      current.start,
      current.handle,
      event.clientX - current.x,
      event.clientY - current.y,
    );
    onPreview(floatAt(current.latest, windowSize()));
  };
  const end = () => {
    const current = drag.current;
    drag.current = null;
    if (current && current.latest !== current.start)
      onCommit(floatAt(current.latest, windowSize()));
  };
  const cancel = () => {
    const current = drag.current;
    drag.current = null;
    if (current) onPreview(floatAt(current.start, windowSize()));
  };
  const withKeys = (event: KeyboardEvent<HTMLButtonElement>) => {
    const step = event.shiftKey ? 50 : 10;
    const by: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const delta = by[event.key];
    if (!delta) return;
    event.preventDefault();
    onCommit(floatAt(resized(rect, "se", delta[0], delta[1]), windowSize()));
  };

  return HANDLES.map((handle) =>
    handle === "se" ? (
      <button
        key={handle}
        type="button"
        data-testid="assistant-resize-se"
        data-handle={handle}
        aria-label={t("assistant.resize.float")}
        title={t("assistant.resize.float")}
        className="assistant-resize"
        onPointerDown={start(handle)}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={cancel}
        onKeyDown={withKeys}
        onDoubleClick={onReset}
      />
    ) : (
      <div
        key={handle}
        aria-hidden="true"
        data-testid={`assistant-resize-${handle}`}
        data-handle={handle}
        className="assistant-resize"
        onPointerDown={start(handle)}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={cancel}
        onDoubleClick={onReset}
      />
    ),
  );
}
