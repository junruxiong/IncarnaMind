export type BarRoom = "viewport" | { x: number; y: number; width: number; height: number };

interface Box {
  left: number;
  right: number;
  bottom: number;
}

/**
 * Where a change's bar may sit: the window, except on the side an assistant docked beside the
 * card takes, where it is the card, since the assistant is drawn over what passes the card's edge.
 */
export function roomForBar(
  card: Box | null,
  dock: string | undefined,
  viewport: { width: number; height: number },
): BarRoom {
  if (!card || (dock !== "right" && dock !== "left" && dock !== "bottom")) return "viewport";
  const left = dock === "left" ? card.left : 0;
  const right = dock === "right" ? card.right : viewport.width;
  return {
    x: left,
    y: 0,
    width: right - left,
    height: dock === "bottom" ? card.bottom : viewport.height,
  };
}
