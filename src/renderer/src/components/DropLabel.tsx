import { useDragState } from "../dragDrop";
import { useT } from "../i18n";

/** What a drop on the target under the pointer would do, in a dark label under the drag's picture. */
export function DropLabel() {
  const t = useT();
  const said = useDragState((state) => state.over?.effect.said ?? null);
  const at = useDragState((state) => state.at);
  if (!said || !at) return null;
  return (
    <div
      data-testid="drop-label"
      aria-hidden="true"
      className="pointer-events-none fixed z-[70] max-w-72 truncate rounded-md bg-ink px-2 py-1 text-[12px] leading-4 font-semibold text-white shadow-[0_4px_12px_rgba(23,25,28,0.18)]"
      style={{ left: at.x - 12, top: at.y + 20 }}
    >
      {said(t)}
    </div>
  );
}
