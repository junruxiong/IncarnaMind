import { type CSSProperties, useLayoutEffect, useMemo, useState } from "react";
import { type AssistantPlace, DEFAULT_ASSISTANT } from "../../../../core/api";
import {
  BOTTOM_MIN,
  bottomHeight,
  bottomMax,
  floatAt,
  floatRect,
  GAP,
  isSide,
  RAIL,
  type Rect,
  restoredSize,
  SIDE_MIN,
  type Size,
  shownDock,
  sideMax,
  sideWidth,
} from "../../assistantPlace";
import { ComposerDockContext } from "../../editor/mindContext";
import { useT } from "../../i18n";
import { useCardWidth } from "../../layoutSlice";
import { useAppStore } from "../../store";
import { useWindowSize } from "../../useWindowWidth";
import { ResizeRod } from "../ResizeRod";
import { TabGroups } from "../tabs/TabGroups";
import { Assistant } from "./Assistant";
import { FloatResize } from "./FloatResize";
import { SnapZones } from "./SnapZones";
import { useAssistantDrag } from "./useAssistantDrag";

/**
 * Everything right of the sidebar: the card with the tab groups, and the
 * window's one assistant beside it (#253), docked on the right (by default),
 * the left or the bottom, with the 8px line between them that resizes it, or
 * floating over the card. A side dock is drawn at the bottom while the window
 * is under 1,000px.
 *
 * The card and the assistant keep their places in the page whatever the dock
 * (only the flex direction changes), so nothing in either is remounted: the
 * Mind's editor, the composer's draft and a streaming Answer stay as they are.
 * Each change is laid out at once at its new size; the card is measured, and
 * the groups share what it is given (`useGroupsMerged`).
 */
export function Workbench({ sidebarHidden }: { sidebarHidden: boolean }) {
  const t = useT();
  const place = useAppStore((state) => state.layout?.assistant) ?? DEFAULT_ASSISTANT;
  const placeAssistant = useAppStore((state) => state.placeAssistant);
  const viewport = useWindowSize();
  /** Sizes while the User drags a line or an edge, before they are kept. */
  const [preview, setPreview] = useState<Partial<AssistantPlace> | null>(null);
  const shown: AssistantPlace = preview ? { ...place, ...preview } : place;
  const dock = shownDock(shown, viewport.width);
  const side = isSide(dock);
  const { folded } = shown;
  const [area, setArea] = useState<HTMLDivElement | null>(null);
  const room = useContentSize(area);
  const [card, setCard] = useState<HTMLDivElement | null>(null);
  useMeasuredCard(card);
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  // The same element every time: moving or resizing the assistant draws the groups again only
  // as far as their own measurements change, so it keeps up with the pointer.
  const groups = useMemo(() => <TabGroups />, []);

  const width = sideWidth(shown.width, room.width);
  const height = bottomHeight(shown.height, room.height);
  const float = floatRect(shown.float, folded, viewport);
  const atWindowLeft = dock === "left" && sidebarHidden;

  const drag = useAssistantDrag({
    area: () => contentRect(area),
    floatSize: { width: float.width, height: floatRect(shown.float, false, viewport).height },
    place: { width: shown.width, height: shown.height },
    onDrop: (drop) => {
      if (drop.dock === "float") {
        void placeAssistant({ dock: "float", float: floatAt(drop.rect, viewport) });
      } else void placeAssistant({ dock: drop.dock });
    },
  });

  const style: CSSProperties =
    dock === "float"
      ? { left: float.left, top: float.top, width: float.width, height: float.height }
      : side
        ? { width: folded ? RAIL : width }
        : folded
          ? {}
          : { height };
  /** Room the assistant takes from the card on its side, for what the window draws there. */
  const beside = side ? (folded ? RAIL : width) + GAP : 0;

  const resize = (next: Partial<AssistantPlace>) => setPreview(next);
  const keep = (next: Partial<AssistantPlace>) => {
    setPreview(null);
    void placeAssistant(next);
  };
  const restore = () => keep(restoredSize(shown, dock));

  return (
    <div
      ref={setArea}
      data-testid="workbench"
      data-dock={dock}
      data-folded={folded || undefined}
      data-sidebar-hidden={sidebarHidden || undefined}
      className="workbench"
      style={
        {
          "--assistant-left": `${dock === "left" ? beside : 0}px`,
          "--assistant-right": `${dock === "right" ? beside : 0}px`,
        } as CSSProperties
      }
    >
      <div
        ref={setCard}
        data-testid="card"
        data-sidebar-hidden={sidebarHidden || undefined}
        data-assistant-right={dock === "right" || undefined}
        className="app-card"
      >
        <ComposerDockContext.Provider value={slot}>{groups}</ComposerDockContext.Provider>
      </div>
      {dock === "float" ? null : folded ? (
        <div aria-hidden="true" className="workbench-gap" />
      ) : (
        <ResizeRod
          testId="assistant-resize"
          label={t(side ? "assistant.resize.side" : "assistant.resize.bottom")}
          axis={side ? "x" : "y"}
          gap
          width={side ? width : height}
          min={side ? SIDE_MIN : BOTTOM_MIN}
          max={side ? sideMax(room.width) : bottomMax(room.height)}
          // The assistant is after the line, except on the left, where it is before it.
          direction={dock === "left" ? 1 : -1}
          onPreview={(px) => resize(side ? { width: px } : { height: px })}
          onCommit={(px) => keep(side ? { width: px } : { height: px })}
          onReset={restore}
        />
      )}
      <Assistant
        dock={dock}
        folded={folded}
        style={style}
        attributes={{
          "data-chosen-dock": place.dock,
          "data-at-window-left": atWindowLeft ? "" : undefined,
        }}
        onComposerSlot={setSlot}
        onHandlePointerDown={drag.onPointerDown}
        justMoved={drag.justDragged}
      >
        {dock === "float" && !folded && (
          <FloatResize
            rect={float}
            onPreview={(next) => resize({ float: next })}
            onCommit={(next) => keep({ float: next })}
            onReset={restore}
          />
        )}
      </Assistant>
      <SnapZones dragging={drag.dragging} standIn={drag.standIn} />
    </div>
  );
}

/** An element's content box in the window: where the card and the assistant are. */
function contentRect(element: HTMLElement | null): Rect | null {
  if (!element) return null;
  const box = element.getBoundingClientRect();
  const style = getComputedStyle(element);
  const left = Number.parseFloat(style.paddingLeft);
  const top = Number.parseFloat(style.paddingTop);
  return {
    left: box.left + left,
    top: box.top + top,
    width: box.width - left - Number.parseFloat(style.paddingRight),
    height: box.height - top - Number.parseFloat(style.paddingBottom),
  };
}

/** The size of an element's content box, followed as it changes. */
function useContentSize(element: HTMLElement | null): Size {
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  useLayoutEffect(() => {
    if (!element) return;
    const measure = () => {
      const rect = contentRect(element);
      if (rect) setSize({ width: rect.width, height: rect.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [element]);
  return size;
}

/** Keeps the card's width where the groups read it (`useCardWidth`). */
function useMeasuredCard(card: HTMLElement | null): void {
  useLayoutEffect(() => {
    if (!card) return;
    const measure = () => useCardWidth.setState({ width: card.getBoundingClientRect().width });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(card);
    return () => observer.disconnect();
  }, [card]);
}
