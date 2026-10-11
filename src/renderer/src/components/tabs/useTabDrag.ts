import {
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { MAX_TAB_GROUPS } from "../../../../core/api";
import { groupOfTab } from "../../layoutSlice";
import { useAppStore } from "../../store";

/*
 * Dragging a tab, as in VS Code and Chrome: within its strip to reorder, onto
 * another group's strip to move it there (an insertion line shows where), or
 * onto a group's body: its outer 30% splits off a new group on that side (the
 * half it will take lights up), its middle moves it into that group. The
 * tab's menu has the same moves (Move to the left or right group, Open to the
 * side), and ⌥⌘\ splits from the keyboard. Pointer events, so it works the
 * same with a mouse, a pen or a test's sliding mouse.
 */

/** How far the pointer moves, held down, before a press is a drag. */
const DRAG_AFTER_PX = 5;
/** The share of a group's width, from each side, that splits rather than moves. */
const EDGE = 0.3;

export type DropTarget =
  | { type: "strip"; groupId: string; before: string | null; x: number; top: number }
  | {
      type: "zone";
      groupId: string;
      zone: "left" | "right" | "center";
      rect: { left: number; top: number; width: number; height: number };
      /** An edge, with three groups already: it moves rather than splits. */
      full: boolean;
    };

export interface DraggedTab {
  tabId: string;
  x: number;
  y: number;
}

/** Where a tab dragged to (x, y) would go, or null where it can't be dropped. */
function dropTargetAt(x: number, y: number, tabId: string, merged: boolean): DropTarget | null {
  const { layout } = useAppStore.getState();
  if (!layout) return null;
  const under = document.elementFromPoint(x, y);
  const strip = under?.closest<HTMLElement>('[data-testid="tab-strip"]');
  if (strip) {
    const tabs = Array.from(strip.querySelectorAll<HTMLElement>("[data-tab-id]"));
    let index = tabs.findIndex((tab) => {
      const box = tab.getBoundingClientRect();
      return x < box.left + box.width / 2;
    });
    if (index === -1) index = tabs.length;
    const at = tabs[index];
    let before = at?.dataset.tabId ?? null;
    if (before === tabId) before = tabs[index + 1]?.dataset.tabId ?? null;
    // Merged, one strip holds every group's tabs: a tab lands in the group of the tab it goes before.
    const groupId = merged
      ? ((before && groupOfTab(layout, before)?.group.id) ?? layout.groups.at(-1)?.id)
      : strip.dataset.groupId;
    if (!groupId) return null;
    const stripBox = strip.getBoundingClientRect();
    const last = tabs.at(-1)?.getBoundingClientRect();
    const lineX = at ? at.getBoundingClientRect().left : (last?.right ?? stripBox.left + 6);
    return {
      type: "strip",
      groupId,
      before,
      x: Math.max(stripBox.left + 2, Math.min(lineX, stripBox.right - 4)),
      top: stripBox.top + 4,
    };
  }
  const body = under?.closest<HTMLElement>("[data-group-body]");
  const groupId = body?.dataset.groupBody;
  if (!body || !groupId || merged) return null;
  const box = body.getBoundingClientRect();
  const across = (x - box.left) / box.width;
  const edge = across < EDGE ? "left" : across > 1 - EDGE ? "right" : null;
  const full = edge !== null && layout.groups.length >= MAX_TAB_GROUPS;
  const zone = edge && !full ? edge : "center";
  const own = groupOfTab(layout, tabId)?.group;
  // Its own group's middle, or either side of a group it is alone in: nowhere to go.
  if (own?.id === groupId && (zone === "center" || own.tabs.length === 1)) return null;
  return {
    type: "zone",
    groupId,
    zone,
    rect: { left: box.left, top: box.top, width: box.width, height: box.height },
    full,
  };
}

function drop(tabId: string, target: DropTarget): void {
  const { changeLayout } = useAppStore.getState();
  if (target.type === "strip") {
    void changeLayout({ type: "move", tabId, groupId: target.groupId, before: target.before });
  } else if (target.zone === "center") {
    void changeLayout({ type: "move", tabId, groupId: target.groupId, before: null });
  } else {
    void changeLayout({
      type: "split",
      tabId,
      side: target.zone,
      copy: false,
      groupId: target.groupId,
    });
  }
}

export function useTabDrag(merged: boolean) {
  const [dragged, setDragged] = useState<DraggedTab | null>(null);
  const [target, setTarget] = useState<DropTarget | null>(null);
  const pressed = useRef<{ tabId: string; x: number; y: number } | null>(null);
  const live = useRef<{ started: boolean; target: DropTarget | null }>({
    started: false,
    target: null,
  });
  /** Set as a drag ends, so the click it makes doesn't show the tab under the pointer. */
  const ended = useRef(false);
  const mergedNow = useRef(merged);
  useEffect(() => {
    mergedNow.current = merged;
  }, [merged]);

  useEffect(() => {
    const stop = () => {
      pressed.current = null;
      live.current = { started: false, target: null };
      document.body.classList.remove("tab-dragging");
      setDragged(null);
      setTarget(null);
    };
    const onMove = (event: PointerEvent) => {
      const press = pressed.current;
      if (!press) return;
      if (!live.current.started) {
        if (Math.hypot(event.clientX - press.x, event.clientY - press.y) < DRAG_AFTER_PX) return;
        live.current.started = true;
        document.body.classList.add("tab-dragging");
      }
      event.preventDefault();
      const next = dropTargetAt(event.clientX, event.clientY, press.tabId, mergedNow.current);
      live.current.target = next;
      setDragged({ tabId: press.tabId, x: event.clientX, y: event.clientY });
      setTarget(next);
    };
    const onUp = () => {
      const press = pressed.current;
      const { started, target: at } = live.current;
      stop();
      if (!press || !started) return;
      ended.current = true;
      setTimeout(() => {
        ended.current = false;
      });
      if (at) drop(press.tabId, at);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || !live.current.started) return;
      event.preventDefault();
      stop();
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", stop);
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("blur", stop);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", stop);
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("blur", stop);
    };
  }, []);

  const onTabPointerDown = useCallback((event: ReactPointerEvent<HTMLElement>, tabId: string) => {
    if (event.button !== 0) return;
    // Its buttons and its name being typed aren't handles.
    if ((event.target as Element).closest("button, input")) return;
    pressed.current = { tabId, x: event.clientX, y: event.clientY };
  }, []);

  return {
    dragged,
    target,
    onTabPointerDown,
    /** Whether a drag has just ended: the click it makes is not a click on a tab. */
    justDragged: () => ended.current,
  };
}

export type TabDrag = ReturnType<typeof useTabDrag>;
