import type { DragEvent } from "react";
import { create } from "zustand";
import { startDocumentDrag } from "./documentDrag";
import { itemsOf, type MoveItems } from "./moves";
import { useSelection } from "./selection";
import { useAppStore } from "./store";
import { carriesLinks, linksIn, useWebPages } from "./webPages";

/**
 * Dragging Minds and Documents in the sidebar onto a Folder, or onto "Not in
 * a Folder" to take them out of theirs: one row, or every row selected with
 * it, Minds and Documents mixed. What is dragged travels as its ids under a
 * type of IncarnaMind's own, so the window's file drop and the note ignore
 * it; Documents also travel as the composer takes them. A target lights up
 * only while the drop would move something there; dropping moves, as "Move
 * to…" does (`moveToFolder`). A link dragged from a browser is saved there as
 * a web page (#261).
 */
export const MOVE_TYPE = "application/x-incarnamind-move";

const OPENS_AFTER_MS = 700;

// A drag's data can only be read on the drop, so what is dragged is kept here for the targets it passes over.
let dragging: { mindIds: string[]; documentIds: string[] } | null = null;

/** Whether something in a Folder is being dragged: Not in a Folder then shows, empty or not, to take it. */
export const useMoveDrag = create<{ outOfFolder: boolean }>()(() => ({ outOfFolder: false }));

export function startMoveDrag(
  event: DragEvent<HTMLElement>,
  items: MoveItems,
  label: string,
): void {
  const moving = itemsOf(items);
  dragging = { mindIds: moving.mindIds, documentIds: moving.documentIds };
  const outOfFolder = wouldMove(null);
  // Not while the drag starts: a page that changes then makes the browser give the drag up.
  if (outOfFolder) requestAnimationFrame(() => useMoveDrag.setState({ outOfFolder }));
  if (moving.documentIds.length > 0) startDocumentDrag(event, moving.documentIds);
  event.dataTransfer.setData(MOVE_TYPE, JSON.stringify(dragging));
  event.dataTransfer.effectAllowed = moving.documentIds.length > 0 ? "linkMove" : "move";
  const count = moving.mindIds.length + moving.documentIds.length;
  setPicture(event, label, count);
}

function setPicture(event: DragEvent<HTMLElement>, label: string, count: number): void {
  const picture = document.createElement("div");
  picture.className =
    "fixed top-0 -left-[2000px] flex h-7 max-w-64 items-center gap-2 rounded-md bg-sheet px-2.5 text-[13px] text-ink shadow-[0_0_0_1px_var(--color-rule),0_12px_32px_rgba(23,25,28,0.12)]";
  const icon = event.currentTarget.querySelector("svg")?.cloneNode(true);
  if (icon instanceof SVGElement) {
    icon.setAttribute("class", "size-4 shrink-0 text-ink-meta");
    picture.append(icon);
  }
  const name = document.createElement("span");
  name.className = "min-w-0 truncate";
  name.textContent = label;
  picture.append(name);
  if (count > 1) {
    const badge = document.createElement("span");
    badge.className =
      "inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-accent px-1.5 text-[11px] font-semibold text-white";
    badge.textContent = String(count);
    picture.append(badge);
  }
  document.body.append(picture);
  event.dataTransfer.setDragImage(picture, 12, 14);
  // The browser copies the picture as the drag starts; it isn't needed after.
  requestAnimationFrame(() => picture.remove());
}

const carriesMove = (event: DragEvent) => event.dataTransfer.types.includes(MOVE_TYPE);

/** What a drop carries, or null if it isn't a move of IncarnaMind's. */
export function droppedItems(event: DragEvent): MoveItems | null {
  try {
    const value = JSON.parse(event.dataTransfer.getData(MOVE_TYPE)) as unknown;
    if (typeof value !== "object" || value === null) return null;
    const { mindIds, documentIds } = value as Record<string, unknown>;
    const ids = (list: unknown) =>
      Array.isArray(list) ? list.filter((id): id is string => typeof id === "string") : [];
    return { mindIds: ids(mindIds), documentIds: ids(documentIds) };
  } catch {
    return null;
  }
}

function wouldMove(folderId: string | null): boolean {
  if (!dragging) return false;
  const { minds, library } = useAppStore.getState();
  const folderOf = new Map(library?.assignments.map((each) => [each.documentId, each.groupId]));
  return (
    dragging.mindIds.some(
      (id) => (minds.find((mind) => mind.id === id)?.folderId ?? null) !== folderId,
    ) || dragging.documentIds.some((id) => (folderOf.get(id) ?? null) !== folderId)
  );
}

/** The target the pointer drags over, by key: one at a time. */
const useDropTarget = create<{ over: string | null }>()(() => ({ over: null }));

let holding: { key: string; timer: number } | null = null;

const stopHolding = () => {
  if (holding) window.clearTimeout(holding.timer);
  holding = null;
};

const clear = () => {
  stopHolding();
  useDropTarget.setState({ over: null });
};

const endDrag = () => {
  clear();
  dragging = null;
  useMoveDrag.setState({ outOfFolder: false });
};

// A drag that ends anywhere (dropped elsewhere, or Esc) leaves no target lit and nothing held.
window.addEventListener("dragend", endDrag);
window.addEventListener("drop", endDrag);

/** The key a Folder's target goes by; Not in a Folder's for null. */
const keyOf = (folderId: string | null) => folderId ?? "not-in-a-folder";

/**
 * Makes an element a place to drop Minds and Documents into a Folder (null:
 * out of every Folder). `active` while something it would take is dragged
 * over it. `open`, given while the Folder is folded, opens it once a drag
 * has rested on it a moment, or once something is dropped into it.
 */
export function useFolderDrop(folderId: string | null, open?: () => void) {
  const key = keyOf(folderId);
  const active = useDropTarget((state) => state.over === key);
  const accepts = (event: DragEvent) =>
    carriesMove(event) ? wouldMove(folderId) : carriesLinks(event);
  const leave = () => {
    if (useDropTarget.getState().over === key) clear();
  };
  return {
    active,
    handlers: {
      onDragOver(event: DragEvent) {
        if (!accepts(event)) {
          leave();
          return;
        }
        event.preventDefault();
        event.dataTransfer.dropEffect = carriesMove(event) ? "move" : "copy";
        if (useDropTarget.getState().over !== key) {
          stopHolding();
          useDropTarget.setState({ over: key });
        }
        if (open && holding?.key !== key) {
          holding = { key, timer: window.setTimeout(open, OPENS_AFTER_MS) };
        }
      },
      onDragLeave(event: DragEvent) {
        // Still over it: only onto one of its own children.
        if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
        leave();
      },
      onDrop(event: DragEvent) {
        if (!accepts(event)) return;
        event.preventDefault();
        clear();
        if (!carriesMove(event)) {
          void useWebPages.getState().add(linksIn(event.dataTransfer), folderId);
          return;
        }
        const items = droppedItems(event);
        if (!items) return;
        void useAppStore.getState().moveToFolder(items, folderId);
        open?.();
        // A selection of one is the row shown, and stays; several chosen to drag together are done with.
        if (useSelection.getState().items.length > 1) useSelection.getState().clear("sidebar");
      },
    },
  };
}
