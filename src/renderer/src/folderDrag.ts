import type { DragEvent } from "react";
import { create } from "zustand";
import { itemsOf, type MoveItems } from "./moves";
import { useAppStore } from "./store";

/**
 * Dragging Minds (and, with #212, Documents) onto a Folder in the sidebar, or
 * onto "Not in a Folder" to take them out of theirs. What is dragged travels
 * as its ids under a type of IncarnaMind's own, so the window's file drop
 * (`useFileDrop`) and the note ignore it. The target under the pointer
 * highlights; dropping moves, as "Move to…" does (`moveToFolder`).
 */
export const MOVE_TYPE = "application/x-incarnamind-move";

/** Starts dragging these Minds and Documents. */
export function startMoveDrag(event: DragEvent, items: MoveItems): void {
  event.dataTransfer.setData(MOVE_TYPE, JSON.stringify(itemsOf(items)));
  event.dataTransfer.effectAllowed = "move";
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

/** The target the pointer drags over, by key: one at a time. */
const useDropTarget = create<{ over: string | null }>()(() => ({ over: null }));

// A drag that ends anywhere (dropped elsewhere, or Esc) leaves no target lit.
const clear = () => useDropTarget.setState({ over: null });
window.addEventListener("dragend", clear);
window.addEventListener("drop", clear);

/** The key a Folder's target goes by; Not in a Folder's for null. */
const keyOf = (folderId: string | null) => folderId ?? "not-in-a-folder";

/**
 * Makes an element a place to drop Minds and Documents into a Folder (null:
 * out of every Folder). `active` while something is dragged over it.
 */
export function useFolderDrop(folderId: string | null) {
  const key = keyOf(folderId);
  const active = useDropTarget((state) => state.over === key);
  return {
    active,
    handlers: {
      onDragOver(event: DragEvent) {
        if (!carriesMove(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        if (useDropTarget.getState().over !== key) useDropTarget.setState({ over: key });
      },
      onDragLeave(event: DragEvent) {
        if (!carriesMove(event)) return;
        // Still over it: only onto one of its own children.
        if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
        if (useDropTarget.getState().over === key) clear();
      },
      onDrop(event: DragEvent) {
        if (!carriesMove(event)) return;
        event.preventDefault();
        clear();
        const items = droppedItems(event);
        if (items) void useAppStore.getState().moveToFolder(items, folderId);
      },
    },
  };
}
