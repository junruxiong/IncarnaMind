import type { DragEvent } from "react";
import { create } from "zustand";
import { startMoveDrag } from "./folderDrag";
import { translateNow } from "./i18n";
import { type MoveItems, movingName } from "./moves";
import type { ObjectRef } from "./objects";
import { selectedOn } from "./selection";
import { useAppStore } from "./store";

/**
 * What a drag or "Move to…" from a sidebar row takes: every Mind and Document
 * selected, when the row is one of them, else the row alone.
 */
export function movedFrom(row: ObjectRef): MoveItems {
  const chosen = selectedOn("sidebar");
  const held = chosen.some((each) => each.kind === row.kind && each.id === row.id);
  const items = held ? chosen : [row];
  const idsOf = (kind: ObjectRef["kind"]) =>
    items.flatMap((each) => (each.kind === kind ? [each.id] : []));
  return { mindIds: idsOf("mind"), documentIds: idsOf("document") };
}

/** What is moved, by name or by how many of what. */
export function nameOfMoved(items: MoveItems): string {
  const { minds, documents } = useAppStore.getState();
  return movingName(translateNow, items, { minds, documents });
}

/** Starts dragging a row: it, or the selection it is part of. */
export function dragFromRow(event: DragEvent<HTMLElement>, row: ObjectRef): void {
  const items = movedFrom(row);
  startMoveDrag(event, items, nameOfMoved(items));
}

interface MoveRequest {
  /** What the picker opens beside, and gives the focus back to: the row, or its ⋯. */
  anchor: HTMLElement;
  items: MoveItems;
}

/** The sidebar's one "Move to…", open for these, if it is. */
export const useSidebarMove = create<{
  request: MoveRequest | null;
  show(anchor: HTMLElement | null, row: ObjectRef): void;
  hide(): void;
}>()((set) => ({
  request: null,
  show: (anchor, row) => {
    if (anchor) set({ request: { anchor, items: movedFrom(row) } });
  },
  hide: () => set({ request: null }),
}));
