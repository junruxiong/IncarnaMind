import type { DragEvent } from "react";
import { create } from "zustand";
import { startMoveDrag } from "./folderDrag";
import { translateNow } from "./i18n";
import { type MoveItems, movingName } from "./moves";
import type { ObjectRef } from "./objects";
import { selectedOn } from "./selection";
import { useAppStore } from "./store";

function movedFrom(row: ObjectRef): MoveItems {
  const chosen = selectedOn("sidebar");
  const held = chosen.some((each) => each.kind === row.kind && each.id === row.id);
  const items = held ? chosen : [row];
  const idsOf = (kind: ObjectRef["kind"]) =>
    items.flatMap((each) => (each.kind === kind ? [each.id] : []));
  return { mindIds: idsOf("mind"), documentIds: idsOf("document") };
}

export function nameOfMoved(items: MoveItems): string {
  const { minds, documents } = useAppStore.getState();
  return movingName(translateNow, items, { minds, documents });
}

export function dragFromRow(event: DragEvent<HTMLElement>, row: ObjectRef): void {
  const items = movedFrom(row);
  startMoveDrag(event, items, nameOfMoved(items));
}

interface MoveRequest {
  /** What the picker opens beside, and gives the focus back to: the row, or its ⋯. */
  anchor: HTMLElement;
  items: MoveItems;
}

/** Opens Move to… from a row's menu, beside its ⋯, for the row or the selection it is in. */
export function moveFromMenu(row: HTMLElement | null, object: ObjectRef): void {
  // Once the menu has closed and given the focus back, so the picker opens beside the button.
  requestAnimationFrame(() =>
    useSidebarMove
      .getState()
      .show(row?.querySelector<HTMLElement>("[data-object-menu]") ?? null, object),
  );
}

// One picker for the whole sidebar: each row opening its own would leave several open at once.
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
