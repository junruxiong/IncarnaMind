import { type KeyboardEvent, type MouseEvent, useEffect, useRef } from "react";
import { create } from "zustand";
import type { ObjectRef } from "./objects";
import {
  choose,
  clear,
  extend,
  keepOnly,
  NOTHING_SELECTED,
  refKey,
  refOfKey,
  type SelectionState,
  type SelectSurface,
  selectAll,
  toggle,
} from "./selectionModel";
import { useAppStore } from "./store";

export type { SelectSurface } from "./selectionModel";

/*
 * One selection for the whole window, held by one surface at a time (the
 * sidebar or the Library): the same gestures in both (docs/agents/interaction.md,
 * rule 6). A row says what it is with `data-select-ref`, and a surface says
 * what is in view, in order, so ⌘A and Shift-click know the range.
 */

const orders = new Map<SelectSurface, () => readonly ObjectRef[]>();

const orderOf = (surface: SelectSurface): readonly ObjectRef[] => orders.get(surface)?.() ?? [];

/** Registers what a surface has in view, in order, for as long as it is mounted. */
export function useSelectionOrder(surface: SelectSurface, order: () => readonly ObjectRef[]): void {
  const latest = useRef(order);
  latest.current = order;
  useEffect(() => {
    const read = () => latest.current();
    orders.set(surface, read);
    return () => {
      if (orders.get(surface) === read) orders.delete(surface);
    };
  }, [surface]);
}

/** The selectable rows inside an element, in the page's order, each once. */
export function rowsIn(root: ParentNode | null): ObjectRef[] {
  const seen = new Set<string>();
  const refs: ObjectRef[] = [];
  for (const row of root?.querySelectorAll<HTMLElement>("[data-select-ref]") ?? []) {
    const key = row.dataset.selectRef ?? "";
    const ref = refOfKey(key);
    if (ref && !seen.has(key)) {
      seen.add(key);
      refs.push(ref);
    }
  }
  return refs;
}

interface Selection extends SelectionState {
  choose(surface: SelectSurface, ref: ObjectRef): void;
  toggle(surface: SelectSurface, ref: ObjectRef): void;
  /** From the anchor to this one, in the order the surface has in view. */
  extendTo(surface: SelectSurface, ref: ObjectRef): void;
  all(surface: SelectSurface): void;
  clear(surface?: SelectSurface): void;
  keepOnly(exists: (ref: ObjectRef) => boolean): void;
}

export const useSelection = create<Selection>()((set, get) => {
  const change = (next: SelectionState) => {
    const now = get();
    if (next.surface === now.surface && next.items === now.items && next.anchor === now.anchor)
      return;
    set({ surface: next.surface, items: next.items, keys: next.keys, anchor: next.anchor });
  };
  return {
    ...NOTHING_SELECTED,
    choose: (surface, ref) => change(choose(get(), surface, ref)),
    toggle: (surface, ref) => change(toggle(get(), surface, ref)),
    extendTo: (surface, ref) => change(extend(get(), surface, ref, orderOf(surface))),
    all: (surface) => change(selectAll(get(), surface, orderOf(surface))),
    clear: (surface) => change(clear(get(), surface)),
    keepOnly: (exists) => change(keepOnly(get(), exists)),
  };
});

/** What is selected on a surface now, in the order chosen. */
const NONE: readonly ObjectRef[] = [];
export const selectedOn = (surface: SelectSurface): readonly ObjectRef[] => {
  const state = useSelection.getState();
  return state.surface === surface ? state.items : NONE;
};

/** The selected items of a surface, as a hook: the same array until the selection changes. */
export const useSelectedOn = (surface: SelectSurface): readonly ObjectRef[] =>
  useSelection((state) => (state.surface === surface ? state.items : NONE));

/**
 * One row's part in the gestures. A click opens it and selects just it;
 * ⌘-click or Ctrl-click adds or removes it, and Shift-click extends from the
 * anchor, neither opening anything. `rowProps` go on the row's element.
 */
export function useSelectable(ref: ObjectRef, surface: SelectSurface, open: () => void) {
  const key = refKey(ref);
  const selected = useSelection((state) => state.surface === surface && state.keys.has(key));
  return {
    selected,
    rowProps: { "aria-selected": selected, "data-select-ref": key } as const,
    onClick(event: MouseEvent) {
      const { toggle, extendTo, choose } = useSelection.getState();
      if (event.metaKey || event.ctrlKey) {
        event.preventDefault();
        toggle(surface, ref);
      } else if (event.shiftKey) {
        event.preventDefault();
        extendTo(surface, ref);
      } else {
        choose(surface, ref);
        open();
      }
    },
  };
}

/**
 * Shift+↑ and Shift+↓ on a row's button: the focus moves to the row above or
 * below and the selection reaches it from the anchor. Plain arrows are
 * `moveThroughRows`. True if the key was used.
 */
export function extendThroughRows(
  event: KeyboardEvent<HTMLElement>,
  surface: SelectSurface,
): boolean {
  if (!event.shiftKey || event.altKey || event.metaKey || event.ctrlKey) return false;
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return false;
  const target = event.target as HTMLElement;
  if (!target.matches("[data-tree-row]")) return false;
  const rows = Array.from(event.currentTarget.querySelectorAll<HTMLElement>("[data-tree-row]"));
  const next = rows[rows.indexOf(target) + (event.key === "ArrowDown" ? 1 : -1)];
  event.preventDefault();
  if (!next) return true;
  const from = refOfKey(target.closest<HTMLElement>("[data-select-ref]")?.dataset.selectRef ?? "");
  const to = refOfKey(next.closest<HTMLElement>("[data-select-ref]")?.dataset.selectRef ?? "");
  next.focus();
  if (!to) return true;
  const { surface: held, anchor, choose, extendTo } = useSelection.getState();
  // Starting from a row that isn't selected yet: it is the anchor.
  if (from && (held !== surface || !anchor)) choose(surface, from);
  extendTo(surface, to);
  return true;
}

/** Keeps the selection to Documents and Minds that still exist, as they are deleted. */
export function usePruneSelection(): void {
  const documents = useAppStore((state) => state.documents);
  const minds = useAppStore((state) => state.minds);
  useEffect(() => {
    const documentIds = new Set(documents.map((each) => each.id));
    const mindIds = new Set(minds.map((each) => each.id));
    useSelection
      .getState()
      .keepOnly((ref) =>
        ref.kind === "document"
          ? documentIds.has(ref.id)
          : ref.kind === "mind"
            ? mindIds.has(ref.id)
            : true,
      );
  }, [documents, minds]);
}
