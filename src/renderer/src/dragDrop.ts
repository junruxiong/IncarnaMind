import { type DragEvent, useState } from "react";
import { create } from "zustand";
import type { Words } from "./lineNames";
import { exitToward } from "./motion";
import { useNotices } from "./notices";
import { OBJECT_KINDS, type ObjectRef } from "./objects";
import type { SelectSurface } from "./selectionModel";
import { type Undoable, useUndo } from "./undo";

/**
 * The one drag-and-drop layer (docs/designs/interaction-system.md, C.5), on
 * the browser's own drag and drop: what this window drags travels as object
 * references under one type, files and links come from other apps, and each
 * place that takes a drop is a target that lights only while it would take
 * what is dragged and says what the drop would do. A drop's change goes on
 * the undo stack with its line; Move to… runs the same `drop` without a drag
 * (`runDrop`).
 */
export const OBJECTS_TYPE = "application/x-incarnamind-objects";

const OPENS_AFTER_MS = 700;

export interface Point {
  x: number;
  y: number;
}

/** What is dragged, as far as a target can know before the drop. */
export type Dragged =
  | { kind: "objects"; refs: readonly ObjectRef[] }
  | { kind: "files" }
  | { kind: "links" };

/** What a drop hands its target. `from`: the list whose selection was dragged. */
export type Dropped =
  | { kind: "objects"; refs: readonly ObjectRef[]; from?: SelectSurface | undefined }
  | { kind: "files"; files: File[]; folders: File[] }
  | { kind: "links"; data: DataTransfer };

export interface DropEffect<Place = unknown> {
  kind: "move" | "add" | "attach" | "import";
  /** What the drop would do, beside the pointer: "Move to Finance". */
  said?: Words;
  /** Where in the target it would land, for a target that marks it. */
  place?: Place;
}

type DropResult = Undoable | null | undefined;

export interface DropTarget<Place = unknown> {
  accepts(dragged: Dragged, at: Point): DropEffect<Place> | null;
  drop(dropped: Dropped, effect: DropEffect<Place>): DropResult | Promise<DropResult>;
}

/** A drag from this window: what it carries, and the name its picture gives it. */
export interface ObjectDrag {
  refs: readonly ObjectRef[];
  name: string;
  from?: SelectSurface;
}

interface DragState {
  over: { key: string; effect: DropEffect } | null;
  /** The pointer, while over a target that says what its drop would do. */
  at: Point | null;
  /** What this window drags, set once the drag is under way. */
  source: ObjectDrag | null;
}

export const useDragState = create<DragState>()(() => ({ over: null, at: null, source: null }));

// A drag's data can only be read on the drop, so what is dragged is kept here for the targets it passes over.
let dragging: ObjectDrag | null = null;
let holding: { key: string; timer: ReturnType<typeof setTimeout> } | null = null;

const stopHolding = () => {
  if (holding) clearTimeout(holding.timer);
  holding = null;
};

function clear(): void {
  stopHolding();
  useDragState.setState({ over: null, at: null });
}

function endDrag(): void {
  clear();
  dragging = null;
  useDragState.setState({ source: null });
}

let listening = false;
// A drag that ends anywhere (dropped elsewhere, or Esc) leaves no target lit and nothing held.
function listenForEnd(): void {
  if (listening || typeof window === "undefined") return;
  listening = true;
  window.addEventListener("dragend", endDrag);
  window.addEventListener("drop", endDrag);
}

const afterFrame = (run: () => void) =>
  typeof requestAnimationFrame === "undefined" ? setTimeout(run) : requestAnimationFrame(run);

function setPicture(event: DragEvent<HTMLElement>, name: string, count: number): void {
  const picture = pictureOf(event.currentTarget, name, count);
  document.body.append(picture);
  event.dataTransfer.setDragImage(picture, 12, 14);
  // The browser copies the picture as the drag starts; it isn't needed after.
  afterFrame(() => picture.remove());
}

function pictureOf(source: Element, name: string, count: number): HTMLElement {
  const picture = document.createElement("div");
  picture.className =
    "fixed top-0 -left-[2000px] flex h-7 max-w-64 items-center gap-2 rounded-md bg-sheet px-2.5 text-[13px] text-ink shadow-[0_0_0_1px_var(--color-rule),0_12px_32px_rgba(23,25,28,0.12)]";
  const icon = source.querySelector("svg")?.cloneNode(true);
  if (icon instanceof SVGElement) {
    icon.setAttribute("class", "size-4 shrink-0 text-ink-meta");
    picture.append(icon);
  }
  const label = document.createElement("span");
  label.className = "min-w-0 truncate";
  label.textContent = name;
  picture.append(label);
  if (count > 1) {
    const badge = document.createElement("span");
    badge.dataset.testid = "drag-count";
    badge.className =
      "inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-accent px-1.5 text-[11px] font-semibold text-white";
    badge.textContent = String(count);
    picture.append(badge);
  }
  return picture;
}

/**
 * Makes an element a source of a drag. `drag` says what it carries when the
 * drag starts (the row, or every row selected with it); null keeps it still.
 */
export function useDragSource(drag: () => ObjectDrag | null) {
  return {
    draggable: true,
    onDragStart(event: DragEvent<HTMLElement>) {
      const what = drag();
      if (!what || what.refs.length === 0) {
        event.preventDefault();
        return;
      }
      listenForEnd();
      dragging = what;
      event.dataTransfer.setData(OBJECTS_TYPE, JSON.stringify(what.refs));
      event.dataTransfer.effectAllowed = what.refs.some((ref) => ref.kind === "tab")
        ? "move"
        : "all";
      setPicture(event, what.name, what.refs.length);
      // Not while the drag starts: a page that changes then makes the browser give the drag up.
      afterFrame(() => {
        if (dragging === what) useDragState.setState({ source: what });
      });
    },
  } as const;
}

type DragData = Pick<DataTransfer, "types">;

/** What a drag over a target carries, or null for nothing the layer knows. */
export function draggedIn(data: DragData): Dragged | null {
  const { types } = data;
  if (types.includes(OBJECTS_TYPE))
    return dragging ? { kind: "objects", refs: dragging.refs } : null;
  if (types.includes("Files")) return { kind: "files" };
  if (types.includes("text/uri-list")) return { kind: "links" };
  return null;
}

const KINDS: readonly string[] = OBJECT_KINDS;

function refsIn(text: string): ObjectRef[] {
  try {
    const value = JSON.parse(text) as unknown;
    if (!Array.isArray(value)) return [];
    return value.flatMap((each: unknown) => {
      if (typeof each !== "object" || each === null) return [];
      const { kind, id } = each as Record<string, unknown>;
      return typeof kind === "string" && KINDS.includes(kind) && typeof id === "string"
        ? [{ kind: kind as ObjectRef["kind"], id }]
        : [];
    });
  } catch {
    return [];
  }
}

function droppedIn(data: DataTransfer): Dropped | null {
  const { types } = data;
  if (types.includes(OBJECTS_TYPE)) {
    const refs = refsIn(data.getData(OBJECTS_TYPE));
    return refs.length > 0 ? { kind: "objects", refs, from: dragging?.from } : null;
  }
  const links = types.includes("text/uri-list");
  if (types.includes("Files") && !(links && data.files.length === 0)) {
    // Read now: the drop's items are only there while it is handled.
    const items = Array.from(data.items).filter((item) => item.kind === "file");
    const folders = items.flatMap((item) => {
      const file = item.webkitGetAsEntry()?.isDirectory ? item.getAsFile() : null;
      return file ? [file] : [];
    });
    return { kind: "files", files: Array.from(data.files), folders };
  }
  return links ? { kind: "links", data } : null;
}

const DROP_EFFECTS = { move: "move", add: "copy", attach: "link", import: "copy" } as const;

const pointOf = (event: { clientX: number; clientY: number }): Point => ({
  x: event.clientX,
  y: event.clientY,
});

/** Puts a drop's change on the undo stack, with its line; a failure is said where it happened. */
function finish(result: DropResult | Promise<DropResult>): Promise<void> {
  return Promise.resolve(result).then(
    (change) => {
      if (change) useUndo.getState().push(change);
    },
    (error: unknown) => useNotices.getState().showError(error),
  );
}

/** Runs a target's drop without a drag, as Move to… does: nothing if it would refuse them. */
export function runDrop(target: DropTarget, refs: readonly ObjectRef[]): Promise<void> {
  const effect = target.accepts({ kind: "objects", refs }, { x: 0, y: 0 });
  if (!effect) return Promise.resolve();
  return finish(target.drop({ kind: "objects", refs }, effect));
}

const samePlace = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

function show(key: string, effect: DropEffect, point: Point): void {
  const state = useDragState.getState();
  const changed = state.over?.key !== key || !samePlace(state.over.effect.place, effect.place);
  const at = effect.said ? point : null;
  if (!changed && at === null && state.at === null) return;
  if (changed) stopHolding();
  useDragState.setState({ ...(changed && { over: { key, effect } }), at });
}

/** The drop's picture shrinks into the row that took it (DESIGN.md, Motion). */
function settleInto(target: Element, at: Point): void {
  if (!dragging) return;
  const picture = pictureOf(target, dragging.name, dragging.refs.length);
  picture.classList.remove("top-0", "-left-[2000px]");
  picture.style.left = `${at.x - 12}px`;
  picture.style.top = `${at.y - 14}px`;
  document.body.append(picture);
  exitToward(picture, target);
  picture.remove();
}

export interface DropProps {
  onDragOver(event: DragEvent): void;
  onDragLeave(event: DragEvent): void;
  onDrop(event: DragEvent): void;
}

/**
 * The handlers that make an element the target `key`, one lit at a time. A
 * target inside another takes the drag first. `open`, given while a target is
 * folded, opens it once a drag has rested on it a moment, or once something
 * is dropped into it.
 */
export function dropProps<Place>(
  key: string,
  target: DropTarget<Place>,
  open?: () => void,
): DropProps {
  const effectOf = (event: DragEvent) => {
    const dragged = event.defaultPrevented ? null : draggedIn(event.dataTransfer);
    return dragged ? target.accepts(dragged, pointOf(event)) : null;
  };
  const leave = () => {
    if (useDragState.getState().over?.key === key) clear();
  };
  return {
    onDragOver(event) {
      listenForEnd();
      const effect = effectOf(event);
      if (!effect) {
        leave();
        return;
      }
      event.preventDefault();
      event.dataTransfer.dropEffect = DROP_EFFECTS[effect.kind];
      show(key, effect, pointOf(event));
      if (open && holding?.key !== key) {
        holding = { key, timer: setTimeout(open, OPENS_AFTER_MS) };
      }
    },
    onDragLeave(event) {
      // Still over it: only onto one of its own children.
      if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
      leave();
    },
    onDrop(event) {
      // Where it is let go, which the last move over it may not have said.
      const effect = effectOf(event);
      if (!effect) return;
      event.preventDefault();
      clear();
      const dropped = droppedIn(event.dataTransfer);
      if (!dropped) return;
      // A target that says what its drop does is a row: the picture shrinks into it.
      if (dropped.kind === "objects" && effect.said)
        settleInto(event.currentTarget, pointOf(event));
      void finish(target.drop(dropped, effect));
      open?.();
    },
  };
}

/** Makes an element a drop target: `over` is what its drop would do while something it takes is over it. */
export function useDropTarget<Place = undefined>(
  key: string,
  target: DropTarget<Place>,
  open?: () => void,
): { over: DropEffect<Place> | null; props: DropProps } {
  const over = useDragState((state) =>
    state.over?.key === key ? (state.over.effect as DropEffect<Place>) : null,
  );
  return { over, props: dropProps(key, target, open) };
}

/**
 * Makes a whole area, such as the window, a target: lit while what it takes
 * is dragged anywhere over it, and taking only what no target inside took.
 * Counting enters and leaves keeps it steady as the pointer crosses the
 * elements inside.
 */
export function useDropArea(target: DropTarget) {
  const [depth, setDepth] = useState(0);
  const [entered, setEntered] = useState<{ effect: DropEffect; dragged: Dragged } | null>(null);
  const effectOf = (event: DragEvent) => {
    const dragged = draggedIn(event.dataTransfer);
    return dragged ? target.accepts(dragged, pointOf(event)) : null;
  };
  return {
    over: depth > 0 ? (entered?.effect ?? null) : null,
    /** What was dragged in last: files or links, say. */
    dragged: entered?.dragged ?? null,
    props: {
      onDragEnter(event: DragEvent) {
        const dragged = draggedIn(event.dataTransfer);
        const effect = dragged ? target.accepts(dragged, pointOf(event)) : null;
        if (!dragged || !effect) return;
        event.preventDefault();
        setEntered({ effect, dragged });
        setDepth((current) => current + 1);
      },
      onDragLeave(event: DragEvent) {
        if (!effectOf(event)) return;
        setDepth((current) => Math.max(0, current - 1));
      },
      onDragOver(event: DragEvent) {
        const next = effectOf(event);
        if (!next) return;
        const taken = event.defaultPrevented;
        // Accepting the drop also stops Electron from opening a file in the window.
        event.preventDefault();
        if (!taken) event.dataTransfer.dropEffect = DROP_EFFECTS[next.kind];
      },
      onDrop(event: DragEvent) {
        const next = effectOf(event);
        if (!next) return;
        const taken = event.defaultPrevented;
        event.preventDefault();
        setDepth(0);
        if (taken) return;
        const dropped = droppedIn(event.dataTransfer);
        if (dropped) void finish(target.drop(dropped, next));
      },
    },
  };
}
