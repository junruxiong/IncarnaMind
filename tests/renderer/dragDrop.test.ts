import type { DragEvent } from "react";
import { beforeAll, beforeEach, describe, expect, test } from "vitest";
import type {
  Dragged,
  Dropped,
  DropTarget,
  OBJECTS_TYPE as ObjectsType,
} from "../../src/renderer/src/dragDrop";
import { moveLabel } from "../../src/renderer/src/moves";
import type { ObjectRef } from "../../src/renderer/src/objects";
import { type Undoable, useUndo } from "../../src/renderer/src/undo";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

/*
 * The drag-and-drop layer (docs/designs/interaction-system.md, C.5): a target
 * lights only while it would take what is dragged, one at a time, clears when
 * the pointer leaves it, and its drop's change goes on the undo stack, with or
 * without a drag.
 */

type Layer = typeof import("../../src/renderer/src/dragDrop");
let layer: Layer;
let OBJECTS_TYPE: typeof ObjectsType;

/** Just enough of an element for the drag's picture, which a test never sees. */
function fakeElement(): Record<string, unknown> {
  const element: Record<string, unknown> = {
    className: "",
    dataset: {},
    style: {},
    textContent: "",
    classList: { remove() {} },
    append() {},
    remove() {},
    querySelector: () => null,
    setAttribute() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
  };
  return element;
}

beforeAll(async () => {
  Object.assign(globalThis, {
    document: { createElement: fakeElement, body: { append() {} } },
    SVGElement: class {},
  });
  layer = await import("../../src/renderer/src/dragDrop");
  OBJECTS_TYPE = layer.OBJECTS_TYPE;
});

beforeEach(() => {
  useUndo.setState({ done: [], undone: [], line: null });
  layer.useDragState.setState({ over: null, at: null, source: null });
});

/** A drag event as the browser gives one: what it carries, and where its pointer is. */
function dragEvent(
  types: string[],
  {
    data = {},
    inside = false,
    taken = false,
  }: { data?: Record<string, string>; inside?: boolean; taken?: boolean } = {},
) {
  const store = { ...data };
  const event = {
    clientX: 40,
    clientY: 60,
    defaultPrevented: taken,
    relatedTarget: {},
    currentTarget: { contains: () => inside, ...fakeElement() },
    dataTransfer: {
      types,
      dropEffect: "none",
      effectAllowed: "uninitialized",
      files: [],
      items: [],
      getData: (type: string) => store[type] ?? "",
      setData: (type: string, value: string) => {
        store[type] = value;
        if (!types.includes(type)) types.push(type);
      },
      setDragImage() {},
    },
    preventDefault() {
      event.defaultPrevented = true;
    },
  };
  return event as typeof event & DragEvent<HTMLElement>;
}

const doc = (id: string): ObjectRef => ({ kind: "document", id });
const mind = (id: string): ObjectRef => ({ kind: "mind", id });

/** Starts dragging these from a row, as its `useDragSource` does, and returns what the drag carries. */
function startDragging(refs: ObjectRef[]) {
  const start = dragEvent([]);
  // A plain function behind a hook's name: it holds no state, so a test may call it here.
  const { useDragSource: sourceOf } = layer;
  sourceOf(() => ({ refs, name: "Budget", from: "sidebar" })).onDragStart(start);
  return {
    types: start.dataTransfer.types,
    data: { [OBJECTS_TYPE]: start.dataTransfer.getData(OBJECTS_TYPE) },
  };
}

function change(said: string): Undoable {
  return { said: () => said, undo: async () => {} };
}

/** A Tag-like target: Documents only, its drop one change to undo. */
function documentsTarget(dropped: Dropped[] = []): DropTarget {
  return {
    accepts: (dragged: Dragged) =>
      dragged.kind === "objects" && dragged.refs.every((ref) => ref.kind === "document")
        ? { kind: "add", said: () => "Add the Tag Contract" }
        : null,
    drop(what) {
      dropped.push(what);
      return Promise.resolve(change("Tagged 2 Documents Contract"));
    },
  };
}

const lit = () => layer.useDragState.getState().over?.key ?? null;

describe("a drop target", () => {
  test("lights while what is dragged is what it takes, from the move that enters it, and says what its drop would do", () => {
    const { types, data } = startDragging([doc("a"), doc("b")]);
    const target = layer.dropProps("tag", documentsTarget());
    target.onDragEnter(dragEvent(types, { data }));
    expect(lit()).toBe("tag");
    const over = dragEvent(types, { data });
    target.onDragOver(over);
    expect(lit()).toBe("tag");
    expect(over.defaultPrevented).toBe(true);
    expect(over.dataTransfer.dropEffect).toBe("copy");
    expect(layer.useDragState.getState().over?.effect.said?.((key) => key)).toBe(
      "Add the Tag Contract",
    );
    expect(layer.useDragState.getState().at).toEqual({ x: 40, y: 60 });
  });

  test("never lights for what it would refuse: a Mind among the Documents, files, a link", () => {
    const { types, data } = startDragging([doc("a"), mind("m")]);
    const target = layer.dropProps("tag", documentsTarget());
    const over = dragEvent(types, { data });
    target.onDragOver(over);
    expect(lit()).toBeNull();
    expect(over.defaultPrevented).toBe(false);
    for (const kinds of [["Files"], ["text/uri-list", "text/plain"]]) {
      target.onDragOver(dragEvent(kinds));
      expect(lit()).toBeNull();
    }
  });

  test("one lights at a time, a target inside another first, and leaving clears it", () => {
    const { types, data } = startDragging([doc("a")]);
    const inner = layer.dropProps("tag", documentsTarget());
    const outer = layer.dropProps("folder", documentsTarget());
    const over = dragEvent(types, { data });
    inner.onDragOver(over);
    outer.onDragOver(over);
    expect(lit()).toBe("tag");
    // Onto one of its own children: still over it.
    inner.onDragLeave(dragEvent(types, { data, inside: true }));
    expect(lit()).toBe("tag");
    inner.onDragLeave(dragEvent(types, { data }));
    expect(lit()).toBeNull();
    outer.onDragOver(dragEvent(types, { data }));
    expect(lit()).toBe("folder");
  });

  test("takes the drop and puts its change on the undo stack, with its line", async () => {
    const { types, data } = startDragging([doc("a"), doc("b")]);
    const dropped: Dropped[] = [];
    const target = layer.dropProps("tag", documentsTarget(dropped));
    target.onDragOver(dragEvent(types, { data }));
    const drop = dragEvent(types, { data });
    target.onDrop(drop);
    expect(drop.defaultPrevented).toBe(true);
    expect(lit()).toBeNull();
    expect(dropped).toEqual([{ kind: "objects", refs: [doc("a"), doc("b")], from: "sidebar" }]);
    await expect.poll(() => useUndo.getState().done.length).toBe(1);
    expect(useUndo.getState().line?.change?.said(String, String)).toBe(
      "Tagged 2 Documents Contract",
    );
  });

  test("leaves a drop a target inside took, and one it would refuse", () => {
    const { types, data } = startDragging([doc("a")]);
    const dropped: Dropped[] = [];
    const target = layer.dropProps("tag", documentsTarget(dropped));
    target.onDrop(dragEvent(types, { data, taken: true }));
    target.onDrop(dragEvent(["Files"]));
    expect(dropped).toEqual([]);
  });

  test("Move to… runs the same drop without a drag, and nothing for what it would refuse", async () => {
    const dropped: Dropped[] = [];
    await layer.runDrop(documentsTarget(dropped), [mind("m")]);
    expect(dropped).toEqual([]);
    await layer.runDrop(documentsTarget(dropped), [doc("a")]);
    expect(dropped).toEqual([{ kind: "objects", refs: [doc("a")] }]);
    expect(useUndo.getState().done).toHaveLength(1);
  });

  test("a folded target opens once the drag has rested on it", async () => {
    const { types, data } = startDragging([doc("a")]);
    let opened = 0;
    layer
      .dropProps("tag", documentsTarget(), () => {
        opened++;
      })
      .onDragOver(dragEvent(types, { data }));
    expect(opened).toBe(0);
    await expect.poll(() => opened, { timeout: 2000 }).toBe(1);
  });
});

describe("what a drop on a Folder says", () => {
  const say = (language: "en" | "zh-CN") => (key: MessageKey, params?: MessageParams) =>
    translate(language, key, params);

  test("one by where it goes, several by how many of what, in English and Chinese", () => {
    expect(moveLabel(say("en"), { mindIds: ["m"] }, "Finance")).toBe("Move to Finance");
    expect(moveLabel(say("en"), { documentIds: ["a", "b", "c"] }, "Finance")).toBe(
      "Move 3 Documents to Finance",
    );
    expect(moveLabel(say("en"), { mindIds: ["m"], documentIds: ["a"] }, "Finance")).toBe(
      "Move 2 items to Finance",
    );
    expect(moveLabel(say("zh-CN"), { documentIds: ["a", "b"] }, "财务")).toBe(
      "将 2 个文档移动到“财务”",
    );
  });
});
