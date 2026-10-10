import { afterAll, beforeAll, beforeEach, describe, expect, test, vi } from "vitest";
import type {
  CoreEventName,
  CoreEvents,
  Document,
  DocumentGroupAssignment,
  LibrarySnapshot,
} from "../../src/core/api";

/*
 * The store follows the core's events (#156): per-Document ones are gathered
 * and applied together, Library assignments in place, without reading the
 * whole Library again. The store reads the bridge the preload exposes, so a
 * fake one stands in for it here.
 */

const listeners = new Map<string, ((payload: unknown) => void)[]>();
const methods = new Map<string, ReturnType<typeof vi.fn>>();
const bridge = new Proxy(
  {
    on(event: string, listener: (payload: unknown) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
      return () => undefined;
    },
  },
  {
    get(target, name: string) {
      if (name === "on") return target.on;
      let method = methods.get(name);
      if (!method) {
        method = vi.fn(async () => undefined);
        methods.set(name, method);
      }
      return method;
    },
  },
);

function emit<E extends CoreEventName>(event: E, payload: CoreEvents[E]) {
  for (const listener of listeners.get(event) ?? []) listener(payload);
}

const method = (name: string) => {
  // Reading it through the bridge makes it if it isn't there yet.
  (bridge as unknown as Record<string, unknown>)[name];
  return methods.get(name) as ReturnType<typeof vi.fn>;
};

const doc = (id: string, version = 0) =>
  ({ id, name: `${id} v${version}`, tags: [], linkedFolderId: null }) as unknown as Document;

const assignment = (
  documentId: string,
  status: DocumentGroupAssignment["status"],
  groupId: string | null = null,
): DocumentGroupAssignment => ({
  documentId,
  groupId,
  source: "automatic",
  status,
  error: null,
  model: null,
});

const snapshot = (assignments: DocumentGroupAssignment[]): LibrarySnapshot => ({
  groups: [{ id: "g", name: "Research", description: "", createdAt: "", updatedAt: "" }],
  deletedGroups: [],
  assignments,
  settings: { classifier: null, automatic: false },
});

let useAppStore: typeof import("../../src/renderer/src/store").useAppStore;

beforeAll(async () => {
  // One clock for the file: the store remembers when it last changed, from one test to the next.
  vi.useFakeTimers();
  (globalThis as unknown as { window: unknown }).window = {
    incarnamind: bridge,
    incarnamindFiles: {},
  };
  ({ useAppStore } = await import("../../src/renderer/src/store"));
});

afterAll(() => {
  vi.useRealTimers();
});

/** Lets the promises already settled run on (a reply from the bridge, and what awaits it). */
const settle = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};

beforeEach(() => {
  // A quiet spell, so each test starts with nothing waiting.
  vi.advanceTimersByTime(1000);
  for (const each of methods.values()) each.mockReset();
  useAppStore.setState({
    documents: [doc("a"), doc("b"), doc("c")],
    library: snapshot([assignment("a", "pending"), assignment("b", "pending")]),
  });
});

describe("the store follows the core's per-Document events", () => {
  test("a burst changes the store once, each Document to its latest copy, the rest untouched", () => {
    const before = useAppStore.getState().documents;
    const updates = vi.fn();
    const stop = useAppStore.subscribe(updates);
    emit("document.status", doc("a", 1));
    emit("documents.tagged", [doc("a", 2), doc("b", 1)]);
    emit("document.status", doc("x", 1));
    emit("documents.moved", [doc("b", 2)]);
    // Nothing yet: the events wait for the next update.
    expect(useAppStore.getState().documents).toBe(before);
    vi.advanceTimersByTime(100);
    stop();
    expect(updates).toHaveBeenCalledTimes(1);
    const after = useAppStore.getState().documents;
    expect(after.map((each) => each.name)).toEqual(["x v1", "a v2", "b v2", "c v0"]);
    expect(after[3]).toBe(before[2]);
  });

  test("events coming fast are gathered: at most one store update per 100ms", () => {
    const updates = vi.fn();
    const stop = useAppStore.subscribe(updates);
    for (let i = 1; i <= 300; i++) {
      emit("documents.tagged", [doc(i % 2 ? "a" : "b", i)]);
      emit("library.assignments", [assignment(i % 2 ? "a" : "b", "classifying")]);
      vi.advanceTimersByTime(2);
    }
    vi.advanceTimersByTime(100);
    stop();
    // 600 events over 600ms: about one update each 100ms, not one each.
    expect(updates.mock.calls.length).toBeGreaterThanOrEqual(6);
    expect(updates.mock.calls.length).toBeLessThanOrEqual(9);
    const { documents } = useAppStore.getState();
    expect(documents.find((each) => each.id === "a")?.name).toBe("a v299");
    expect(documents.find((each) => each.id === "b")?.name).toBe("b v300");
  });

  test("removed Documents go, even with a change to them still waiting", () => {
    emit("documents.tagged", [doc("a", 1)]);
    emit("documents.removed", ["a", "c"]);
    vi.advanceTimersByTime(100);
    expect(useAppStore.getState().documents.map((each) => each.id)).toEqual(["b"]);
  });

  test("a change the User makes isn't undone by an older event still waiting", async () => {
    emit("document.status", doc("a", 1));
    method("renameDocument").mockResolvedValue({ ...doc("a", 2), name: "Renamed" });
    await useAppStore.getState().renameDocument("a", "Renamed");
    vi.advanceTimersByTime(1000);
    expect(useAppStore.getState().documents.find((each) => each.id === "a")?.name).toBe("Renamed");
  });
});

describe("the store follows the Library's assignments", () => {
  test("they change in place, without reading the whole Library again", () => {
    const before = useAppStore.getState().library;
    emit("library.assignments", [assignment("b", "classified", "g"), assignment("c", "pending")]);
    vi.advanceTimersByTime(100);
    const library = useAppStore.getState().library;
    expect(method("getLibrary")).not.toHaveBeenCalled();
    expect(
      library?.assignments.map((each) => [each.documentId, each.status, each.groupId]),
    ).toEqual([
      ["a", "pending", null],
      ["b", "classified", "g"],
      ["c", "pending", null],
    ]);
    expect(library?.assignments[0]).toBe(before?.assignments[0]);
    expect(library?.groups).toBe(before?.groups);
  });

  test("a change to the Folders reads the Library again, and older assignments don't land on it", async () => {
    // Waiting, not yet applied: the snapshot read after it has it already, and newer.
    emit("library.assignments", [assignment("a", "classifying")]);
    method("getLibrary").mockResolvedValue(
      snapshot([assignment("a", "classified", "g"), assignment("b", "pending")]),
    );
    emit("library.changed", null);
    expect(method("getLibrary")).toHaveBeenCalledTimes(1);
    await settle();
    expect(useAppStore.getState().library?.assignments[0]?.status).toBe("classified");
    vi.advanceTimersByTime(1000);
    expect(useAppStore.getState().library?.assignments[0]).toMatchObject({
      status: "classified",
      groupId: "g",
    });
    // Assignments after the snapshot change it as usual.
    emit("library.assignments", [assignment("b", "failed")]);
    vi.advanceTimersByTime(1000);
    expect(useAppStore.getState().library?.assignments[1]?.status).toBe("failed");
  });

  test("before the Library is loaded, its assignments wait for the snapshot loading reads", () => {
    useAppStore.setState({ library: null });
    emit("library.assignments", [assignment("a", "classified")]);
    emit("documents.tagged", [doc("a", 1)]);
    vi.advanceTimersByTime(100);
    expect(useAppStore.getState().library).toBeNull();
    expect(useAppStore.getState().documents[0]?.name).toBe("a v1");
  });
});
