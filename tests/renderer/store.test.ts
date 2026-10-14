import { afterAll, beforeAll, beforeEach, describe, expect, test, vi } from "vitest";
import type {
  CoreEventName,
  CoreEvents,
  Document,
  DocumentGroupAssignment,
  Folder,
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
  suggested: false,
  status,
  error: null,
  model: null,
});

const snapshot = (assignments: DocumentGroupAssignment[]): LibrarySnapshot => ({
  groups: [
    {
      id: "g",
      name: "Research",
      description: "",
      countsOwnFiles: false,
      createdAt: "",
      updatedAt: "",
    },
  ],
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

describe("undoing a move", () => {
  test("moves each Mind back, and puts each Document's assignment back with who chose it", async () => {
    const { useUndo } = await import("../../src/renderer/src/undo");
    const filed = assignment("a", "classified", "g");
    useAppStore.setState({
      minds: [
        {
          id: "m",
          title: "Plan",
          autoTitle: "",
          kind: "mind",
          folderId: null,
          createdAt: "",
          updatedAt: "",
        },
      ],
      library: snapshot([filed]),
    });
    method("moveToFolder").mockResolvedValue({
      folderId: "h",
      minds: [{ id: "m", from: null }],
      documents: [{ id: "a", from: "g", previous: filed }],
      conversations: [],
    });

    await useAppStore.getState().moveToFolder({ mindIds: ["m"], documentIds: ["a"] }, "h");
    await useUndo.getState().undo();

    expect(method("moveToFolder")).toHaveBeenLastCalledWith({
      mindIds: ["m"],
      documentIds: [],
      conversationIds: [],
      folderId: null,
    });
    expect(method("restoreAssignment")).toHaveBeenCalledWith("a", filed);
  });
});

describe("changes the User can take back", () => {
  const mind = {
    id: "m",
    title: "Plan",
    autoTitle: "",
    kind: "mind" as const,
    folderId: null,
    createdAt: "",
    updatedAt: "",
  };
  let useUndo: typeof import("../../src/renderer/src/undo").useUndo;

  beforeAll(async () => {
    ({ useUndo } = await import("../../src/renderer/src/undo"));
  });

  beforeEach(() => {
    useUndo.setState({ done: [], undone: [], line: null });
    useAppStore.setState({ minds: [mind], settings: null });
  });

  const top = () => useUndo.getState().done.at(-1);

  test("a rename says so, is undone by the old name and done again by the new", async () => {
    method("renameMind").mockResolvedValue({ ...mind, title: "Budget" });
    await useAppStore.getState().renameMind("m", "Budget");
    expect(top()?.message).toBe("Renamed to “Budget”");
    expect(top()?.name).toBe("Rename");
    await useUndo.getState().undo();
    expect(method("renameMind")).toHaveBeenLastCalledWith("m", "Plan");
    await useUndo.getState().redo();
    expect(method("renameMind")).toHaveBeenLastCalledWith("m", "Budget");
  });

  test("a delete asks nothing, says it went to Recently deleted, is undone by restoring it and done again", async () => {
    await useAppStore.getState().deleteDocuments(["a"]);
    expect(top()?.message).toBe("Moved “a v0” to Recently deleted");
    expect(useAppStore.getState().documents.map((each) => each.id)).toEqual(["b", "c"]);
    await useUndo.getState().undo();
    expect(method("restore")).toHaveBeenCalledWith([{ kind: "document", id: "a" }]);
    await useUndo.getState().redo();
    expect(method("deleteDocument")).toHaveBeenCalledTimes(2);
  });

  test("several Documents deleted together are one change, said by their count", async () => {
    await useAppStore.getState().deleteDocuments(["a", "b"]);
    expect(top()?.message).toBe("Moved 2 Documents to Recently deleted");
    expect(top()?.name).toBe("Delete 2 Documents");
    expect(useAppStore.getState().documents.map((each) => each.id)).toEqual(["c"]);
    await useUndo.getState().undo();
    expect(method("restore")).toHaveBeenCalledWith([
      { kind: "document", id: "a" },
      { kind: "document", id: "b" },
    ]);
  });

  test("a Tag added is undone by putting each link back as it was, with who chose it", async () => {
    useAppStore.setState({
      tags: [{ id: "t", name: "Contract" } as never],
      documents: [
        {
          ...doc("a"),
          tags: [{ tagId: "t", source: "automatic", confidence: 0.4, needsReview: true }],
        } as unknown as Document,
        doc("b"),
      ],
    });
    await useAppStore.getState().addTagToDocuments(["a", "b"], "t");
    expect(top()?.message).toBe("Tagged 2 Documents Contract");
    await useUndo.getState().undo();
    expect(method("revertDocumentTags")).toHaveBeenCalledWith([
      {
        documentId: "a",
        tagId: "t",
        link: { source: "automatic", confidence: 0.4, needsReview: true },
      },
      { documentId: "b", tagId: "t", link: null },
    ]);
    await useUndo.getState().redo();
    expect(method("addTagToDocuments")).toHaveBeenCalledTimes(2);
  });

  test("a move is done again where it went, and refused, with Restore, once the Folder it came from is gone", async () => {
    const filed = assignment("a", "classified", "g");
    useAppStore.setState({ library: snapshot([filed]) });
    method("moveToFolder").mockResolvedValue({
      folderId: null,
      minds: [],
      documents: [{ id: "a", from: "g", previous: filed }],
      conversations: [],
    });
    await useAppStore.getState().moveToFolder({ documentIds: ["a"] }, null);
    expect(top()?.name).toBe("Move “a v0”");
    // Research is deleted meanwhile.
    useAppStore.setState({ library: { ...snapshot([]), groups: [] } });
    await useUndo.getState().undo();
    const problem = useUndo.getState().line?.problem;
    expect(problem?.reason).toBe("“Research” is deleted");
    expect(problem?.fix?.label).toBe("Restore Research");
    expect(method("restoreAssignment")).not.toHaveBeenCalled();
    method("getLibrary").mockResolvedValue(snapshot([]));
    await problem?.fix?.run();
    expect(method("restore")).toHaveBeenCalledWith([{ kind: "folder", id: "g" }]);
    expect(method("restoreAssignment")).toHaveBeenCalledWith("a", filed);
    await useUndo.getState().redo();
    expect(method("moveToFolder")).toHaveBeenCalledTimes(2);
  });

  test("the line and the Edit menu's name in Chinese, one and several", async () => {
    useAppStore.setState({ settings: { language: "zh-CN" } as never });
    method("moveToFolder").mockResolvedValue({
      folderId: "g",
      minds: [],
      documents: [
        { id: "a", from: null, previous: null },
        { id: "b", from: null, previous: null },
      ],
      conversations: [],
    });
    useAppStore.setState({ library: snapshot([]) });
    await useAppStore.getState().moveToFolder({ documentIds: ["a", "b"] }, "g");
    expect(top()?.message).toBe("已将 2 个文档移动到“Research”");
    expect(top()?.name).toBe("移动 2 个文档");
    await useAppStore.getState().deleteMind("m");
    expect(top()?.message).toBe("已将“Plan”移到回收站");
    expect(top()?.name).toBe("删除“Plan”");
  });
});

describe("making a Mind", () => {
  const made = {
    id: "n",
    title: "",
    autoTitle: "",
    kind: "mind" as const,
    folderId: null,
    createdAt: "",
    updatedAt: "",
  };
  let useUndo: typeof import("../../src/renderer/src/undo").useUndo;

  beforeAll(async () => {
    ({ useUndo } = await import("../../src/renderer/src/undo"));
  });

  beforeEach(() => {
    useUndo.setState({ done: [], undone: [], line: null });
    useAppStore.setState({ minds: [], settings: null });
    method("createMind").mockResolvedValue(made);
  });

  const top = () => useUndo.getState().done.at(-1);

  test("is one change, resolved to, which Undo sends to Recently deleted and Redo brings back", async () => {
    const change = await useAppStore.getState().createMind();
    expect(change).toBe(top());
    expect(change?.message).toBe("Made a new Mind");
    expect(change?.name).toBe("New Mind");
    await useUndo.getState().undo();
    expect(method("deleteMind")).toHaveBeenCalledWith("n");
    await useUndo.getState().redo();
    expect(method("restore")).toHaveBeenCalledWith([{ kind: "mind", id: "n" }]);
    expect(useUndo.getState().done).toEqual([change]);
  });

  test("one started from Documents or a Folder is said by its name, in Chinese too", async () => {
    method("createMind").mockResolvedValue({ ...made, title: "Tides" });
    await useAppStore.getState().createMind({ title: "Tides", startQuestion: true });
    expect(method("createMind")).toHaveBeenCalledWith({ title: "Tides" });
    expect(top()?.message).toBe("Started “Tides”");
    useAppStore.setState({ settings: { language: "zh-CN" } as never });
    await useAppStore.getState().createMind({ title: "Tides", startQuestion: true });
    expect(top()?.message).toBe("已新建“Tides”");
    expect(top()?.name).toBe("新建 Mind");
    method("createMind").mockResolvedValue(made);
    await useAppStore.getState().createMind();
    expect(top()?.message).toBe("已新建 Mind");
  });

  test("a Mind the core can't make is a problem, with no step to undo", async () => {
    method("createMind").mockRejectedValue(new Error("The disk is full."));
    expect(await useAppStore.getState().createMind()).toBeUndefined();
    expect(useUndo.getState().done).toEqual([]);
    const { useNotices } = await import("../../src/renderer/src/notices");
    for (const notice of useNotices.getState().errors)
      useNotices.getState().dismissError(notice.id);
  });

  test("asking about Documents makes a Mind only when there is none to ask in", async () => {
    const scope = { folderIds: [], tagIds: [], documentIds: ["a"] };
    const bridge = { kind: "documents" as const, title: "a", count: 1, scope };
    const change = await useAppStore.getState().askAbout(bridge);
    expect(change?.message).toBe("Made a new Mind");
    useUndo.setState({ done: [], undone: [], line: null });
    useAppStore.setState({ minds: [made] });
    expect(await useAppStore.getState().askAbout(bridge)).toBeUndefined();
    expect(useUndo.getState().done).toEqual([]);
  });
});

describe("changing a Linked folder", () => {
  const linked = {
    id: "l",
    path: "/Users/me/Papers",
    folderId: "f",
    status: "watching" as const,
    layout: "tree" as const,
    progress: { files: 0, indexed: 0 },
    onlineOnly: { files: 0, bytes: 0, downloading: false },
    unsupportedFiles: 0,
    createdAt: "",
    updatedAt: "",
  };
  let useUndo: typeof import("../../src/renderer/src/undo").useUndo;

  beforeAll(async () => {
    ({ useUndo } = await import("../../src/renderer/src/undo"));
  });

  beforeEach(() => {
    useUndo.setState({ done: [], undone: [], line: null });
    useAppStore.setState({ linkedFolders: [linked], settings: null });
  });

  const top = () => useUndo.getState().done.at(-1);

  test("pausing is undone by resuming it and done again by pausing", async () => {
    method("setLinkedFolderPaused").mockResolvedValue({ ...linked, status: "paused" });
    const change = await useAppStore.getState().setLinkedFolderPaused("l", true);
    expect(change).toBe(top());
    expect(change?.message).toBe("Paused reading in “Papers”");
    expect(change?.name).toBe("Pause reading");
    expect(useAppStore.getState().linkedFolders[0]?.status).toBe("paused");
    method("setLinkedFolderPaused").mockResolvedValue(linked);
    await useUndo.getState().undo();
    expect(method("setLinkedFolderPaused")).toHaveBeenLastCalledWith("l", false);
    expect(useAppStore.getState().linkedFolders[0]?.status).toBe("watching");
    method("setLinkedFolderPaused").mockResolvedValue({ ...linked, status: "paused" });
    await useUndo.getState().redo();
    expect(method("setLinkedFolderPaused")).toHaveBeenLastCalledWith("l", true);
    expect(top()?.message).toBe("Paused reading in “Papers”");
  });

  test("resuming says so, and pausing what is paused already offers nothing to undo", async () => {
    useAppStore.setState({ linkedFolders: [{ ...linked, status: "paused" }] });
    method("setLinkedFolderPaused").mockResolvedValue({ ...linked, status: "paused" });
    expect(await useAppStore.getState().setLinkedFolderPaused("l", true)).toBeUndefined();
    method("setLinkedFolderPaused").mockResolvedValue({ ...linked, status: "scanning" });
    const change = await useAppStore.getState().setLinkedFolderPaused("l", false);
    expect(change?.message).toBe("Resumed reading in “Papers”");
    expect(change?.name).toBe("Resume reading");
  });

  test("a layout is undone by the one it had and done again by the new one", async () => {
    method("setLinkedFolderLayout").mockResolvedValue({ ...linked, layout: "flat" });
    const change = await useAppStore.getState().setLinkedFolderLayout("l", "flat");
    expect(change?.message).toBe("Showing “Papers” as a flat list");
    expect(change?.name).toBe("Show as a flat list");
    await useUndo.getState().undo();
    expect(method("setLinkedFolderLayout")).toHaveBeenLastCalledWith("l", "tree");
    await useUndo.getState().redo();
    expect(method("setLinkedFolderLayout")).toHaveBeenLastCalledWith("l", "flat");
    expect(await useAppStore.getState().setLinkedFolderLayout("l", "flat")).toBeUndefined();
  });

  test("the lines in Chinese", async () => {
    useAppStore.setState({ settings: { language: "zh-CN" } as never });
    method("setLinkedFolderPaused").mockResolvedValue({ ...linked, status: "paused" });
    await useAppStore.getState().setLinkedFolderPaused("l", true);
    expect(top()?.message).toBe("已暂停读取“Papers”");
    expect(top()?.name).toBe("暂停读取");
    method("setLinkedFolderLayout").mockResolvedValue({ ...linked, layout: "flat" });
    await useAppStore.getState().setLinkedFolderLayout("l", "flat");
    expect(top()?.message).toBe("已将“Papers”显示为平铺列表");
    expect(top()?.name).toBe("显示为平铺列表");
  });
});

describe("making a Folder", () => {
  const made = {
    id: "n",
    name: "Untitled Folder",
    description: "",
    countsOwnFiles: false,
    createdAt: "",
    updatedAt: "",
  };
  let useUndo: typeof import("../../src/renderer/src/undo").useUndo;
  let useRequests: typeof import("../../src/renderer/src/requests").useRequests;
  let useNotices: typeof import("../../src/renderer/src/notices").useNotices;

  beforeAll(async () => {
    ({ useUndo } = await import("../../src/renderer/src/undo"));
    ({ useRequests } = await import("../../src/renderer/src/requests"));
    ({ useNotices } = await import("../../src/renderer/src/notices"));
  });

  /** The Library as the core has it once the new Folder is in it, named `name`. */
  const withNewFolder = (name = made.name, assignments: DocumentGroupAssignment[] = []) => {
    const library = snapshot(assignments);
    method("getLibrary").mockResolvedValue({
      ...library,
      groups: [...library.groups, { ...made, name }],
    });
  };
  const top = () => useUndo.getState().done.at(-1);

  beforeEach(() => {
    useUndo.setState({ done: [], undone: [], line: null });
    useRequests.setState({ nameFolder: null });
    useAppStore.setState({ settings: null, minds: [] });
    method("createLibraryGroup").mockResolvedValue(made);
    withNewFolder();
  });

  test("makes “Untitled Folder”, asks for its name in place, and one Undo removes it, one Redo brings it back", async () => {
    expect(await useAppStore.getState().createFolder()).toBe("n");
    expect(method("createLibraryGroup")).toHaveBeenCalledWith({
      name: "Untitled Folder",
      description: "",
    });
    expect(useRequests.getState().nameFolder).toBe("n");
    expect(top()?.message).toBe("Made “Untitled Folder”");
    expect(top()?.name).toBe("New Folder");
    await useUndo.getState().undo();
    expect(method("deleteLibraryGroup")).toHaveBeenCalledWith("n");
    await useUndo.getState().redo();
    expect(method("restore")).toHaveBeenCalledWith([{ kind: "folder", id: "n" }]);
  });

  test("a Folder the core can't make is a problem in place, with no step to undo", async () => {
    method("createLibraryGroup").mockRejectedValue(
      new Error("A folder with this name already exists."),
    );
    expect(await useAppStore.getState().createFolder()).toBeNull();
    expect(useUndo.getState().done).toEqual([]);
    const [notice] = useNotices.getState().errors;
    expect(notice?.message).toContain("already exists");
    if (notice) useNotices.getState().dismissError(notice.id);
  });

  test("a name the Folders already have is numbered, and Chinese names are made in Chinese", async () => {
    useAppStore.setState({
      library: {
        ...snapshot([]),
        groups: [{ ...made, id: "x", name: "untitled folder" }],
      },
    });
    await useAppStore.getState().createFolder();
    expect(method("createLibraryGroup")).toHaveBeenLastCalledWith({
      name: "Untitled Folder 2",
      description: "",
    });
    useAppStore.setState({ settings: { language: "zh-CN" } as never, library: snapshot([]) });
    await useAppStore.getState().createFolder();
    expect(method("createLibraryGroup")).toHaveBeenLastCalledWith({
      name: "未命名文件夹",
      description: "",
    });
    expect(top()?.message).toBe("已新建文件夹“未命名文件夹”");
  });

  test("its first name is part of making it: one step, named, which one Undo removes", async () => {
    await useAppStore.getState().createFolder();
    withNewFolder("Chapter 2");
    method("updateLibraryGroup").mockResolvedValue({ ...made, name: "Chapter 2" });
    await useAppStore.getState().nameNewFolder("n", "  Chapter 2 ");
    expect(method("updateLibraryGroup")).toHaveBeenCalledWith("n", {
      name: "Chapter 2",
      description: "",
    });
    expect(useUndo.getState().done).toHaveLength(1);
    expect(top()?.message).toBe("Made “Chapter 2”");
    await useUndo.getState().undo();
    expect(method("deleteLibraryGroup")).toHaveBeenCalledWith("n");
    expect(method("updateLibraryGroup")).toHaveBeenCalledTimes(1);
  });

  test("a blank name, or one the Folder has, changes nothing; a later name is a step of its own", async () => {
    await useAppStore.getState().createFolder();
    await useAppStore.getState().nameNewFolder("n", "   ");
    await useAppStore.getState().nameNewFolder("n", "Untitled Folder");
    expect(method("updateLibraryGroup")).not.toHaveBeenCalled();
    expect(top()?.message).toBe("Made “Untitled Folder”");
    // Another change came after it, so the name no longer belongs to making it.
    useUndo.getState().push({ message: "Something else", undo: async () => undefined });
    method("updateLibraryGroup").mockResolvedValue({ ...made, name: "Later" });
    await useAppStore.getState().nameNewFolder("n", "Later");
    expect(useUndo.getState().done.map((each) => each.message)).toEqual([
      "Made “Untitled Folder”",
      "Something else",
      "Renamed to “Later”",
    ]);
  });

  test("with Documents and Minds: they move in at once, in one step, and Undo puts each back before the Folder goes", async () => {
    const filed = assignment("a", "classified", "g");
    useAppStore.setState({
      documents: [doc("a"), doc("b"), doc("c")],
      minds: [
        {
          id: "m",
          title: "Plan",
          autoTitle: "",
          kind: "mind",
          folderId: "g",
          createdAt: "",
          updatedAt: "",
        },
      ],
      library: snapshot([filed]),
    });
    method("moveToFolder").mockResolvedValue({
      folderId: "n",
      minds: [{ id: "m", from: "g" }],
      documents: [
        { id: "a", from: "g", previous: filed },
        { id: "b", from: null, previous: null },
        { id: "c", from: null, previous: null },
      ],
      conversations: [],
    });
    withNewFolder("Untitled Folder", [filed]);

    await useAppStore.getState().createFolder({ mindIds: ["m"], documentIds: ["a", "b", "c"] });
    expect(method("moveToFolder")).toHaveBeenCalledWith({
      mindIds: ["m"],
      documentIds: ["a", "b", "c"],
      conversationIds: [],
      folderId: "n",
    });
    expect(top()?.message).toBe("Made “Untitled Folder” with 4 items");

    await useUndo.getState().undo();
    expect(method("moveToFolder")).toHaveBeenLastCalledWith({
      mindIds: ["m"],
      documentIds: [],
      conversationIds: [],
      folderId: "g",
    });
    expect(method("restoreAssignment")).toHaveBeenCalledWith("a", filed);
    expect(method("restoreAssignment")).toHaveBeenCalledWith("b", null);
    expect(method("restoreAssignment")).toHaveBeenCalledWith("c", null);
    expect(method("deleteLibraryGroup").mock.invocationCallOrder[0]).toBeGreaterThan(
      method("restoreAssignment").mock.invocationCallOrder[2] ?? 0,
    );

    await useUndo.getState().redo();
    expect(method("restore")).toHaveBeenCalledWith([{ kind: "folder", id: "n" }]);
    expect(method("moveToFolder")).toHaveBeenLastCalledWith({
      mindIds: ["m"],
      documentIds: ["a", "b", "c"],
      conversationIds: [],
      folderId: "n",
    });
  });

  test("the line says how many went in, in English and in Chinese", async () => {
    useAppStore.setState({ documents: [doc("a"), doc("b"), doc("c")] });
    method("moveToFolder").mockResolvedValue({
      folderId: "n",
      minds: [],
      documents: ["a", "b", "c"].map((id) => ({ id, from: null, previous: null })),
      conversations: [],
    });
    await useAppStore.getState().createFolder({ documentIds: ["a", "b", "c"] });
    expect(top()?.message).toBe("Made “Untitled Folder” with 3 Documents");
    useUndo.setState({ done: [], undone: [], line: null });
    useAppStore.setState({ settings: { language: "zh-CN" } as never, library: snapshot([]) });
    await useAppStore.getState().createFolder({ documentIds: ["a"] });
    expect(top()?.message).toBe("已新建文件夹“未命名文件夹”，放入“a v0”");
    useAppStore.setState({ library: snapshot([]) });
    await useAppStore.getState().createFolder({ documentIds: ["a", "b", "c"] });
    expect(top()?.message).toBe("已新建文件夹“未命名文件夹”，放入 3 个文档");
  });
});

describe("loading keeps a list the core pushed after reading it", () => {
  beforeEach(() => {
    method("getExamples").mockResolvedValue({
      group: "tea",
      available: false,
      mindId: null,
      linkedFolderId: null,
      answerId: null,
      more: [],
    });
  });
  const folder = (id: string) => ({ id, name: id }) as unknown as Folder;
  const folderIds = () => useAppStore.getState().folders.map((each) => each.id);
  const later = <T>() => {
    let reply: (value: T) => void = () => undefined;
    const promise = new Promise<T>((resolve) => {
      reply = resolve;
    });
    return { promise, reply: (value: T) => reply(value) };
  };

  test("a Folder pushed while loading still waits stays", async () => {
    const connectors = later<never[]>();
    method("listConnectors").mockReturnValue(connectors.promise);
    method("listFolders").mockResolvedValue([folder("mine")]);

    const loading = useAppStore.getState().load();
    await settle();
    await settle();
    // The example Linked folder's first scan makes its Folder after the list was read.
    emit("folders.changed", [folder("mine"), folder("examples")]);
    connectors.reply([]);
    await loading;

    expect(folderIds()).toEqual(["mine", "examples"]);
  });

  test("a list pushed before it was read gives way to the read, which is newer", async () => {
    const folders = later<Folder[]>();
    method("listConnectors").mockResolvedValue([]);
    method("listFolders").mockReturnValue(folders.promise);

    const loading = useAppStore.getState().load();
    await settle();
    emit("folders.changed", [folder("mine")]);
    folders.reply([folder("mine"), folder("examples")]);
    await loading;

    expect(folderIds()).toEqual(["mine", "examples"]);
  });
});

describe("Organize says what it did (#763)", () => {
  test("once it has gone through the Documents it says how many got a Suggested Folder, and Undo puts each back", async () => {
    const { sayWhenOrganized } = await import("../../src/renderer/src/organizeRun");
    const { useUndo } = await import("../../src/renderer/src/undo");
    const before = new Map([
      ["a", assignment("a", "classified")],
      ["b", assignment("b", "classified")],
    ]);
    sayWhenOrganized(["a", "b"], before);

    useAppStore.setState({
      library: snapshot([assignment("a", "pending"), assignment("b", "classifying")]),
    });
    const earlier = useUndo.getState().line;
    expect(useUndo.getState().line).toBe(earlier);
    const suggested = (id: string) => ({
      ...assignment(id, "classified", "g"),
      suggested: true,
    });
    useAppStore.setState({ library: snapshot([suggested("a"), suggested("b")]) });

    const change = useUndo.getState().line?.change;
    expect(change?.message).toBe("Organize suggested a Folder for 2 Documents");
    await useUndo.getState().undo();
    const restored = method("restoreAssignment");
    expect(restored).toHaveBeenCalledTimes(2);
    expect(restored.mock.calls[0]?.[0]).toBe("a");
    expect(restored.mock.calls[0]?.[1]).toMatchObject({ groupId: null, suggested: false });
  });

  test("Undo leaves a Document the User moved since", async () => {
    const { sayWhenOrganized } = await import("../../src/renderer/src/organizeRun");
    const { useUndo } = await import("../../src/renderer/src/undo");
    method("restoreAssignment").mockClear();
    const before = new Map([
      ["a", assignment("a", "classified")],
      ["b", assignment("b", "classified")],
    ]);
    useAppStore.setState({
      library: snapshot([assignment("a", "pending"), assignment("b", "pending")]),
    });
    sayWhenOrganized(["a", "b"], before);
    const suggested = (id: string) => ({ ...assignment(id, "classified", "g"), suggested: true });
    useAppStore.setState({ library: snapshot([suggested("a"), suggested("b")]) });
    // The User puts b somewhere else.
    useAppStore.setState({
      library: snapshot([
        suggested("a"),
        { ...assignment("b", "classified", null), source: "user" },
      ]),
    });
    await useUndo.getState().undo();
    const restored = method("restoreAssignment");
    expect(restored).toHaveBeenCalledTimes(1);
    expect(restored.mock.calls[0]?.[0]).toBe("a");
  });
});
