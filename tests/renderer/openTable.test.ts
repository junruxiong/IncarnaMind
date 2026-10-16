import { beforeEach, describe, expect, test, vi } from "vitest";
import type { ConversationAnswer } from "../../src/core/api";
import type { NodeJSON } from "../../src/shared/blockJson";

const calls: string[] = [];
const fill = { release: () => {}, fails: false, cleanupFails: false };

vi.mock("../../src/renderer/src/core", () => ({
  core: {
    on: () => () => undefined,
    createMind: async () => {
      calls.push("createMind");
      return { id: "made", title: "Tides", folderId: null };
    },
    addToNote: () => {
      calls.push("addToNote");
      return new Promise((resolve, reject) => {
        fill.release = () => (fill.fails ? reject(new Error("full")) : resolve({ blockIds: [] }));
      });
    },
    deleteMind: async () => {
      calls.push("deleteMind");
      if (fill.cleanupFails) throw new Error("locked");
    },
  },
}));

const changeLayout = vi.fn(async () => {
  calls.push("changeLayout");
  return true;
});
const reportError = vi.fn();
vi.mock("../../src/renderer/src/store", () => ({
  useAppStore: { getState: () => ({ changeLayout, reportError, minds: [] }) },
}));

const pushed: unknown[] = [];
vi.mock("../../src/renderer/src/undo", () => ({
  answerUndoPlace: (id: string) => `answer:${id}`,
  useUndo: { getState: () => ({ push: (entry: unknown) => pushed.push(entry) }) },
}));

const answer: ConversationAnswer = {
  kind: "answer",
  id: "a1",
  conversationId: "c1",
  questionId: "q1",
  text: "",
  citations: [],
  figures: [],
  model: null,
  status: "done",
  errorKind: null,
  errorMessage: null,
  citationSupport: null,
  toolCalls: [],
  textOrigin: null,
  read: null,
  steps: null,
  createdAt: "2026-10-16T00:00:00.000Z",
  updatedAt: "2026-10-16T00:00:00.000Z",
};
const table: NodeJSON = { type: "table", content: [] };

beforeEach(() => {
  calls.length = 0;
  pushed.length = 0;
  fill.fails = false;
  fill.cleanupFails = false;
  reportError.mockClear();
});

describe("Open as a table", () => {
  test("its tab is opened before the table is put in, so the tab shows while a big table is still going in", async () => {
    const { openTable } = await import("../../src/renderer/src/addToNote");
    const opening = openTable({ answer, conversation: undefined, blocks: [table] });
    await vi.waitFor(() => expect(calls).toEqual(["createMind", "changeLayout", "addToNote"]));
    expect(pushed).toHaveLength(0);
    fill.release();
    await opening;
    expect(pushed).toHaveLength(1);
  });

  test("a second click while one is under way makes no second Mind", async () => {
    const { openTable } = await import("../../src/renderer/src/addToNote");
    const first = openTable({ answer, conversation: undefined, blocks: [table] });
    await openTable({ answer, conversation: undefined, blocks: [table] });
    await vi.waitFor(() => expect(calls).toContain("addToNote"));
    fill.release();
    await first;
    expect(calls.filter((call) => call === "createMind")).toHaveLength(1);
    expect(pushed).toHaveLength(1);
  });

  test("once it is done, the next one opens its own Mind", async () => {
    const { openTable } = await import("../../src/renderer/src/addToNote");
    for (const time of [1, 2]) {
      const opening = openTable({ answer, conversation: undefined, blocks: [table] });
      await vi.waitFor(() =>
        expect(calls.filter((call) => call === "addToNote")).toHaveLength(time),
      );
      fill.release();
      await opening;
    }
    expect(calls.filter((call) => call === "createMind")).toHaveLength(2);
  });

  test("a table that can't go in leaves no empty Mind behind, and says so", async () => {
    const { openTable } = await import("../../src/renderer/src/addToNote");
    fill.fails = true;
    const opening = openTable({ answer, conversation: undefined, blocks: [table] });
    await vi.waitFor(() => expect(calls).toContain("addToNote"));
    fill.release();
    await opening;
    expect(calls).toEqual(["createMind", "changeLayout", "addToNote", "deleteMind"]);
    expect(reportError).toHaveBeenCalledOnce();
    expect(reportError).toHaveBeenCalledWith(new Error("full"));
    expect(pushed).toHaveLength(0);
  });

  test("when the Mind can't be deleted either, the table's own problem is still the first shown", async () => {
    const { openTable } = await import("../../src/renderer/src/addToNote");
    fill.fails = true;
    fill.cleanupFails = true;
    const opening = openTable({ answer, conversation: undefined, blocks: [table] });
    await vi.waitFor(() => expect(calls).toContain("addToNote"));
    fill.release();
    await opening;
    expect(reportError.mock.calls).toEqual([[new Error("full")], [new Error("locked")]]);
  });
});
