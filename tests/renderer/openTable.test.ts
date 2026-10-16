import { beforeEach, describe, expect, test, vi } from "vitest";
import type { ConversationAnswer } from "../../src/core/api";

const calls: string[] = [];
const fill = { release: () => {}, fails: false };

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

const answer = { id: "a1", conversationId: "c1", model: null } as unknown as ConversationAnswer;
const table = { type: "table", content: [] };

beforeEach(() => {
  calls.length = 0;
  pushed.length = 0;
  fill.fails = false;
  reportError.mockClear();
});

describe("Open as a table", () => {
  test("its tab is opened before the table is put in, so the tab shows while a big table is still going in", async () => {
    const { openTable } = await import("../../src/renderer/src/addToNote");
    const opening = openTable({ answer, conversation: undefined, blocks: [table] });
    // The table is still going in: the tab is already open, and nothing says it is done.
    await vi.waitFor(() => expect(calls).toEqual(["createMind", "changeLayout", "addToNote"]));
    expect(pushed).toHaveLength(0);
    fill.release();
    await opening;
    expect(pushed).toHaveLength(1);
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
    expect(pushed).toHaveLength(0);
  });
});
