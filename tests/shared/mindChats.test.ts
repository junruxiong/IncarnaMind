import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import { blockReference } from "../../src/core";
import { placeInThread, takeBackFromThread } from "../../src/renderer/src/editor/threadAsk";
import { allOrigins, originOf, removeOrigins, setOrigins } from "../../src/shared/blockOrigins";
import {
  chatBlocks,
  chatContaining,
  chatSummary,
  createChat,
  listChats,
  observeChats,
} from "../../src/shared/mindChats";
import {
  countCitations,
  parseNoteBlocks,
  recheckCitations,
  withoutIds,
} from "../../src/shared/noteBlocks";

const NO_SCOPE = { folderIds: [], tagIds: [], documentIds: [] };

const ask = (doc: Y.Doc, chatId: string | null, text: string) =>
  placeInThread(doc, chatId, {
    id: crypto.randomUUID(),
    text,
    model: null,
    scope: NO_SCOPE,
    skill: null,
  });

describe("A Mind's chats", () => {
  test("a Question asked with no chat starts one, titled by it; the next goes into the same chat", () => {
    const doc = new Y.Doc();
    const first = ask(doc, null, "How much cooler\nare parks?");
    expect(first.startedChat).toBe(true);
    const second = ask(doc, first.chatId, "Does size matter?");
    expect(second).toMatchObject({ chatId: first.chatId, startedChat: false });
    expect(listChats(doc)).toEqual([
      {
        id: first.chatId,
        createdAt: expect.any(String),
        title: "How much cooler are parks?",
        questions: 2,
      },
    ]);
    // Its Blocks are Questions in the stored shape: a line break between the lines.
    const blocks = chatBlocks(doc, first.chatId);
    expect(blocks?.toString()).toMatch(/<hardbreak/i);
    expect(chatContaining(doc, second.questionId)).toBe(first.chatId);
    expect(chatContaining(doc, "nowhere")).toBeNull();
  });

  test("a chat that is gone gets a new one; the newest is listed first", () => {
    const doc = new Y.Doc();
    createChat(doc, "old", "2026-10-01T09:00:00.000Z");
    const placed = ask(doc, "gone", "Where did it go?");
    expect(placed.chatId).not.toBe("gone");
    expect(listChats(doc).map((chat) => chat.id)).toEqual([placed.chatId, "old"]);
    expect(chatSummary(doc, "old")).toMatchObject({ title: "", questions: 0 });
  });

  test("a Question that couldn't be asked goes back out, and the chat it started with it", () => {
    const doc = new Y.Doc();
    const placed = ask(doc, null, "Is anyone there?");
    takeBackFromThread(doc, placed);
    expect(listChats(doc)).toEqual([]);
    const kept = ask(doc, null, "First");
    const second = ask(doc, kept.chatId, "Second");
    takeBackFromThread(doc, second);
    expect(chatSummary(doc, kept.chatId)).toMatchObject({ title: "First", questions: 1 });
  });

  test("chats sync like the rest of the Mind, and observers hear of changes", () => {
    const a = new Y.Doc();
    const b = new Y.Doc();
    a.on("update", (update: Uint8Array) => Y.applyUpdate(b, update));
    let heard = 0;
    const stop = observeChats(b, () => heard++);
    const placed = ask(a, null, "Synced?");
    expect(listChats(b)).toEqual(listChats(a));
    expect(heard).toBeGreaterThan(0);
    stop();
    expect(chatContaining(b, placed.questionId)).toBe(placed.chatId);
  });
});

describe("Where Blocks came from", () => {
  test("a Note's link back is kept by Block id, beside the Blocks, and goes with Undo", () => {
    const doc = new Y.Doc();
    const origin = { from: blockReference("mind-1", "answer-1"), label: "Why?" };
    setOrigins(doc, ["block-1", "block-2"], origin);
    expect(originOf(doc, "block-1")).toEqual(origin);
    expect(allOrigins(doc).size).toBe(2);
    removeOrigins(doc, ["block-1"]);
    expect(originOf(doc, "block-1")).toBeNull();
    // Anything malformed reads as no origin.
    doc.getMap("origins").set("bad", { from: { artifactId: "" } });
    expect(originOf(doc, "bad")).toBeNull();
  });
});

describe("Blocks on their way into a note", () => {
  const cited = {
    type: "paragraph",
    attrs: { id: "p1" },
    content: [
      { type: "text", text: "Spring tides" },
      {
        type: "citation",
        attrs: { documentId: "doc-1", quote: "Spring tides", pageFrom: 2, check: "not-found" },
      },
    ],
  };

  test("only Notes are taken, and they lose their ids at any depth", () => {
    expect(parseNoteBlocks([])).toBeNull();
    expect(parseNoteBlocks([{ type: "question" }])).toBeNull();
    expect(parseNoteBlocks("nope")).toBeNull();
    const list = {
      type: "bulletList",
      attrs: { id: "l1" },
      content: [{ type: "listItem", content: [{ type: "paragraph", attrs: { id: "p2" } }] }],
    };
    const parsed = parseNoteBlocks([cited, list]);
    expect(parsed).toHaveLength(2);
    expect(JSON.stringify(withoutIds(parsed ?? []))).not.toMatch(/"id"/);
    expect(countCitations(parsed ?? [])).toBe(1);
  });

  test("each Citation is checked again where it lands; one whose Document is gone keeps its check", async () => {
    const rechecked = await recheckCitations([cited], async (input) => ({
      check: input.documentId === "doc-1" ? "found" : "cant-check",
      checkReason: null,
      contentHash: "v2",
      passageId: "passage-2",
      pageFrom: 2,
      pageTo: 2,
      location: { kind: "page", from: 2, to: 2 },
    }));
    expect(rechecked[0]?.content?.[1]?.attrs).toMatchObject({
      check: "found",
      contentHash: "v2",
      passageId: "passage-2",
      location: { kind: "page", from: 2, to: 2 },
    });
    const gone = await recheckCitations([cited], () => ({
      check: "cant-check",
      checkReason: "document-removed",
      contentHash: null,
      passageId: null,
      pageFrom: 2,
      pageTo: 2,
      location: null,
    }));
    expect(gone[0]?.content?.[1]?.attrs).toMatchObject({ check: "not-found" });
    const failing = await recheckCitations([cited], () => {
      throw new Error("offline");
    });
    expect(failing[0]?.content?.[1]?.attrs).toMatchObject({ check: "not-found" });
  });
});
