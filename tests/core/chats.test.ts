import { randomUUID } from "node:crypto";
import { writeFile } from "node:fs/promises";
import type { JSONContent } from "@tiptap/core";
import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test, vi } from "vitest";
import * as Y from "yjs";
import {
  blockReference,
  type CitationAttributes,
  type Core,
  MIND_CONTENT_FIELD,
} from "../../src/core";
import { allOrigins } from "../../src/shared/blockOrigins";
import { chatContaining, listChats } from "../../src/shared/mindChats";
import { answerEnded, citingModel, setUpWithDocuments } from "../helpers/citations";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { sha256, waitForProcessing } from "../helpers/documents";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import {
  chatAnswer,
  chatOutline,
  note,
  outline,
  question,
  readChat,
  readMind,
  writeChat,
  writeMind,
} from "../helpers/minds";
import { controlledModel, promptOf, scriptedModels, streamingModel } from "../helpers/models";
import { buildPdf } from "../helpers/pdf";

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon's gravity raises two tidal bulges on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);
const SPRING = "Spring tides happen at new moon and at full moon.";
/** The same Document, rewritten: the quote is nowhere in it. */
const CHANGED = buildPdf([
  { lines: ["Tides and the Moon", "The Moon's gravity raises two tidal bulges on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides follow the new and the full moon."] },
]);

async function setUp(model: MockLanguageModelV4) {
  const models = scriptedModels(model);
  const core = startCore(await createTempDataFolder(), { createChatModel: models.createChatModel });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const mind = await core.createMind({ title: "Tides" });
  const client = await connectToMind(core, mind.id);
  return { core, mind, client };
}

/** Puts a Question at the end of a chat, as the thread does, and asks it there. */
async function askInChat(
  core: Core,
  client: MindClient,
  mindId: string,
  chatId: string,
  text: string,
  before: JSONContent[] = [],
) {
  const asked = question(text);
  writeChat(client, chatId, [...before, asked]);
  await client.settled();
  const result = await core.askQuestion({ mindId, questionId: asked.attrs.id, chatId });
  if (!result.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(result)}`);
  return { questionId: asked.attrs.id, answerId: result.answerId };
}

/** The Mind's stored note, encoded: unchanged byte for byte if nothing wrote to it. */
const storedNote = async (core: Core, mindId: string) => {
  const doc = new Y.Doc();
  Y.applyUpdate(doc, (await core.openMind(mindId)).state);
  return doc.getXmlFragment(MIND_CONTENT_FIELD).toJSON();
};

describe("Asking in a Mind's chat (#201)", () => {
  test("the Answer streams into the chat, right after its Question, and the note is left as it was", async () => {
    const controlled = controlledModel();
    const { core, mind, client } = await setUp(controlled.model);
    writeMind(client, [note("Parks cool cities at night."), note("")]);
    await client.settled();
    const before = await storedNote(core, mind.id);

    const chatId = randomUUID();
    const { questionId, answerId } = await askInChat(
      core,
      client,
      mind.id,
      chatId,
      "How much cooler are parks?",
    );
    // In the chat: the Question, then its Answer being written; no line to write on after it.
    await vi.waitFor(() =>
      expect(chatOutline(client, chatId)).toEqual(["question", "answer:streaming"]),
    );
    await controlled.requested();
    controlled.push("About one degree.");
    await vi.waitFor(() =>
      expect(chatAnswer(client, chatId, answerId).textContent).toBe("About one degree."),
    );
    const finished = answerEnded(core, answerId);
    controlled.finish();
    expect((await finished).event).toBe("finished");
    await vi.waitFor(() =>
      expect(chatOutline(client, chatId)).toEqual(["question", "answer:done"]),
    );
    expect(chatAnswer(client, chatId, answerId).attrs.questionId).toBe(questionId);

    // The note's stored content is unchanged.
    expect(outline(client)).toEqual(["paragraph", "paragraph"]);
    expect(await storedNote(core, mind.id)).toEqual(before);
    // The thread's history lists the chat, titled by its first Question.
    expect(listChats(client.doc)).toEqual([
      {
        id: chatId,
        createdAt: expect.any(String),
        title: "How much cooler are parks?",
        questions: 1,
      },
    ]);
    expect(chatContaining(client.doc, answerId)).toBe(chatId);
  });

  test("its Question context is the note, then the chat's earlier Questions and Answers", async () => {
    const model = streamingModel(["Fine."]);
    const { core, mind, client } = await setUp(model);
    writeMind(client, [
      note("Parks cool cities at night."),
      note("A side note", { off: true }),
      note("Ask Dr Hale about Leeds."),
    ]);
    const chatId = randomUUID();
    const first = await askInChat(core, client, mind.id, chatId, "How much cooler are parks?");
    await answerEnded(core, first.answerId);
    await client.settled();

    const asked = question("Does size matter?");
    writeChat(client, chatId, [...readChatJSON(client, chatId), asked]);
    await client.settled();
    const result = await core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id, chatId });
    expect(result.asked).toBe(true);
    if (result.asked) await answerEnded(core, result.answerId);

    const prompt = promptOf(model, 1);
    expect(prompt.map((message) => message.role)).toEqual(["system", "user", "assistant", "user"]);
    expect(prompt[1]?.text).toBe(
      "Parks cool cities at night.\n\nAsk Dr Hale about Leeds.\n\nHow much cooler are parks?",
    );
    expect(prompt[2]?.text).toBe("Fine.");
    expect(prompt[3]?.text).toBe("Does size matter?");
  });

  test("Regenerate writes a chat's Answer again in place; Stop keeps what was written", async () => {
    const controlled = controlledModel();
    const { core, mind, client } = await setUp(controlled.model);
    const chatId = randomUUID();
    const { answerId } = await askInChat(core, client, mind.id, chatId, "Why?");
    await controlled.requested();
    controlled.push("Because");
    await vi.waitFor(() =>
      expect(chatAnswer(client, chatId, answerId).textContent).toBe("Because"),
    );
    await core.stopAnswer({ mindId: mind.id, answerId });
    await vi.waitFor(() =>
      expect(chatOutline(client, chatId)).toEqual(["question", "answer:stopped"]),
    );
    expect(chatAnswer(client, chatId, answerId).textContent).toBe("Because");

    const again = await core.regenerateAnswer({ mindId: mind.id, answerId, chatId });
    expect(again).toEqual({ asked: true, answerId });
    await vi.waitFor(() =>
      expect(chatOutline(client, chatId)).toEqual(["question", "answer:streaming"]),
    );
    await controlled.requested(2);
    controlled.push("Of the Moon.");
    controlled.finish();
    await vi.waitFor(() =>
      expect(chatOutline(client, chatId)).toEqual(["question", "answer:done"]),
    );
    expect(chatAnswer(client, chatId, answerId).textContent).toBe("Of the Moon.");
    expect(outline(client)).toEqual([]);
  });

  test("a chat that isn't in the Mind, or a Question that isn't in the chat, is refused", async () => {
    const { core, mind, client } = await setUp(streamingModel(["Fine."]));
    const asked = question("In the note");
    writeMind(client, [asked]);
    await client.settled();
    await expect(
      core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id, chatId: randomUUID() }),
    ).rejects.toThrow("That chat isn't in this Mind.");
    const chatId = randomUUID();
    writeChat(client, chatId, [question("In the chat")]);
    await client.settled();
    await expect(
      core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id, chatId }),
    ).rejects.toThrow("That Question isn't in this Mind.");
    await expect(
      core.askQuestion({
        mindId: mind.id,
        questionId: asked.attrs.id,
        chatId: 4 as unknown as string,
      }),
    ).rejects.toThrow("A chat id must be text.");
  });

  test("an Answer a quit left streaming in a chat is stopped when the Mind opens again", async () => {
    const dataDir = await createTempDataFolder();
    const controlled = controlledModel();
    const models = scriptedModels(controlled.model);
    const core = startCore(dataDir, { createChatModel: models.createChatModel });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const mind = await core.createMind({ title: "Tides" });
    const client = await connectToMind(core, mind.id);
    const chatId = randomUUID();
    const { answerId } = await askInChat(core, client, mind.id, chatId, "Why?");
    await controlled.requested();
    controlled.push("Half an");
    await vi.waitFor(() =>
      expect(chatAnswer(client, chatId, answerId).textContent).toBe("Half an"),
    );
    // Quit without stopping: the stored Answer is still "streaming".
    core.close();

    const reopened = startCore(dataDir, { createChatModel: models.createChatModel });
    const again = await connectToMind(reopened, mind.id);
    expect(chatOutline(again, chatId)).toEqual(["question", "answer:stopped"]);
    expect(chatAnswer(again, chatId, answerId).textContent).toBe("Half an");
    reopened.close();
  });
});

describe("Add to note, into a Mind that isn't open (#201)", () => {
  test("an Answer's Blocks go at the end of the Mind's note with their Citations checked there, and a link back; Undo takes them out", async () => {
    const model = citingModel({
      query: "spring tides",
      records: (passages) => [
        { marker: 1, passage: passages[0]?.id, pageFrom: 2, pageTo: 2, quote: SPRING },
      ],
      answer: "Spring tides come at new and full moon [^1].",
    });
    const { core, mind, client, documents } = await setUpWithDocuments(model, [
      { name: "Tides.pdf", contents: TIDES },
    ]);
    const chatId = randomUUID();
    const { answerId } = await askInChat(core, client, mind.id, chatId, "When are spring tides?");
    const ended = await answerEnded(core, answerId);
    expect(ended.event).toBe("finished");
    await client.settled();
    const answer = chatAnswer(client, chatId, answerId).toJSON() as JSONContent;
    const cited = (answer.content?.[0]?.content ?? []).find((node) => node.type === "citation");
    // As if it had been checked against an earlier version: "Add to note" checks it again.
    const blocks = (answer.content ?? []).map((block) => ({
      ...block,
      content: block.content?.map((node) =>
        node.type === "citation"
          ? {
              ...node,
              attrs: { ...node.attrs, check: "not-found", checkReason: "quote-not-on-pages" },
            }
          : node,
      ),
    }));
    expect(cited?.attrs?.check).toBe("found");

    const other = await core.createMind({ title: "Chapter 2 outline" });
    const target = await connectToMind(core, other.id);
    writeMind(target, [note("My outline."), note("")]);
    await target.settled();
    const origin = { from: blockReference(mind.id, answerId), label: "When are spring tides?" };
    const added = await core.addToNote({ mindId: other.id, blocks, origin });
    expect(added.mindId).toBe(other.id);
    expect(added.blockIds).toHaveLength(1);

    // At the end, in place of the empty last line, with a new id; its Citation checked again: found.
    await vi.waitFor(() => expect(outline(target)).toEqual(["paragraph", "paragraph"]));
    const doc = readMind(target);
    expect(doc.child(1).attrs.id).toBe(added.blockIds[0]);
    expect(doc.child(1).attrs.id).not.toBe(answer.content?.[0]?.attrs?.id);
    const citations: CitationAttributes[] = [];
    doc.child(1).descendants((node) => {
      if (node.type.name === "citation") citations.push(node.attrs as CitationAttributes);
    });
    expect(citations).toEqual([
      expect.objectContaining({
        documentId: documents[0]?.id,
        quote: SPRING,
        check: "found",
        checkReason: null,
        pageFrom: 2,
        pageTo: 2,
      }),
    ]);
    // Its link back names the Answer in the chat.
    expect(allOrigins(target.doc)).toEqual(new Map([[added.blockIds[0] as string, origin]]));
    // The Mind it came from is unchanged.
    expect(outline(client)).toEqual([]);

    await core.undoAddToNote({ mindId: other.id, blockIds: added.blockIds });
    await vi.waitFor(() => expect(outline(target)).toEqual(["paragraph"]));
    expect(readMind(target).child(0).textContent).toBe("My outline.");
    expect(allOrigins(target.doc).size).toBe(0);
  });

  test("only Notes go into a note, and only a Mind's: a chat, or anything else, is refused", async () => {
    const { core, mind } = await setUp(streamingModel(["Fine."]));
    const paragraph = { type: "paragraph", content: [{ type: "text", text: "Hi" }] };
    await expect(core.addToNote({ mindId: mind.id, blocks: [], origin: null })).rejects.toThrow(
      "addToNote takes a list of Notes",
    );
    await expect(
      core.addToNote({ mindId: mind.id, blocks: [{ type: "question" }], origin: null }),
    ).rejects.toThrow("addToNote takes a list of Notes");
    await expect(
      core.addToNote({ mindId: mind.id, blocks: [paragraph], origin: { from: "x" } as never }),
    ).rejects.toThrow("That isn't a Block's origin.");
    const chat = await core.createMind({ kind: "chat" });
    await expect(
      core.addToNote({ mindId: chat.id, blocks: [paragraph], origin: null }),
    ).rejects.toThrow("Answers are added to a Mind's note, not to a chat.");
    // Added with no origin: no link back.
    const added = await core.addToNote({ mindId: mind.id, blocks: [paragraph], origin: null });
    const client = await connectToMind(core, mind.id);
    expect(outline(client)).toEqual(["paragraph"]);
    expect(allOrigins(client.doc).size).toBe(0);
    expect(added.blockIds).toHaveLength(1);
  });

  test("old versions' text stays while a Citation in a chat quotes it", {
    timeout: 30_000,
  }, async () => {
    const model = citingModel({
      query: "spring tides",
      records: (passages) => [
        { marker: 1, passage: passages[0]?.id, pageFrom: 2, pageTo: 2, quote: SPRING },
      ],
      answer: "Spring tides come at new and full moon [^1].",
    });
    const { core, mind, client, documents, dataDir } = await setUpWithDocuments(model, [
      { name: "Tides.pdf", contents: TIDES },
    ]);
    const document = documents[0];
    if (!document) throw new Error("Nothing was added.");
    const chatId = randomUUID();
    const { answerId } = await askInChat(core, client, mind.id, chatId, "When are spring tides?");
    expect((await answerEnded(core, answerId)).event).toBe("finished");
    await client.settled();

    // A new version: the cited one is old now, and only the chat's Citation quotes it.
    await writeFile(document.path, CHANGED);
    await core.reconcileDocuments();
    await waitForProcessing(core, [document.id]);
    core.close();
    startCore(dataDir).close();
    expect(versionsKept(dataDir)).toEqual([sha256(CHANGED), sha256(TIDES)].sort());
  });
});

/** A chat's Blocks as JSON, to write back with more after them. */
function readChatJSON(client: MindClient, chatId: string): JSONContent[] {
  const nodes: JSONContent[] = [];
  readChat(client, chatId).forEach((node) => {
    nodes.push(node.toJSON() as JSONContent);
  });
  return nodes;
}

/** The versions of Documents whose text is kept, by content hash. */
const versionsKept = (dataDir: string) =>
  queryDatabase<{ content_hash: string }>(
    dataDir,
    "SELECT DISTINCT content_hash FROM document_pages WHERE deleted_at IS NULL",
  )
    .map((row) => row.content_hash)
    .sort();
