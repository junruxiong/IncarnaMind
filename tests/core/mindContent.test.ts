import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import * as Y from "yjs";
import {
  type Core,
  DATABASE_FILE,
  InvalidInputError,
  MIND_CONTENT_FIELD,
  NotFoundError,
} from "../../src/core";
import { COMPACT_AFTER_UPDATES, createMindContent } from "../../src/core/mindContent";
import { migrate, openDatabase } from "../../src/core/storage";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { appendParagraph, connectToMind, typeInto } from "../helpers/mindClient";

/** How many update rows the Mind has in storage: one once compacted. */
function storedRows(dataDir: string, mindId: string): number {
  const [row] = queryDatabase<{ count: number }>(
    dataDir,
    "SELECT count(*) AS count FROM mind_updates WHERE mind_id = ? AND deleted_at IS NULL",
    [mindId],
  );
  return row?.count ?? 0;
}

/** A valid Yjs update that adds one paragraph. */
function paragraphUpdate(text: string): Uint8Array {
  const doc = new Y.Doc();
  const paragraph = new Y.XmlElement("paragraph");
  paragraph.insert(0, [new Y.XmlText(text)]);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph]);
  return Y.encodeStateAsUpdate(doc);
}

describe("Mind content", () => {
  test("a new Mind opens with no Blocks", async () => {
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind({ title: "Empty" });

    const opened = await core.openMind(mind.id);

    expect(opened.mind).toEqual(mind);
    expect(opened.state).toBeInstanceOf(Uint8Array);
    const doc = new Y.Doc();
    Y.applyUpdate(doc, opened.state);
    expect(doc.getXmlFragment(MIND_CONTENT_FIELD).length).toBe(0);
  });

  test("a second client connected through the core sees the first client's edits and converges", async () => {
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind();
    const first = await connectToMind(core, mind.id);
    const second = await connectToMind(core, mind.id);

    appendParagraph(first, "Notes from the first client");
    await first.settled();
    expect(second.blocks.toString()).toBe("<paragraph>Notes from the first client</paragraph>");

    // Concurrent edits: the second client edits without seeing the first client's latest change.
    second.goOffline();
    typeInto(first, 0, 0, "Edited: ");
    typeInto(second, 0, 27, " (and the second)");
    appendParagraph(second, "A paragraph from the second client");
    await first.settled();
    await second.goOnline();

    const merged =
      "<paragraph>Edited: Notes from the first client (and the second)</paragraph>" +
      "<paragraph>A paragraph from the second client</paragraph>";
    expect(first.blocks.toString()).toBe(merged);
    expect(second.blocks.toString()).toBe(merged);
    expect(Y.encodeStateVector(second.doc)).toEqual(Y.encodeStateVector(first.doc));

    // A client connecting now gets the same content from the core.
    const third = await connectToMind(core, mind.id);
    expect(third.blocks.toString()).toBe(merged);
  });

  test("after a restart a Mind's content is identical, before and after compaction", async () => {
    const dataDir = await createTempDataFolder();
    const firstRun = startCore(dataDir);
    const mind = await firstRun.createMind({ title: "Restarts" });
    const writer = await connectToMind(firstRun, mind.id);
    appendParagraph(writer, "First paragraph");
    appendParagraph(writer, "Second paragraph");
    appendParagraph(writer, "Third paragraph");
    typeInto(writer, 0, 5, " (edited)");
    writer.blocks.delete(1, 1);
    await writer.settled();
    const expected = writer.blocks.toString();
    const expectedVersion = Y.encodeStateVector(writer.doc);
    firstRun.close();
    expect(storedRows(dataDir, mind.id)).toBe(5);

    // The core loads the separate updates, and then compacts them.
    const secondRun = startCore(dataDir);
    const beforeCompaction = await connectToMind(secondRun, mind.id);
    expect(beforeCompaction.blocks.toString()).toBe(expected);
    expect(Y.encodeStateVector(beforeCompaction.doc)).toEqual(expectedVersion);
    expect(storedRows(dataDir, mind.id)).toBe(1);
    secondRun.close();

    // The core loads the compacted state.
    const thirdRun = startCore(dataDir);
    const afterCompaction = await connectToMind(thirdRun, mind.id);
    expect(afterCompaction.blocks.toString()).toBe(expected);
    expect(Y.encodeStateVector(afterCompaction.doc)).toEqual(expectedVersion);
    expect(expected).toBe(
      "<paragraph>First (edited) paragraph</paragraph><paragraph>Third paragraph</paragraph>",
    );
  });

  test("stored updates are compacted once enough have piled up, and nothing is lost", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const mind = await core.createMind();
    const writer = await connectToMind(core, mind.id);

    appendParagraph(writer, "");
    for (let i = 0; i < COMPACT_AFTER_UPDATES; i++) typeInto(writer, 0, i, "x");
    await writer.settled();

    expect(storedRows(dataDir, mind.id)).toBeLessThan(COMPACT_AFTER_UPDATES / 10);
    core.close();
    const reader = await connectToMind(startCore(dataDir), mind.id);
    expect(reader.blocks.toString()).toBe(
      `<paragraph>${"x".repeat(COMPACT_AFTER_UPDATES)}</paragraph>`,
    );
  });

  test("closing a Mind compacts its stored updates, and it can be edited again", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const mind = await core.createMind();
    const writer = await connectToMind(core, mind.id);
    appendParagraph(writer, "One");
    appendParagraph(writer, "Two");
    await writer.settled();
    expect(storedRows(dataDir, mind.id)).toBe(2);

    await core.closeMind(mind.id);
    expect(storedRows(dataDir, mind.id)).toBe(1);

    appendParagraph(writer, "Three");
    await writer.settled();
    const reader = await connectToMind(core, mind.id);
    expect(reader.blocks.toString()).toBe(
      "<paragraph>One</paragraph><paragraph>Two</paragraph><paragraph>Three</paragraph>",
    );
  });

  test("peeking at a Mind that isn't loaded reads what is stored, and loads and writes nothing", () => {
    const db = openDatabase(":memory:");
    migrate(db);
    const now = () => new Date().toISOString();
    const rows = () =>
      db.get<{ count: number }>("SELECT count(*) AS count FROM mind_updates")?.count;
    const writer = createMindContent(db, now, () => {});
    writer.apply("mind", paragraphUpdate("One"));
    writer.apply("mind", paragraphUpdate("Two"));
    expect(rows()).toBe(2);

    // As after a restart: nothing is loaded.
    const content = createMindContent(db, now, () => {});
    expect(content.peek("mind", (blocks) => blocks.length)).toBe(2);
    expect(rows()).toBe(2); // loading would have compacted them into one
    // A loaded Mind is read as it is in memory.
    content.apply("mind", paragraphUpdate("Three"));
    expect(rows()).toBe(2);
    expect(content.peek("mind", (blocks) => blocks.toString())).toContain("Three");
    db.close();
  });

  test("every stored update is pushed to listeners, and one the Mind already has is ignored", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const mind = await core.createMind();
    const listener = vi.fn();
    core.on("mind.update", listener);
    const update = paragraphUpdate("Hello");

    await core.applyMindUpdate(mind.id, update);
    await core.applyMindUpdate(mind.id, update);

    expect(listener).toHaveBeenCalledExactlyOnceWith({
      mindId: mind.id,
      update: expect.any(Uint8Array),
    });
    expect(storedRows(dataDir, mind.id)).toBe(1);
  });

  test("an update that isn't a Yjs update is rejected", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const mind = await core.createMind();
    const listener = vi.fn();
    core.on("mind.update", listener);

    for (const bad of [new Uint8Array([1, 2, 3]), new Uint8Array(), "text", [0, 0], null]) {
      await expect(core.applyMindUpdate(mind.id, bad as never)).rejects.toThrow(InvalidInputError);
    }

    expect(listener).not.toHaveBeenCalled();
    expect(storedRows(dataDir, mind.id)).toBe(0);
  });

  test("a Mind that doesn't exist can't be opened or edited", async () => {
    const core = startCore(await createTempDataFolder());
    const missing = randomUUID();

    await expect(core.openMind(missing)).rejects.toThrow(NotFoundError);
    await expect(core.applyMindUpdate(missing, paragraphUpdate("Hi"))).rejects.toThrow(
      NotFoundError,
    );
    await expect(core.openMind(42 as never)).rejects.toThrow(InvalidInputError);
    await expect(core.closeMind(missing)).resolves.toBeUndefined();
  });
});

/** Makes storing Mind content fail, as a full disk would, until the returned function is called. */
function failWrites(dataDir: string): () => void {
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  db.exec(
    "CREATE TRIGGER fail_writes BEFORE INSERT ON mind_updates BEGIN SELECT RAISE(ABORT, 'disk full'); END",
  );
  return () => {
    db.exec("DROP TRIGGER fail_writes");
    db.close();
  };
}

/** A copy of a Mind holding one paragraph, "BASE", stored, that sends each change as the editor does. */
async function typingIn(core: Core) {
  const mind = await core.createMind({ title: "Notes" });
  const doc = new Y.Doc();
  Y.applyUpdate(doc, (await core.openMind(mind.id)).state, core);
  const sent: Promise<"stored" | "refused">[] = [];
  doc.on("update", (update: Uint8Array, origin: unknown) => {
    if (origin === core) return;
    sent.push(
      core.applyMindUpdate(mind.id, update).then(
        () => "stored" as const,
        () => "refused" as const,
      ),
    );
  });
  const paragraph = new Y.XmlElement("paragraph");
  const text = new Y.XmlText("BASE");
  paragraph.insert(0, [text]);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph]);
  await Promise.all(sent);
  return {
    mindId: mind.id,
    doc,
    /** Types at the end of the paragraph, and says whether the core stored it. */
    type: (words: string) => {
      text.insert(text.length, words);
      const outcome = sent.at(-1);
      if (!outcome) throw new Error("Nothing was sent.");
      return outcome;
    },
  };
}

/** What a Mind holds once the app has quit and started again. */
async function afterRestart(dataDir: string, mindId: string): Promise<string> {
  const core = startCore(dataDir);
  const doc = new Y.Doc();
  Y.applyUpdate(doc, (await core.openMind(mindId)).state);
  return doc.getXmlFragment(MIND_CONTENT_FIELD).toString();
}

describe("A change that can't be stored (#1119)", () => {
  // protects=after one write fails, the User's later typing is stored, and the failed change with it, so all of it is there after a restart; fails_when=later writes resolve but vanish on reopen; no_claim=a real full disk or power loss.
  test("typing after a failed write is kept, and the failed change with it", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const restore = failWrites(dataDir);

    expect(await copy.type(" LOST")).toBe("refused");
    restore();
    expect(await copy.type(" AFTER")).toBe("stored");
    expect(await copy.type(" MORE")).toBe("stored");
    core.close();

    expect(await afterRestart(dataDir, copy.mindId)).toBe(
      "<paragraph>BASE LOST AFTER MORE</paragraph>",
    );
  });

  // protects=quitting once storage works again stores the change that failed, though nothing was typed since; fails_when=it is lost at quit; no_claim=none beyond protects.
  test("quitting once storage works again stores the change that failed", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const restore = failWrites(dataDir);
    expect(await copy.type(" LOST")).toBe("refused");
    restore();

    core.close();

    expect(await afterRestart(dataDir, copy.mindId)).toBe("<paragraph>BASE LOST</paragraph>");
  });

  // protects=quitting while storage still fails quits cleanly, says in the log which Mind lost changes, and what was stored before opens and takes typing again; fails_when=quitting throws, the stored text is damaged, or the loss goes unlogged; no_claim=that the unstored change survives, which nothing can store.
  test("quitting while storage still fails keeps what was stored, and says what was lost", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const restore = failWrites(dataDir);
    expect(await copy.type(" LOST")).toBe("refused");
    const logged = vi.spyOn(console, "error").mockImplementation(() => undefined);

    expect(() => core.close()).not.toThrow();
    restore();

    expect(logged.mock.calls.flat().join(" ")).toContain(copy.mindId);
    logged.mockRestore();
    expect(await afterRestart(dataDir, copy.mindId)).toBe("<paragraph>BASE</paragraph>");
  });

  // protects=a change that depends on one the core never got is refused, never acknowledged as stored, and the copy's whole state then stores all of it; fails_when=the dependent change resolves but isn't stored; no_claim=none beyond protects.
  test("a change after one the core never got is refused, and the whole copy then stores both", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const missed: Uint8Array[] = [];
    const before = Y.encodeStateVector(copy.doc);
    copy.doc.once("update", (update: Uint8Array) => missed.push(update));
    copy.doc.transact(() => {
      const text = copy.doc.getXmlFragment(MIND_CONTENT_FIELD).get(0);
      if (!(text instanceof Y.XmlElement)) throw new Error("No paragraph.");
      const words = text.get(0);
      if (words instanceof Y.XmlText) words.insert(words.length, " MISSED");
    }, core);
    expect(missed).toHaveLength(1);
    expect(Y.encodeStateVector(copy.doc)).not.toEqual(before);

    expect(await copy.type(" NEXT")).toBe("refused");
    await core.applyMindUpdate(copy.mindId, Y.encodeStateAsUpdate(copy.doc));
    core.close();

    expect(await afterRestart(dataDir, copy.mindId)).toBe(
      "<paragraph>BASE MISSED NEXT</paragraph>",
    );
  });
});
