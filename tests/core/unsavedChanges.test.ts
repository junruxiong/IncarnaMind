import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test, vi } from "vitest";
import * as Y from "yjs";
import { type Core, DATABASE_FILE, MIND_CONTENT_FIELD, type UnsavedChanges } from "../../src/core";
import { openDatabase } from "../../src/core/storage";
import { saveFailureReason } from "../../src/core/unsavedChanges";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { appendParagraph, connectToMind } from "../helpers/mindClient";
import { note } from "../helpers/minds";

/**
 * Changes storage refuses (#1119), through the core's public interface as a
 * window uses it: they stay in memory, reach every window at once, are said
 * with a reason, are stored with the next change, on a backoff and on closing,
 * and a crash leaves a count for the next start. Storage fails through a
 * SQLite trigger on synthetic data, not a real full disk.
 */

afterEach(() => {
  vi.useRealTimers();
});

/** Makes storing Mind content fail as a full disk does, until the returned function is called. */
function failWrites(dataDir: string): () => void {
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  db.exec(
    "CREATE TRIGGER fail_writes BEFORE INSERT ON mind_updates BEGIN SELECT RAISE(ABORT, 'database or disk is full'); END",
  );
  return () => {
    db.exec("DROP TRIGGER fail_writes");
    db.close();
  };
}

/** A copy of a new Mind holding "BASE", stored, that sends each change as the editor does. */
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
  core.on("mind.update", (event) => {
    if (event.mindId === mind.id) Y.applyUpdate(doc, event.update, core);
  });
  const paragraph = new Y.XmlElement("paragraph");
  const text = new Y.XmlText("BASE");
  paragraph.insert(0, [text]);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph]);
  await Promise.all(sent);
  return {
    mindId: mind.id,
    doc,
    /** Types at the end of the paragraph, and says whether the core took it. */
    type: (words: string) => {
      text.insert(text.length, words);
      const outcome = sent.at(-1);
      if (!outcome) throw new Error("Nothing was sent.");
      return outcome;
    },
  };
}

const textOf = (state: Uint8Array) => {
  const doc = new Y.Doc();
  Y.applyUpdate(doc, state);
  return doc.getXmlFragment(MIND_CONTENT_FIELD).toString();
};

/** What a Mind holds once the app has quit and started again. */
async function afterRestart(dataDir: string, mindId: string): Promise<string> {
  return textOf((await startCore(dataDir).openMind(mindId)).state);
}

const storedRows = (dataDir: string, mindId: string) =>
  queryDatabase<{ count: number }>(
    dataDir,
    "SELECT count(*) AS count FROM mind_updates WHERE mind_id = ? AND deleted_at IS NULL",
    [mindId],
  )[0]?.count ?? 0;

describe("Changes storage refuses (#1119)", () => {
  // protects=after one write fails, the User's later typing is stored, and the refused change with it, there after a restart, and nothing tells the window a change failed; fails_when=later typing vanishes on reopen, or the window is told to send again; no_claim=a real full disk.
  test("typing after a refused write is stored with it, and all of it is there after a restart", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const restore = failWrites(dataDir);

    expect(await copy.type(" LOST")).toBe("stored");
    restore();
    expect(await copy.type(" AFTER")).toBe("stored");
    core.close();

    expect(await afterRestart(dataDir, copy.mindId)).toBe("<paragraph>BASE LOST AFTER</paragraph>");
  });

  // protects=the core's own edit while storage fails (Add to note) applies, reaches the windows at once, doesn't throw, so its caller never tries again, and is stored once storage is back: one paragraph, not two; fails_when=Add to note fails and, tried again, leaves two paragraphs, or the windows don't see it; no_claim=none beyond protects.
  test("Add to note while storage fails applies once, reaches the windows, and is stored later", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const mind = await core.createMind({ title: "Notes" });
    const client = await connectToMind(core, mind.id);
    const restore = failWrites(dataDir);

    await core.addToNote({ mindId: mind.id, blocks: [note("ADDED")], origin: null, model: null });

    expect(client.blocks.toString()).toContain("ADDED");
    restore();
    await core.saveChangesAgain();
    core.close();
    const after = await afterRestart(dataDir, mind.id);
    expect(after.match(/ADDED/g)).toHaveLength(1);
  });

  // protects=while changes wait, the core says how many, in which Mind, why in plain words (a full disk) and the database's own words for Details, as an event and when asked; and that nothing waits once they are stored; fails_when=the User isn't told, or the reason is SQLite's text; no_claim=the notice's look, which the e2e checks.
  test("what waits is said with its reason, and nothing once it is stored", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const heard: (UnsavedChanges | null)[] = [];
    core.on("minds.unsaved", (state) => heard.push(state));
    const restore = failWrites(dataDir);

    await copy.type(" ONE");
    await copy.type(" TWO");

    expect(await core.unsavedChanges()).toEqual({
      count: 2,
      mindIds: [copy.mindId],
      reason: "full",
      detail: expect.stringContaining("database or disk is full"),
    });
    restore();
    expect(await core.saveChangesAgain()).toBeNull();
    expect(heard.at(-1)).toBeNull();
    expect(heard[0]).toMatchObject({ count: 1, reason: "full" });
  });

  // protects=storage's refusals read as plain reasons by SQLite's own codes (full, I/O, read-only, busy) and by their words; fails_when=a full disk reads as an unknown failure; no_claim=none beyond protects.
  test.each([
    [{ errcode: 13, message: "database or disk is full" }, "full"],
    [{ errcode: 266, message: "disk I/O error" }, "io"],
    [{ errcode: 8, message: "attempt to write a readonly database" }, "readOnly"],
    [{ errcode: 5, message: "database is locked" }, "busy"],
    [{ errcode: 19, message: "database or disk is full" }, "full"],
    [{ errcode: 19, message: "constraint failed" }, "other"],
  ] as const)("%o reads as %s", (failure, reason) => {
    expect(saveFailureReason(Object.assign(new Error(failure.message), failure))).toBe(reason);
  });

  // protects=changes that wait are stored as one merged row, not once per try, and on their own once storage is back, on a backoff, with nothing typed; fails_when=each try stores the queue again, or nothing is stored until the next change; no_claim=the exact delays.
  test("what waits is stored once storage is back, as one row, with nothing typed", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const before = storedRows(dataDir, copy.mindId);
    const restore = failWrites(dataDir);
    for (const word of [" A", " B", " C"]) await copy.type(word);
    restore();

    await vi.advanceTimersByTimeAsync(60_000);

    expect(await core.unsavedChanges()).toBeNull();
    expect(storedRows(dataDir, copy.mindId)).toBe(before + 1);
    core.close();
    expect(await afterRestart(dataDir, copy.mindId)).toBe("<paragraph>BASE A B C</paragraph>");
  });

  // protects=closing a Mind stores what waits first, though its rows were compacted to one; fails_when=closing drops it from memory and it is lost; no_claim=none beyond protects.
  test("closing a Mind stores what waits first", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    await core.closeMind(copy.mindId);
    const restore = failWrites(dataDir);
    await copy.type(" LOST");
    restore();

    await core.closeMind(copy.mindId);

    expect(await core.unsavedChanges()).toBeNull();
    const reader = new Y.Doc();
    Y.applyUpdate(reader, (await startCore(dataDir).openMind(copy.mindId)).state);
    expect(reader.getXmlFragment(MIND_CONTENT_FIELD).toString()).toBe(
      "<paragraph>BASE LOST</paragraph>",
    );
  });

  // protects=a change that builds on one the core never got is refused, the whole copy then stores it, and a later change of another window's isn't refused for it; fails_when=the dependent change is acknowledged and lost, or every later change says it wasn't saved; no_claim=none beyond protects.
  test("a change after one the core never got is refused, alone", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    copy.doc.transact(() => {
      const paragraph = copy.doc.getXmlFragment(MIND_CONTENT_FIELD).get(0);
      const words = paragraph instanceof Y.XmlElement ? paragraph.get(0) : null;
      if (words instanceof Y.XmlText) words.insert(words.length, " MISSED");
    }, core);

    expect(await copy.type(" NEXT")).toBe("refused");
    const other = await connectToMind(core, copy.mindId);
    appendParagraph(other, "ELSEWHERE");
    await expect(other.settled()).resolves.toBeUndefined();
    await core.applyMindUpdate(copy.mindId, Y.encodeStateAsUpdate(copy.doc));
    core.close();

    expect(await afterRestart(dataDir, copy.mindId)).toBe(
      "<paragraph>BASE MISSED NEXT</paragraph><paragraph>ELSEWHERE</paragraph>",
    );
  });

  // protects=quitting tries once more: storage back, what waits is stored though nothing was typed since; fails_when=it is lost at quit; no_claim=none beyond protects.
  test("quitting once storage works again stores what waits", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const restore = failWrites(dataDir);
    await copy.type(" LOST");
    restore();

    core.close();

    expect(await afterRestart(dataDir, copy.mindId)).toBe("<paragraph>BASE LOST</paragraph>");
  });

  // protects=a run that ends without quitting while changes wait (a crash) leaves their count for the next start, said once; a run that quits leaves none, as the quit sheet told the User; fails_when=the next start says nothing of a crash's loss, says it twice, or speaks after a quit; no_claim=what the count file survives.
  test("a crash leaves the count of changes lost for the next start, once; quitting leaves none", async () => {
    const dataDir = await createTempDataFolder();
    const tempDir = await mkdtemp(join(tmpdir(), "incarnamind-temp-"));
    const paths = {
      dataDir,
      home: join(dataDir, "home"),
      documents: join(dataDir, "Documents"),
      tempDir,
    };
    const crashed = startCore(dataDir, { paths });
    const copy = await typingIn(crashed);
    const restore = failWrites(dataDir);
    await copy.type(" ONE");
    await copy.type(" TWO");

    const next = startCore(dataDir, { paths });

    expect(await next.takeLostChanges()).toBe(2);
    expect(await next.takeLostChanges()).toBeNull();
    crashed.close();
    restore();
    expect(await startCore(dataDir, { paths }).takeLostChanges()).toBeNull();
  });

  // protects=quitting while storage still fails quits cleanly, keeps what was stored before, and says in the log which Mind lost how many changes; fails_when=quitting throws, the stored text is damaged, or the loss goes unlogged; no_claim=that the refused change survives, which nothing can store.
  test("quitting while storage still fails keeps what was stored, and logs what was lost", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir);
    const copy = await typingIn(core);
    const restore = failWrites(dataDir);
    await copy.type(" LOST");
    const logged = vi.spyOn(console, "error").mockImplementation(() => undefined);

    expect(() => core.close()).not.toThrow();
    restore();

    expect(logged.mock.calls.flat().join(" ")).toContain(copy.mindId);
    logged.mockRestore();
    expect(await afterRestart(dataDir, copy.mindId)).toBe("<paragraph>BASE</paragraph>");
  });
});
