import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import {
  type ArtifactReference,
  blockReference,
  CONTENT_SCHEMA_VERSION,
  CONTENT_SCHEMA_VERSION_KEY,
  DATABASE_FILE,
  InvalidInputError,
  MIND_CONTENT_FIELD,
  MIND_SETTINGS_FIELD,
  MindReadOnlyError,
  parseReference,
} from "../../src/core";
import { migrate, migrations, openDatabase } from "../../src/core/storage";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { appendParagraph, connectToMind } from "../helpers/mindClient";

/** Runs `change` on the data folder's database while no core has it open. */
function withDatabase(dataDir: string, change: (db: ReturnType<typeof openDatabase>) => void) {
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    change(db);
  } finally {
    db.close();
  }
}

/** Replaces what is stored for the Mind with one Yjs document, as another computer's update would leave it. */
function storeContent(dataDir: string, mindId: string, doc: Y.Doc) {
  withDatabase(dataDir, (db) => {
    db.run("DELETE FROM mind_updates WHERE mind_id = ?", [mindId]);
    db.run(
      "INSERT INTO mind_updates (id, mind_id, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
      [
        randomUUID(),
        mindId,
        Y.encodeStateAsUpdate(doc),
        "2026-10-06T09:00:00Z",
        "2026-10-06T09:00:00Z",
      ],
    );
  });
}

/** A document a newer version wrote: a node type this version doesn't know, and a newer schema version. */
function newerDocument(version = CONTENT_SCHEMA_VERSION + 1): Y.Doc {
  const doc = new Y.Doc();
  const paragraph = new Y.XmlElement("paragraph");
  paragraph.insert(0, [new Y.XmlText("Written on another computer")]);
  const future = new Y.XmlElement("futureWidget");
  future.setAttribute("layout", "grid");
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph, future]);
  doc.getMap(MIND_SETTINGS_FIELD).set(CONTENT_SCHEMA_VERSION_KEY, version);
  return doc;
}

const storedRows = (dataDir: string, mindId: string) =>
  queryDatabase<{ id: string; data: Uint8Array; updated_at: string }>(
    dataDir,
    "SELECT id, data, updated_at FROM mind_updates WHERE mind_id = ? ORDER BY rowid",
    [mindId],
  ).map((row) => ({ ...row, data: Array.from(row.data) }));

describe("the kind of a Mind", () => {
  test("a Mind is of kind mind unless made as another kind", async () => {
    const core = startCore(await createTempDataFolder());

    const plain = await core.createMind({ title: "Notes" });
    const chat = await core.createMind({ title: "Ask", kind: "chat" });

    expect(plain.kind).toBe("mind");
    expect(chat.kind).toBe("chat");
    expect((await core.listMinds()).map((mind) => mind.kind).sort()).toEqual(["chat", "mind"]);
    await expect(core.openMind(chat.id)).resolves.toMatchObject({ access: "edit" });
  });

  test("creating a Mind of a kind this version doesn't know is refused", async () => {
    const core = startCore(await createTempDataFolder());

    await expect(core.createMind({ kind: "sheet" as never })).rejects.toThrow(InvalidInputError);
  });

  test("the migration gives every existing Mind the kind mind", async () => {
    const dataDir = await createTempDataFolder();
    const before = migrations.filter((migration) => migration.version < 30);
    withDatabase(dataDir, (db) => {
      migrate(db, before);
      db.run("INSERT INTO minds (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)", [
        "old-mind",
        "From an older version",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
      ]);
    });

    const core = startCore(dataDir);

    expect(await core.listMinds()).toEqual([
      expect.objectContaining({ id: "old-mind", title: "From an older version", kind: "mind" }),
    ]);
  });

  test("a Mind of a kind this version doesn't know is listed but opens as update to open", async () => {
    const dataDir = await createTempDataFolder();
    const first = startCore(dataDir);
    const mind = await first.createMind({ title: "Made by a newer version" });
    await first.close();
    withDatabase(dataDir, (db) => db.run("UPDATE minds SET kind = 'deck' WHERE id = ?", [mind.id]));
    const rowsBefore = storedRows(dataDir, mind.id);

    const core = startCore(dataDir);

    expect(await core.listMinds()).toEqual([
      expect.objectContaining({ id: mind.id, kind: "deck" }),
    ]);
    const opened = await core.openMind(mind.id);
    expect(opened.access).toBe("update-required");
    expect(opened.mind.kind).toBe("deck");
    await expect(core.applyMindUpdate(mind.id, new Uint8Array([0, 0]))).rejects.toThrow(
      MindReadOnlyError,
    );
    expect(storedRows(dataDir, mind.id)).toEqual(rowsBefore);
  });
});

describe("the content schema version of a Mind", () => {
  test("a Mind without a recorded version reads as version 1 and opens normally", async () => {
    const dataDir = await createTempDataFolder();
    const first = startCore(dataDir);
    const mind = await first.createMind();
    await first.close();
    const old = new Y.Doc();
    const paragraph = new Y.XmlElement("paragraph");
    paragraph.insert(0, [new Y.XmlText("Written before versions were recorded")]);
    old.getXmlFragment(MIND_CONTENT_FIELD).push([paragraph]);
    storeContent(dataDir, mind.id, old);

    const core = startCore(dataDir);
    const opened = await core.openMind(mind.id);

    expect(opened.access).toBe("edit");
    const doc = new Y.Doc();
    Y.applyUpdate(doc, opened.state);
    expect(doc.getXmlFragment(MIND_CONTENT_FIELD).toString()).toContain("before versions");
    expect(doc.getMap(MIND_SETTINGS_FIELD).has(CONTENT_SCHEMA_VERSION_KEY)).toBe(false);

    // Opening records nothing; the first write does.
    const client = await connectToMind(core, mind.id);
    appendParagraph(client, "More");
    await client.settled();
    const after = new Y.Doc();
    Y.applyUpdate(after, (await core.openMind(mind.id)).state);
    expect(after.getMap(MIND_SETTINGS_FIELD).get(CONTENT_SCHEMA_VERSION_KEY)).toBe(
      CONTENT_SCHEMA_VERSION,
    );
  });

  test("a Mind written by a newer version opens read-only and its Yjs state is unchanged", async () => {
    const dataDir = await createTempDataFolder();
    const first = startCore(dataDir);
    const mind = await first.createMind({ title: "From the other computer" });
    await first.close();
    const newer = newerDocument();
    storeContent(dataDir, mind.id, newer);
    const rowsBefore = storedRows(dataDir, mind.id);
    const stateBefore = Array.from(Y.encodeStateAsUpdate(newer));

    const core = startCore(dataDir);
    const opened = await core.openMind(mind.id);

    expect(opened.access).toBe("read-only");
    expect(Array.from(opened.state)).toEqual(stateBefore);

    // The editor's Yjs binding would delete the node type it doesn't know: that change is refused.
    const local = new Y.Doc();
    Y.applyUpdate(local, opened.state);
    const sent: Uint8Array[] = [];
    local.on("update", (update: Uint8Array) => sent.push(update));
    local.getXmlFragment(MIND_CONTENT_FIELD).delete(1, 1);
    expect(sent).toHaveLength(1);
    await expect(core.applyMindUpdate(mind.id, sent[0] as Uint8Array)).rejects.toThrow(
      MindReadOnlyError,
    );

    await core.closeMind(mind.id);
    await core.close();
    expect(storedRows(dataDir, mind.id)).toEqual(rowsBefore);

    const reopened = await startCore(dataDir).openMind(mind.id);
    expect(Array.from(reopened.state)).toEqual(stateBefore);
  });

  test("a write that records a newer version keeps it, and the Mind is read-only after", async () => {
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind();

    await core.applyMindUpdate(mind.id, Y.encodeStateAsUpdate(newerDocument(7)));

    const opened = await core.openMind(mind.id);
    expect(opened.access).toBe("read-only");
    const doc = new Y.Doc();
    Y.applyUpdate(doc, opened.state);
    expect(doc.getMap(MIND_SETTINGS_FIELD).get(CONTENT_SCHEMA_VERSION_KEY)).toBe(7);
    await expect(
      core.applyMindUpdate(mind.id, Y.encodeStateAsUpdate(newerDocument(7))),
    ).rejects.toThrow(MindReadOnlyError);
  });

  test("a Mind written with the newest version this one understands opens normally", async () => {
    const dataDir = await createTempDataFolder();
    const first = startCore(dataDir);
    const mind = await first.createMind();
    await first.close();
    storeContent(dataDir, mind.id, newerDocument(CONTENT_SCHEMA_VERSION));

    const opened = await startCore(dataDir).openMind(mind.id);

    expect(opened.access).toBe("edit");
  });

  test("a malformed version reads as version 1", async () => {
    const dataDir = await createTempDataFolder();
    const first = startCore(dataDir);
    const mind = await first.createMind();
    await first.close();
    for (const bad of ["9", -3, 2.5, null]) {
      storeContent(dataDir, mind.id, newerDocument(bad as never));
      const opened = await startCore(dataDir).openMind(mind.id);
      expect(opened.access).toBe("edit");
    }
  });
});

describe("a reference from one artifact to another", () => {
  test("is the artifact's ID and an anchor, a block anchor holding the Block's ID", () => {
    const reference: ArtifactReference = blockReference("chat-1", "block-9");

    expect(reference).toEqual({
      artifactId: "chat-1",
      anchor: { kind: "block", blockId: "block-9" },
    });
  });

  test("survives being stored as JSON", () => {
    const reference = blockReference("chat-1", "block-9");

    expect(parseReference(JSON.parse(JSON.stringify(reference)))).toEqual(reference);
  });

  test("is not read from anything else", () => {
    expect(parseReference(null)).toBeNull();
    expect(parseReference({ artifactId: "a" })).toBeNull();
    expect(parseReference({ artifactId: "", anchor: { kind: "block", blockId: "b" } })).toBeNull();
    expect(parseReference({ artifactId: "a", anchor: { kind: "block", blockId: "" } })).toBeNull();
    // A position is not an anchor.
    expect(parseReference({ artifactId: "a", anchor: { kind: "position", offset: 4 } })).toBeNull();
  });
});
