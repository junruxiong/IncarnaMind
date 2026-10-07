import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { DATABASE_FILE, type Document } from "../../src/core";
import { PROCESSING_VERSION } from "../../src/core/documents/processing";
import { createFakeEmbedder } from "../../src/core/embedding/fake";
import { migrate, openDatabase } from "../../src/core/storage";
import { migrations } from "../../src/core/storage/migrations";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { sha256, storedFile, waitForProcessing } from "../helpers/documents";
import { createControlledEmbedder } from "../helpers/embedding";

const NOTES = `# Transformers

The Transformer architecture relies entirely on self-attention to draw global
dependencies between input and output.`;

const OLD_ID = "6f1c1c55-8a0e-4f43-9b5e-1d1b7c8b6a01";

/**
 * A data folder as #25 left it: the schema before migration 10, and one ready
 * Document with a Passage built by the old pipeline (trigram index, no vector).
 */
async function oldDataFolder(): Promise<string> {
  const dataDir = await createTempDataFolder();
  await mkdir(join(dataDir, "documents"), { recursive: true });
  await writeFile(storedFile(dataDir, sha256(NOTES)), NOTES);
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    migrate(
      db,
      migrations.filter((migration) => migration.version < 10),
    );
    const at = "2026-10-01T09:00:00.000Z";
    db.run(
      `INSERT INTO documents (id, content_hash, name, kind, size, status, page_count, created_at, updated_at)
       VALUES (?, ?, 'Old notes', 'markdown', ?, 'ready', NULL, ?, ?)`,
      [OLD_ID, sha256(NOTES), Buffer.byteLength(NOTES), at, at],
    );
    db.run(
      `INSERT INTO passages (id, document_id, position, page_from, page_to, window_from, window_to,
         text, created_at, updated_at)
       VALUES ('old-passage', ?, 0, NULL, NULL, 0, 0, ?, ?, ?)`,
      [OLD_ID, `Old notes\n${NOTES}`, at, at],
    );
  } finally {
    db.close();
  }
  return dataDir;
}

describe("Re-processing", { timeout: 30_000 }, () => {
  test("Documents processed by an older pipeline are processed again at startup", async () => {
    const dataDir = await oldDataFolder();
    const statuses: Document["status"][] = [];

    const core = startCore(dataDir);
    core.on("document.status", (document) => statuses.push(document.status));
    // Queued again at startup, before anything is listening.
    expect((await core.listDocuments())[0]).toMatchObject({ id: OLD_ID, status: "queued" });
    const [document] = await waitForProcessing(core, [OLD_ID]);

    expect(document).toMatchObject({ name: "Old notes", status: "ready", progress: null });
    expect(statuses).toEqual(["extracting", "embedding", "ready"]);
    // The new Passages are found by keyword and by vector; the old one is gone.
    const keyword = await core.searchPassages("self-attention", { mode: "keyword" });
    const vector = await core.searchPassages("self-attention", { mode: "vector" });
    expect(keyword.map((result) => result.documentId)).toEqual([OLD_ID]);
    expect(vector.map((result) => result.passageId)).toEqual(
      keyword.map((result) => result.passageId),
    );
    expect(keyword[0]?.passageId).not.toBe("old-passage");
    expect(keyword[0]?.text.startsWith("# Transformers")).toBe(true);
    const rows = queryDatabase<{ id: string; deleted: number; embedded: number }>(
      dataDir,
      `SELECT id, deleted_at IS NOT NULL AS deleted, embedding IS NOT NULL AS embedded
       FROM passages ORDER BY seq`,
    );
    expect(rows).toEqual([
      { id: "old-passage", deleted: 1, embedded: 0 },
      { id: keyword[0]?.passageId, deleted: 0, embedded: 1 },
    ]);
    expect(
      queryDatabase(dataDir, "SELECT processing_version, embedding_model FROM documents"),
    ).toEqual([
      { processing_version: PROCESSING_VERSION, embedding_model: "multilingual-e5-small-int8" },
    ]);
  });

  test("Documents already processed by this pipeline are left alone at startup", async () => {
    const dataDir = await oldDataFolder();
    const first = startCore(dataDir);
    await waitForProcessing(first, [OLD_ID]);
    const passages = await first.searchPassages("Transformer", { mode: "keyword" });
    first.close();

    const second = startCore(dataDir);
    const statuses: Document["status"][] = [];
    second.on("document.status", (document) => statuses.push(document.status));

    expect((await second.listDocuments())[0]?.status).toBe("ready");
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(statuses).toEqual([]);
    expect(await second.searchPassages("Transformer", { mode: "keyword" })).toEqual(passages);
  });

  test("embedding interrupted by quitting carries on after a restart, keeping the vectors it made", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const long = Array.from(
      { length: 300 },
      (_, index) => `Sentence ${index} is about attention and transformers.`,
    ).join(" ");
    const path = join(sources, "long.md");
    await writeFile(path, long);
    // An embedder that embeds two Passages, then hangs, as if the app quit while it worked.
    const fake = createFakeEmbedder();
    let calls = 0;
    let hung = () => {};
    const hanging = new Promise<void>((resolve) => {
      hung = resolve;
    });
    const before = startCore(dataDir, {
      embedder: {
        load: (files) => fake.load(files),
        embed: async (text) => {
          if (++calls > 2) {
            hung();
            await new Promise(() => {});
          }
          return fake.embed(text);
        },
        close: () => fake.close(),
      },
    });
    const [added] = (await before.addDocuments([path])).documents;
    if (!added) throw new Error("Nothing was added.");
    await hanging;
    before.close();
    const counts = () =>
      queryDatabase<{ total: number; embedded: number }>(
        dataDir,
        "SELECT count(*) AS total, count(embedding) AS embedded FROM passages WHERE deleted_at IS NULL",
      )[0];
    const total = counts()?.total ?? 0;
    expect(total).toBeGreaterThan(3);
    expect(counts()?.embedded).toBe(2);

    const embedder = createControlledEmbedder();
    const after = startCore(dataDir, { embedder });
    const listed = (await after.listDocuments())[0];
    expect(listed).toMatchObject({ status: "embedding", progress: 2 / total });
    const [ready] = await waitForProcessing(after, [added.id]);

    expect(ready?.status).toBe("ready");
    expect(counts()).toEqual({ total, embedded: total });
    expect(embedder.texts).toHaveLength(total - 2);
  });
});
