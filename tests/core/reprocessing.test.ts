import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { DATABASE_FILE, type Document } from "../../src/core";
import { PROCESSING_VERSION } from "../../src/core/documents/processing";
import { createFakeEmbedder } from "../../src/core/embedding/fake";
import { migrate, openDatabase } from "../../src/core/storage";
import { migrations } from "../../src/core/storage/migrations";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import {
  addAndProcess,
  createSourceFolder,
  sha256,
  storedFile,
  unfoldedKeywords,
  waitForDocuments,
  waitForProcessing,
  writeSourceFile,
} from "../helpers/documents";
import { createControlledEmbedder, turnOnEmbeddings } from "../helpers/embedding";
import { docxOf } from "../helpers/office";

const NOTES = `# Transformers

The Transformer architecture relies entirely on self-attention to draw global
dependencies between input and output.`;

/** In traditional characters. */
const GOALS = `# 永續發展

可持續發展目標是聯合國制定的十七個全球發展目標，以綜合方式解決社會和環境的發展問題。`;

const OLD_ID = "6f1c1c55-8a0e-4f43-9b5e-1d1b7c8b6a01";

/**
 * Takes a data folder back to what processing version 5 left: every Document
 * marked as of version 5, and the keyword index holding each Passage's words
 * and its Document's name as written, traditional characters and all.
 */
function asVersion5(dataDir: string): void {
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    db.transaction(() => {
      db.run("UPDATE documents SET processing_version = 5");
      const passages = db.all<{ seq: number; text: string; name: string }>(
        `SELECT p.seq, p.text, d.name FROM passages p JOIN documents d ON d.id = p.document_id
         WHERE p.deleted_at IS NULL`,
      );
      for (const { seq, text, name } of passages) {
        db.run("DELETE FROM passages_fts WHERE rowid = ?", [BigInt(seq)]);
        db.run("INSERT INTO passages_fts (rowid, text) VALUES (?, ?)", [
          BigInt(seq),
          `${unfoldedKeywords(name)} ${unfoldedKeywords(text)}`,
        ]);
      }
    });
  } finally {
    db.close();
  }
}

/** A Word comment, which processing version 6 didn't read. */
const FINANCE = "Finance has asked to lower the contingency to 8 per cent.";

/**
 * Takes a data folder back to what processing version 6 left: every Document
 * marked as of version 6, and the Word Document's Passages and keyword index
 * without the comment it didn't read.
 */
function asVersion6(dataDir: string, wordId: string): void {
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    db.transaction(() => {
      db.run("UPDATE documents SET processing_version = 6");
      const passages = db.all<{ seq: number; text: string; name: string }>(
        `SELECT p.seq, p.text, d.name FROM passages p JOIN documents d ON d.id = p.document_id
         WHERE p.deleted_at IS NULL AND p.document_id = ?`,
        [wordId],
      );
      for (const { seq, text, name } of passages) {
        const read = text.replace(`\n\n${FINANCE}`, "");
        db.run("UPDATE passages SET text = ? WHERE seq = ?", [read, BigInt(seq)]);
        db.run("DELETE FROM passages_fts WHERE rowid = ?", [BigInt(seq)]);
        db.run("INSERT INTO passages_fts (rowid, text) VALUES (?, ?)", [
          BigInt(seq),
          `${unfoldedKeywords(name)} ${unfoldedKeywords(read)}`,
        ]);
      }
    });
  } finally {
    db.close();
  }
}

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

    // An install from before had the built-in model on by default: embeddings are off now.
    expect(document).toMatchObject({ name: "Old notes", status: "ready", progress: null });
    expect(statuses).toEqual(["extracting", "ready"]);
    // The new Passages are found by their words; the old one is gone.
    const keyword = await core.searchPassages("self-attention", { mode: "keyword" });
    expect(keyword.map((result) => result.documentId)).toEqual([OLD_ID]);
    expect(keyword[0]?.passageId).not.toBe("old-passage");
    expect(keyword[0]?.text.startsWith("# Transformers")).toBe(true);
    const rows = queryDatabase<{ id: string; deleted: number; embedded: number }>(
      dataDir,
      `SELECT id, deleted_at IS NOT NULL AS deleted, embedding IS NOT NULL AS embedded
       FROM passages ORDER BY seq`,
    );
    expect(rows).toEqual([
      { id: "old-passage", deleted: 1, embedded: 0 },
      { id: keyword[0]?.passageId, deleted: 0, embedded: 0 },
    ]);
    expect(
      queryDatabase(dataDir, "SELECT processing_version, embedding_model FROM documents"),
    ).toEqual([{ processing_version: PROCESSING_VERSION, embedding_model: null }]);

    // Turned on, the new Passages are embedded, and vector search finds them too.
    await core.saveEmbeddingProvider({ kind: "built-in" });
    await waitForDocuments(core, (documents) => {
      expect(documents[0]?.status).toBe("ready");
      expect(statuses).toEqual(["extracting", "ready", "embedding", "ready"]);
    });
    const vector = await core.searchPassages("self-attention", { mode: "vector" });
    expect(vector.map((result) => result.passageId)).toEqual(
      keyword.map((result) => result.passageId),
    );
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

  test("for the keyword index's folding, only Documents with Han characters in their text or name are processed again", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createSourceFolder();
    const embedder = createControlledEmbedder();
    const first = startCore(dataDir, { embedder });
    await turnOnEmbeddings(first);
    const [goals, report, notes] = (await addAndProcess(first, [
      await writeSourceFile(sources, "目標.md", GOALS),
      await writeSourceFile(sources, "季度報告.md", "Quarterly figures rose by four per cent."),
      await writeSourceFile(sources, "Transformers.md", NOTES),
    ])) as [Document, Document, Document];
    first.close();
    asVersion5(dataDir);
    const passages = (documentId: string) =>
      queryDatabase<{ id: string; text: string; embedding: Uint8Array | null }>(
        dataDir,
        `SELECT id, text, embedding FROM passages
         WHERE document_id = ? AND deleted_at IS NULL ORDER BY position`,
        [documentId],
      );
    const before = { goals: passages(goals.id), notes: passages(notes.id) };
    const matching = (word: string) =>
      queryDatabase(dataDir, "SELECT rowid FROM passages_fts WHERE passages_fts MATCH ?", [
        `"${word}"`,
      ]).length;
    // As version 5 left it, the index has the traditional words only.
    expect(matching("目標")).toBeGreaterThan(0);
    expect(matching("目标")).toBe(0);
    const embeddedBefore = embedder.texts.length;

    const second = startCore(dataDir, { embedder });
    const statuses: [string, Document["status"]][] = [];
    second.on("document.status", (document) => statuses.push([document.id, document.status]));
    // Queued at startup: the traditional Document, and the English one by its name alone.
    expect(
      Object.fromEntries(
        (await second.listDocuments()).map((document) => [document.id, document.status]),
      ),
    ).toEqual({ [goals.id]: "queued", [report.id]: "queued", [notes.id]: "ready" });
    await waitForProcessing(second, [goals.id, report.id]);

    // The other was left as it was: no status, and the Passages and vectors it had.
    expect(statuses.filter(([id]) => id === notes.id)).toEqual([]);
    expect(passages(notes.id)).toEqual(before.notes);
    const embeddedAgain = embedder.texts.slice(embeddedBefore);
    expect(embeddedAgain).toHaveLength(passages(goals.id).length + passages(report.id).length);
    expect(embeddedAgain.some((text) => text.includes("self-attention"))).toBe(false);
    // All three are as of this version, and the next start won't look through their text again.
    expect(queryDatabase(dataDir, "SELECT processing_version FROM documents")).toEqual(
      Array(3).fill({ processing_version: PROCESSING_VERSION }),
    );

    // Processed again, the traditional Document's Passages have the text they had; its
    // index has the simplified words, which a simplified query finds.
    expect(passages(goals.id).map((passage) => passage.text)).toEqual(
      before.goals.map((passage) => passage.text),
    );
    expect(matching("目标")).toBeGreaterThan(0);
    expect(matching("目標")).toBe(0);
    const found = await second.searchPassages("可持续发展目标", { mode: "keyword" });
    expect(found.map((result) => result.documentId)).toEqual([goals.id]);
    expect(found[0]?.text).toContain("可持續發展目標");
    const byName = await second.searchPassages("季度报告", { mode: "keyword" });
    expect(byName.map((result) => result.documentId)).toEqual([report.id]);
    second.close();

    // The next start has nothing to do.
    const third = startCore(dataDir, { embedder });
    const later: string[] = [];
    third.on("document.status", (document) => later.push(document.id));
    expect((await third.listDocuments()).map((document) => document.status)).toEqual(
      Array(3).fill("ready"),
    );
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(later).toEqual([]);
    third.close();
  });

  test("Word Documents are processed once more, so their comments are searched; Documents of other kinds are left as they are (#76)", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createSourceFolder();
    const first = startCore(dataDir);
    const [word, notes] = (await addAndProcess(first, [
      await writeSourceFile(
        sources,
        "Library.docx",
        docxOf([
          { text: "Costs", heading: 1 },
          {
            text: "A contingency of 10 per cent is held.",
            comment: { text: FINANCE, author: "Reviewer" },
          },
        ]),
      ),
      await writeSourceFile(sources, "Transformers.md", NOTES),
    ])) as [Document, Document];
    first.close();
    asVersion6(dataDir, word.id);
    const passages = (documentId: string) =>
      queryDatabase(
        dataDir,
        "SELECT id, text FROM passages WHERE document_id = ? AND deleted_at IS NULL",
        [documentId],
      );
    const before = passages(notes.id);
    // As version 6 left it, the comment isn't in the keyword index.
    const matching = (word: string) =>
      queryDatabase(dataDir, "SELECT rowid FROM passages_fts WHERE passages_fts MATCH ?", [
        `"${word}"`,
      ]).length;
    expect(matching("finance")).toBe(0);

    const second = startCore(dataDir);
    const statuses: [string, Document["status"]][] = [];
    second.on("document.status", (document) => statuses.push([document.id, document.status]));
    // Queued at startup: the Word Document only.
    expect(
      Object.fromEntries(
        (await second.listDocuments()).map((document) => [document.id, document.status]),
      ),
    ).toEqual({ [word.id]: "queued", [notes.id]: "ready" });
    await waitForProcessing(second, [word.id]);

    const found = await second.searchPassages("finance lower", { mode: "keyword" });
    expect(found.map((result) => result.documentId)).toEqual([word.id]);
    expect(found[0]?.text).toContain(`is held.\n\n${FINANCE}`);
    // The other was left as it was, and both are as of this version.
    expect(statuses.filter(([id]) => id === notes.id)).toEqual([]);
    expect(passages(notes.id)).toEqual(before);
    expect(queryDatabase(dataDir, "SELECT processing_version FROM documents")).toEqual(
      Array(2).fill({ processing_version: PROCESSING_VERSION }),
    );
    second.close();
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
    await turnOnEmbeddings(before);
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
