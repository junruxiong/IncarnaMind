/**
 * Excel, CSV and PowerPoint Documents get no vectors while keyword search
 * alone is measured for them (ADR-0009, 2026-10-10): they are ready all the
 * same, keyword and hybrid search find them, and hybrid search doesn't rank
 * them below Passages both searches found only for having no vector.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { DATABASE_FILE } from "../../src/core";
import { fuseRankingScores } from "../../src/core/documents/search";
import { embedsPassages, KEYWORD_ONLY_KINDS } from "../../src/core/documents/vectors";
import { openDatabase } from "../../src/core/storage";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { createControlledEmbedder } from "../helpers/embedding";
import { mockEmbeddingModels } from "../helpers/embeddingProviders";

const fixture = (name: string) =>
  new Uint8Array(readFileSync(new URL(`../fixtures/formats/${name}`, import.meta.url)));

const STRUCTURED = ["Orders.csv", "Regional Revenue.xlsx", "Quarterly Research Update.pptx"];

/** A note that names a customer in Orders.csv once, among other words. */
const NOTES = `# Supplier notes

Evergreen Labs asked for its invoices by post from now on.

The Leeds warehouse closes early on Fridays, and the loading bay is shut for repairs.`;

async function setUp() {
  const dataDir = await createTempDataFolder();
  const sources = await createSourceFolder();
  const embedder = createControlledEmbedder();
  const core = startCore(dataDir, { embedder });
  const documents = await addAndProcess(core, [
    ...(await Promise.all(STRUCTURED.map((name) => writeSourceFile(sources, name, fixture(name))))),
    await writeSourceFile(sources, "Supplier notes.md", NOTES),
  ]);
  const [orders, , , notes] = documents;
  if (!orders || !notes) throw new Error("The Documents weren't added.");
  return { dataDir, embedder, core, documents, orders, notes };
}

describe("Excel, CSV and PowerPoint Documents", { timeout: 30_000 }, () => {
  test("are the kinds that get no vectors; the others are embedded", () => {
    expect([...KEYWORD_ONLY_KINDS].sort()).toEqual(["csv", "pptx", "xlsx"]);
    expect(["pdf", "docx", "markdown", "text"].every(embedsPassages)).toBe(true);
    expect(["csv", "pptx", "xlsx"].some(embedsPassages)).toBe(false);
  });

  test("are ready with no vectors: the model embeds only the other Documents' Passages", async () => {
    const { dataDir, embedder, documents } = await setUp();

    expect(documents.map((document) => [document.kind, document.status])).toEqual([
      ["csv", "ready"],
      ["xlsx", "ready"],
      ["pptx", "ready"],
      ["markdown", "ready"],
    ]);
    const stored = queryDatabase<{ kind: string; passages: number; vectors: number }>(
      dataDir,
      `SELECT d.kind, count(*) AS passages, count(p.embedding) AS vectors
       FROM passages p JOIN documents d ON d.id = p.document_id
       WHERE p.deleted_at IS NULL GROUP BY d.kind ORDER BY d.kind`,
    );
    expect(stored.map(({ kind, vectors }) => [kind, vectors > 0])).toEqual([
      ["csv", false],
      ["markdown", true],
      ["pptx", false],
      ["xlsx", false],
    ]);
    const markdown = stored.find((row) => row.kind === "markdown");
    expect(stored.every((row) => row.passages > 0)).toBe(true);
    expect(markdown?.vectors).toBe(markdown?.passages);
    expect(embedder.texts).toHaveLength(markdown?.passages ?? -1);
    // Recorded with the current model, as an embedded Document is: nothing is left to embed.
    expect(
      queryDatabase<{ model: string | null }>(
        dataDir,
        "SELECT DISTINCT embedding_model AS model FROM documents WHERE deleted_at IS NULL",
      ),
    ).toHaveLength(1);
  });

  test("are found by keyword search and hybrid search, never by vector search, and hybrid search keeps their keyword rank", async () => {
    const { core, orders, notes } = await setUp();
    const query = "Evergreen Labs";
    const ids = (found: { documentId: string }[]) => [
      ...new Set(found.map((each) => each.documentId)),
    ];

    const keyword = await core.searchPassages(query, { mode: "keyword" });
    const vector = await core.searchPassages(query, { mode: "vector" });
    const hybrid = await core.searchPassages(query);

    // Keyword search ranks the orders first and the notes after them; vector search
    // finds only the notes, the one Document with vectors.
    expect(ids(keyword)).toEqual([orders.id, notes.id]);
    expect(ids(vector)).toEqual([notes.id]);
    // Plain fusion would put the notes first, found by both searches. The orders' Passage
    // can't be in vector search's list, so its keyword rank counts for both, and it stays first.
    expect(ids(hybrid)).toEqual([orders.id, notes.id]);
    expect(hybrid[0]?.passageId).toBe(keyword[0]?.passageId);
  });

  test("keep any vectors they had from before unused", async () => {
    const { dataDir, core, orders, notes } = await setUp();
    core.close();
    // As if Orders.csv had been embedded before: its Passages get the notes' first vector.
    const db = openDatabase(join(dataDir, DATABASE_FILE));
    try {
      db.run(
        `UPDATE passages SET embedding = (
           SELECT embedding FROM passages WHERE document_id = ? AND embedding IS NOT NULL
           ORDER BY position LIMIT 1)
         WHERE document_id = ?`,
        [notes.id, orders.id],
      );
    } finally {
      db.close();
    }

    const restarted = startCore(dataDir, { embedder: createControlledEmbedder() });
    const found = await restarted.searchPassages("Evergreen Labs invoices", { mode: "vector" });

    expect(found.length).toBeGreaterThan(0);
    expect(found.every((passage) => passage.documentId === notes.id)).toBe(true);
    expect((await restarted.listDocuments()).every((each) => each.status === "ready")).toBe(true);
  });

  test("are ready again at once when the User switches embedding model, and nothing of theirs is sent", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createSourceFolder();
    const embeddings = mockEmbeddingModels();
    const core = startCore(dataDir, {
      embedder: createControlledEmbedder(),
      createEmbeddingModel: embeddings.createEmbeddingModel,
    });
    await addAndProcess(core, [
      await writeSourceFile(sources, "Orders.csv", fixture("Orders.csv")),
      await writeSourceFile(sources, "Supplier notes.md", NOTES),
    ]);
    const rebuilt = new Promise<void>((resolve) => {
      const stop = core.on("embedding.changed", (settings) => {
        if (settings.rebuild !== null) return;
        stop();
        resolve();
      });
    });

    // A server on this computer: no consent is asked.
    await core.saveEmbeddingProvider({
      kind: "openai-compatible",
      baseUrl: "http://127.0.0.1:9/v1",
      modelId: "mock-embed",
    });
    await rebuilt;

    expect(embeddings.passages().length).toBeGreaterThan(0);
    expect(embeddings.passages().every((text) => text.startsWith("Supplier notes\n"))).toBe(true);
    expect((await core.listDocuments()).map((each) => [each.kind, each.status])).toEqual([
      ["markdown", "ready"],
      ["csv", "ready"],
    ]);
    expect(
      queryDatabase<{ model: string | null }>(
        dataDir,
        "SELECT DISTINCT embedding_model AS model FROM documents WHERE deleted_at IS NULL",
      ),
    ).toEqual([{ model: "openai-compatible@http://127.0.0.1:9/v1:mock-embed" }]);
  });
});

describe("Hybrid search's fusion", () => {
  test("counts a Passage with no vector by its keyword rank for both lists, so it isn't ranked below Passages both found", () => {
    const keyword = [1, 2, 3];
    const vector = [2, 4];
    const order = (keywordOnly?: Set<number>) =>
      fuseRankingScores([keyword, vector], 10, keywordOnly).map((hit) => hit.seq);

    // Plain fusion: 2, in both lists, comes first, though keyword search put 1 first.
    expect(order()).toEqual([2, 1, 4, 3]);
    // 1 and 3 have no vector: their keyword ranks count for vector search's list too.
    expect(order(new Set([1, 3]))).toEqual([1, 2, 3, 4]);
    expect(fuseRankingScores([keyword, vector], 1, new Set([1]))).toEqual([
      { seq: 1, score: 2 / 61 },
    ]);
    // While vector search returns nothing (the query couldn't be embedded), nothing changes.
    expect(fuseRankingScores([keyword, []], 10, new Set([1, 3]))).toEqual(
      fuseRankingScores([keyword, []], 10),
    );
  });
});
