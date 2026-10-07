import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { type Document, InvalidInputError, NotFoundError } from "../../src/core";
import { createTempDataFolder, startCore, tickingClock } from "../helpers/core";
import {
  addAndProcess,
  createSourceFolder,
  sha256,
  waitForProcessing,
  writeSourceFile,
} from "../helpers/documents";
import { buildPdf, type PdfPage } from "../helpers/pdf";

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

const ENGLISH_NOTES = `# Transformers

The Transformer architecture relies entirely on self-attention to draw global
dependencies between input and output.

Recurrent networks process tokens one at a time, which limits parallelism.`;

const CHINESE_NOTES = `第一章 深度学习

卷积神经网络擅长处理图像。循环神经网络擅长处理序列数据。

注意力机制让模型能够关注输入中最重要的部分。`;

/** About 300 approximate tokens: 20 sentences, with `marker` in the middle one. */
const pageOfText = (page: number, marker: string): PdfPage => ({
  lines: Array.from({ length: 20 }, (_, line) =>
    line === 10
      ? `Page ${page} holds the ${marker} that this test searches for.`
      : `Page ${page}, line ${line}: the quick brown fox jumps over a lazy dog.`,
  ),
});

const MARKERS = ["Alphamarker", "Bravomarker", "Charliemarker"];

describe("Documents", { timeout: 30_000 }, () => {
  test("a new data folder has no Documents", async () => {
    const core = startCore(await createTempDataFolder());

    expect(await core.listDocuments()).toEqual([]);
    expect(await core.searchPassages("anything")).toEqual([]);
  });

  test("adding a file indexes it where it is, under the SHA-256 of its content, without copying it", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createSourceFolder();
    const core = startCore(dataDir, { now: () => new Date("2026-10-06T12:00:00Z") });
    const path = await writeSourceFile(sources, "Reading notes.md", ENGLISH_NOTES);

    const { documents, skipped } = await core.addDocuments([path]);

    expect(skipped).toEqual([]);
    expect(documents).toEqual([
      {
        id: expect.stringMatching(UUID_V4),
        name: "Reading notes",
        kind: "markdown",
        contentHash: sha256(ENGLISH_NOTES),
        path,
        fileStatus: "available",
        linkedFolderId: null,
        size: Buffer.byteLength(ENGLISH_NOTES),
        pageCount: null,
        status: expect.stringMatching(/^(queued|extracting)$/),
        progress: null,
        failure: null,
        folderId: null,
        tags: [],
        tagging: "pending",
        taggingError: null,
        createdAt: "2026-10-06T12:00:00.000Z",
        updatedAt: "2026-10-06T12:00:00.000Z",
      },
    ]);
    expect(existsSync(join(dataDir, "documents"))).toBe(false);
    expect(await readFile(path, "utf8")).toBe(ENGLISH_NOTES);
  });

  test("a file is a Document at its path: added twice it is one, and a copy elsewhere is another", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createSourceFolder();
    const core = startCore(dataDir);
    const original = await writeSourceFile(sources, "notes.txt", CHINESE_NOTES);
    const copy = await writeSourceFile(sources, "copy of notes.txt", CHINESE_NOTES);

    const first = await core.addDocuments([original]);
    const second = await core.addDocuments([original, copy]);

    const id = first.documents[0]?.id;
    const [again, other] = second.documents;
    expect(again?.id).toBe(id);
    expect(other?.id).not.toBe(id);
    expect(other).toMatchObject({ path: copy, contentHash: sha256(CHINESE_NOTES) });
    expect((await core.listDocuments()).map((document) => document.path).sort()).toEqual(
      [copy, original].sort(),
    );
  });

  test("processing reports queued, extracting, embedding, then ready", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    const seen: Document[] = [];
    core.on("document.status", (document) => seen.push(document));

    const [added] = (
      await core.addDocuments([await writeSourceFile(sources, "a.md", ENGLISH_NOTES)])
    ).documents;
    if (!added) throw new Error("Nothing was added.");
    const [ready] = await waitForProcessing(core, [added.id]);

    expect(seen.filter((each) => each.id === added.id).map((each) => each.status)).toEqual([
      "queued",
      "extracting",
      "embedding",
      "ready",
    ]);
    expect(seen.find((each) => each.status === "embedding")?.progress).toBe(0);
    expect(ready).toMatchObject({
      status: "ready",
      progress: null,
      failure: null,
      pageCount: null,
    });
    expect(await core.listDocuments()).toEqual([ready]);
  });

  test("a PDF's text is extracted page by page, and each Passage records the pages it covers", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    const pdf = buildPdf(MARKERS.map((marker, index) => pageOfText(index + 1, marker)));

    const [document] = await addAndProcess(core, [
      await writeSourceFile(sources, "Report.pdf", pdf),
    ]);

    expect(document).toMatchObject({ kind: "pdf", status: "ready", pageCount: 3 });
    // Every Passage has the word "fox"; read them all back in order.
    const passages = (await core.searchPassages("fox", { mode: "keyword", limit: 200 })).sort(
      (a, b) => a.position - b.position,
    );
    expect(passages.map((passage) => passage.position)).toEqual(passages.map((_, index) => index));
    expect(passages.length).toBeGreaterThan(1);
    expect(passages[0]?.pageFrom).toBe(1);
    expect(passages.at(-1)?.pageTo).toBe(3);
    for (const passage of passages) {
      expect(passage.pageFrom).not.toBeNull();
      expect(passage.pageFrom as number).toBeLessThanOrEqual(passage.pageTo as number);
    }
    // Pages are about 300 tokens and Passages about 500, so some cross a page break.
    expect(
      passages.some((passage) => (passage.pageFrom as number) < (passage.pageTo as number)),
    ).toBe(true);
    // Each marker is found within the page range of every Passage that contains it.
    for (const [index, marker] of MARKERS.entries()) {
      const found = await core.searchPassages(marker, { mode: "keyword" });
      expect(found.length).toBeGreaterThan(0);
      for (const passage of found) {
        expect(passage.text).toContain(marker);
        expect(passage.pageFrom as number).toBeLessThanOrEqual(index + 1);
        expect(passage.pageTo as number).toBeGreaterThanOrEqual(index + 1);
      }
    }
  });

  test.each([
    {
      name: "a scanned PDF (images, no text layer)",
      file: "Scan.pdf",
      contents: buildPdf([{ image: true }, { image: true }]),
      pageCount: 2,
    },
    { name: "an empty text file", file: "Empty.txt", contents: "  \n\n ", pageCount: null },
  ])("$name ends in 'no text found'", async ({ file, contents, pageCount }) => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());

    const [document] = await addAndProcess(core, [await writeSourceFile(sources, file, contents)]);

    expect(document).toMatchObject({ status: "no-text", failure: null, pageCount });
  });

  test("a corrupt file fails with a reason, and other Documents still process", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    const corrupt = await writeSourceFile(sources, "Broken.pdf", "This is not a PDF at all.");
    const fine = await writeSourceFile(sources, "Fine.md", ENGLISH_NOTES);

    const [broken, other] = await addAndProcess(core, [corrupt, fine]);

    expect(broken?.status).toBe("failed");
    expect(broken?.failure).toEqual({
      reason: "unreadable",
      message: expect.stringContaining("Invalid PDF"),
    });
    expect(other?.status).toBe("ready");
  });

  test("a Document that failed is processed again on retry, from its file as it is now", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    const path = await writeSourceFile(sources, "Report.pdf", "This is not a PDF at all.");
    const [broken] = await addAndProcess(core, [path]);
    expect(broken?.status).toBe("failed");

    // Only a failed Document can be retried; an unknown one can't.
    const [fine] = await addAndProcess(core, [
      await writeSourceFile(sources, "Fine.md", ENGLISH_NOTES),
    ]);
    await expect(core.retryDocument(fine?.id as string)).rejects.toThrow(InvalidInputError);
    await expect(core.retryDocument("no-such-document")).rejects.toThrow(NotFoundError);

    // The User fixes the file, then retries: it is read again, and ready.
    await writeSourceFile(sources, "Report.pdf", buildPdf([{ lines: ["Quarterly report"] }]));
    const queued = await core.retryDocument(broken?.id as string);
    expect(queued).toMatchObject({ id: broken?.id, status: "queued", failure: null });
    const [retried] = await waitForProcessing(core, [broken?.id as string]);
    expect(retried).toMatchObject({ status: "ready", failure: null, pageCount: 1 });
  });

  test("keyword search finds Passages in English and Chinese", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    const chinesePdf = buildPdf([
      { chineseLines: ["机器学习导论"] },
      { chineseLines: ["自注意力机制计算序列中每个位置与其他位置的关系。"] },
    ]);
    const [english, chinese, pdf] = await addAndProcess(core, [
      await writeSourceFile(sources, "Transformers.md", ENGLISH_NOTES),
      await writeSourceFile(sources, "深度学习笔记.txt", CHINESE_NOTES),
      await writeSourceFile(sources, "讲义.pdf", chinesePdf),
    ]);

    const selfAttention = await core.searchPassages("self-attention", { mode: "keyword" });
    expect(selfAttention).toEqual([
      {
        passageId: expect.stringMatching(UUID_V4),
        documentId: english?.id,
        documentName: "Transformers",
        // Its first section (ADR-0011): Markdown is stored by heading.
        pageFrom: 1,
        pageTo: 1,
        position: 0,
        text: expect.stringContaining("relies entirely on self-attention"),
      },
    ]);

    const attention = await core.searchPassages("注意力机制", { mode: "keyword" });
    expect(attention.map((result) => result.documentId).sort()).toEqual(
      [chinese?.id, pdf?.id].sort(),
    );
    // The PDF is short: one Passage covers both its pages.
    const inPdf = attention.find((result) => result.documentId === pdf?.id);
    expect(inPdf).toMatchObject({ documentName: "讲义", pageFrom: 1, pageTo: 2 });
    expect(inPdf?.text).toContain("自注意力机制");

    // Chinese is split into words, so a two-character word matches on its own.
    const model = await core.searchPassages("模型", { mode: "keyword" });
    expect(model.map((result) => result.documentId)).toEqual([chinese?.id]);

    expect(await core.searchPassages("convolution", { mode: "keyword" })).toEqual([]);
    expect(await core.searchPassages("   ")).toEqual([]);
  });

  test("search returns the best matches first, up to the limit", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    await addAndProcess(core, [
      await writeSourceFile(sources, "once.txt", "Attention appears once here, with other words."),
      await writeSourceFile(sources, "twice.txt", "Attention, attention: the word appears twice."),
    ]);

    const results = await core.searchPassages("attention", { mode: "keyword" });
    expect(results.map((result) => result.documentName)).toEqual(["twice", "once"]);
    expect(await core.searchPassages("attention", { mode: "keyword", limit: 1 })).toEqual([
      results[0],
    ]);
    await expect(core.searchPassages("attention", { limit: 0 })).rejects.toThrow(InvalidInputError);
  });

  test("renaming a Document changes its name in the list and in search results", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder(), { now: tickingClock() });
    const [document] = await addAndProcess(core, [
      await writeSourceFile(sources, "draft.md", ENGLISH_NOTES),
    ]);
    if (!document) throw new Error("Nothing was added.");

    const renamed = await core.renameDocument(document.id, "  Attention paper  ");

    expect(renamed.name).toBe("Attention paper");
    expect(renamed.updatedAt > document.updatedAt).toBe(true);
    expect(await core.listDocuments()).toEqual([renamed]);
    expect((await core.searchPassages("Transformer"))[0]?.documentName).toBe("Attention paper");
    await expect(core.renameDocument(document.id, "   ")).rejects.toThrow(InvalidInputError);
    await expect(core.renameDocument("no-such-id", "Name")).rejects.toThrow(NotFoundError);
  });

  test("deleting a Document hides it and its Passages; its file isn't touched", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const englishPath = await writeSourceFile(sources, "english.md", ENGLISH_NOTES);
    const [english, chinese] = await addAndProcess(core, [
      englishPath,
      await writeSourceFile(sources, "chinese.txt", CHINESE_NOTES),
    ]);
    if (!english || !chinese) throw new Error("Nothing was added.");

    await core.deleteDocument(english.id);

    expect(await core.listDocuments()).toEqual([chinese]);
    expect(await core.searchPassages("Transformer", { mode: "keyword" })).toEqual([]);
    expect(await core.searchPassages("Recurrent", { mode: "keyword" })).toEqual([]);
    expect(await core.searchPassages("注意力", { mode: "keyword" })).toHaveLength(1);
    expect(await readFile(englishPath, "utf8")).toBe(ENGLISH_NOTES);
    await expect(core.deleteDocument(english.id)).rejects.toThrow(NotFoundError);
    await expect(core.renameDocument(english.id, "Back")).rejects.toThrow(NotFoundError);
  });

  test("a deleted file added again becomes a new Document", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const path = await writeSourceFile(sources, "notes.md", ENGLISH_NOTES);
    const [first] = await addAndProcess(core, [path]);
    if (!first) throw new Error("Nothing was added.");
    await core.deleteDocument(first.id);

    const [second] = await addAndProcess(core, [path]);

    expect(second?.id).not.toBe(first.id);
    expect(second?.status).toBe("ready");
    expect((await core.searchPassages("Transformer")).map((result) => result.documentId)).toEqual([
      second?.id,
    ]);
  });

  test("files of a kind IncarnaMind doesn't read, or that can't be read, are skipped", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    // Word 97–2003: the second wave of formats (ADR-0011).
    const word = await writeSourceFile(sources, "Letter.doc", "not supported");
    const missing = join(sources, "missing.txt");

    const result = await core.addDocuments([word, missing, sources]);

    expect(result).toEqual({
      documents: [],
      skipped: [
        { path: word, reason: "unsupported-type" },
        { path: missing, reason: "unreadable" },
        { path: sources, reason: "unsupported-type" },
      ],
    });
    expect(await core.listDocuments()).toEqual([]);
    await expect(core.addDocuments(["relative/notes.txt"])).rejects.toThrow(InvalidInputError);
    await expect(core.addDocuments("notes.txt" as never)).rejects.toThrow(InvalidInputError);
  });

  test("Documents and their Passages survive a restart", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const before = startCore(dataDir, { now: tickingClock() });
    await addAndProcess(before, [
      await writeSourceFile(sources, "english.md", ENGLISH_NOTES),
      await writeSourceFile(sources, "Report.pdf", buildPdf([pageOfText(1, "Alphamarker")])),
    ]);
    const listed = await before.listDocuments();
    const found = await before.searchPassages("Alphamarker");
    before.close();

    const after = startCore(dataDir);

    expect(await after.listDocuments()).toEqual(listed);
    expect(listed.map((document) => document.status)).toEqual(["ready", "ready"]);
    expect(await after.searchPassages("Alphamarker")).toEqual(found);
  });

  test("processing interrupted by quitting picks up again after a restart", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const before = startCore(dataDir);
    const { documents } = await before.addDocuments([
      await writeSourceFile(sources, "english.md", ENGLISH_NOTES),
    ]);
    before.close();
    const [document] = documents;
    if (!document) throw new Error("Nothing was added.");
    expect(["queued", "extracting"]).toContain(document.status);

    const after = startCore(dataDir);
    const [finished] = await waitForProcessing(after, [document.id]);

    expect(finished?.status).toBe("ready");
    expect(await after.searchPassages("Transformer")).toHaveLength(1);
  });
});
