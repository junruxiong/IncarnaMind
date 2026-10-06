import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { type Document, InvalidInputError, NotFoundError } from "../../src/core";
import { createTempDataFolder, startCore, tickingClock } from "../helpers/core";
import {
  addAndProcess,
  sha256,
  storedFile,
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

  test("adding a file copies it into the data folder under its SHA-256 content hash", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
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
        size: Buffer.byteLength(ENGLISH_NOTES),
        pageCount: null,
        status: "queued",
        failure: null,
        folderId: null,
        createdAt: "2026-10-06T12:00:00.000Z",
        updatedAt: "2026-10-06T12:00:00.000Z",
      },
    ]);
    expect(await readFile(storedFile(dataDir, sha256(ENGLISH_NOTES)), "utf8")).toBe(ENGLISH_NOTES);
  });

  test("adding the same file twice, even under another name, gives the same Document", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const original = await writeSourceFile(sources, "notes.txt", CHINESE_NOTES);
    const copy = await writeSourceFile(sources, "copy of notes.txt", CHINESE_NOTES);

    const first = await core.addDocuments([original]);
    const second = await core.addDocuments([original, copy]);

    const id = first.documents[0]?.id;
    expect(second.documents.map((document) => document.id)).toEqual([id, id]);
    expect((await core.listDocuments()).map((document) => document.id)).toEqual([id]);
    const stored = await readdir(join(dataDir, "documents"));
    expect(stored.filter((name) => !name.startsWith("."))).toEqual([sha256(CHINESE_NOTES)]);
  });

  test("processing reports queued, then extracting, then ready", async () => {
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
      "ready",
    ]);
    expect(ready).toMatchObject({ status: "ready", failure: null, pageCount: null });
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
    const passages = (await core.searchPassages("fox", 200)).sort(
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
    // Pages are about 300 tokens and Passages about 400, so some cross a page break.
    expect(
      passages.some((passage) => (passage.pageFrom as number) < (passage.pageTo as number)),
    ).toBe(true);
    // Each marker is found within the page range of every Passage that contains it.
    for (const [index, marker] of MARKERS.entries()) {
      const found = await core.searchPassages(marker);
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

    const selfAttention = await core.searchPassages("self-attention");
    expect(selfAttention).toEqual([
      {
        passageId: expect.stringMatching(UUID_V4),
        documentId: english?.id,
        documentName: "Transformers",
        pageFrom: null,
        pageTo: null,
        position: 0,
        text: expect.stringContaining("relies entirely on self-attention"),
      },
    ]);

    const attention = await core.searchPassages("注意力机制");
    expect(attention.map((result) => result.documentId).sort()).toEqual(
      [chinese?.id, pdf?.id].sort(),
    );
    // The PDF is short: one Passage covers both its pages.
    const inPdf = attention.find((result) => result.documentId === pdf?.id);
    expect(inPdf).toMatchObject({ documentName: "讲义", pageFrom: 1, pageTo: 2 });
    expect(inPdf?.text).toContain("自注意力机制");

    // Two-character words are too short for the trigram index, so they are scanned for.
    const model = await core.searchPassages("模型");
    expect(model.map((result) => result.documentId)).toEqual([chinese?.id]);

    expect(await core.searchPassages("convolution")).toEqual([]);
    expect(await core.searchPassages("   ")).toEqual([]);
  });

  test("search returns the best matches first, up to the limit", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    await addAndProcess(core, [
      await writeSourceFile(sources, "once.txt", "Attention appears once here, with other words."),
      await writeSourceFile(sources, "twice.txt", "Attention, attention: the word appears twice."),
    ]);

    const results = await core.searchPassages("attention");
    expect(results.map((result) => result.documentName)).toEqual(["twice", "once"]);
    expect(await core.searchPassages("attention", 1)).toEqual([results[0]]);
    await expect(core.searchPassages("attention", 0)).rejects.toThrow(InvalidInputError);
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

  test("deleting a Document hides it and its Passages, and removes its file", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const [english, chinese] = await addAndProcess(core, [
      await writeSourceFile(sources, "english.md", ENGLISH_NOTES),
      await writeSourceFile(sources, "chinese.txt", CHINESE_NOTES),
    ]);
    if (!english || !chinese) throw new Error("Nothing was added.");

    await core.deleteDocument(english.id);

    expect(await core.listDocuments()).toEqual([chinese]);
    expect(await core.searchPassages("Transformer")).toEqual([]);
    expect(await core.searchPassages("Recurrent")).toEqual([]);
    expect(await core.searchPassages("注意力")).toHaveLength(1);
    expect(await readdir(join(dataDir, "documents"))).not.toContain(english.contentHash);
    expect(await readdir(join(dataDir, "documents"))).toContain(chinese.contentHash);
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
    expect(await readFile(storedFile(dataDir, first.contentHash), "utf8")).toBe(ENGLISH_NOTES);
    expect((await core.searchPassages("Transformer")).map((result) => result.documentId)).toEqual([
      second?.id,
    ]);
  });

  test("files that aren't PDF, TXT or Markdown, or can't be read, are skipped", async () => {
    const sources = await createTempDataFolder();
    const core = startCore(await createTempDataFolder());
    const word = await writeSourceFile(sources, "Letter.docx", "not supported");
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

  test("Documents, their Passages and their files survive a restart", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const before = startCore(dataDir, { now: tickingClock() });
    const processed = await addAndProcess(before, [
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
    for (const document of processed) {
      expect(await readdir(join(dataDir, "documents"))).toContain(document.contentHash);
    }
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
    expect(document.status).toBe("queued");

    const after = startCore(dataDir);
    const [finished] = await waitForProcessing(after, [document.id]);

    expect(finished?.status).toBe("ready");
    expect(await after.searchPassages("Transformer")).toHaveLength(1);
  });
});
