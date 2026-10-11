import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  type Core,
  type CoreEvents,
  DATABASE_FILE,
  type Document,
  type RecognizedLine,
  type TextRecognizer,
} from "../../src/core";
import { createFakeTextRecognizer, type FakeScanScript } from "../../src/core/documents/scan/fake";
import { BUILT_IN_TEXT_RECOGNITION_MODEL } from "../../src/core/documents/scan/model";
import { openDatabase } from "../../src/core/storage";
import { askAndFinish, citingModel, onlyCitation, setUpWithDocuments } from "../helpers/citations";
import { createTempDataFolder, NO_MODEL_FILES, queryDatabase, startCore } from "../helpers/core";
import {
  addAndProcess,
  createSourceFolder,
  waitForProcessing,
  writeSourceFile,
} from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";

const SCANS = join(import.meta.dirname, "../fixtures/scans");
const picture = (id: string) => readFileSync(join(SCANS, `${id}.jpg`));
const set = JSON.parse(readFileSync(join(SCANS, "set.json"), "utf8")) as {
  pages: { id: string; lines: string[] }[];
};
const linesOf = (id: string) => set.pages.find((page) => page.id === id)?.lines ?? [];

const READER = BUILT_IN_TEXT_RECOGNITION_MODEL.reader;

/** A recognizer reading what `script` says, counting the pages it reads. */
function countingRecognizer(script: FakeScanScript, fail: ReadonlySet<number> = new Set()) {
  const fake = createFakeTextRecognizer({ script });
  const read: number[] = [];
  const recognizer: TextRecognizer = {
    load: (files) => fake.load(files),
    async read(picture, page) {
      read.push(page);
      if (fail.has(page)) throw new Error("The model failed on this page.");
      return fake.read(picture, page);
    },
    close: () => fake.close(),
  };
  return { recognizer, read };
}

const scanning = (recognizer: TextRecognizer) => ({
  textRecognizer: recognizer,
  textRecognitionModelSource: NO_MODEL_FILES,
});

/** The stored Units of a Document's current version. */
function unitsOf(dataDir: string, document: Document) {
  return queryDatabase<{
    page: number;
    text: string;
    label: string | null;
    anchors: string | null;
  }>(
    dataDir,
    `SELECT page, text, label, anchors FROM document_pages
     WHERE document_id = ? AND content_hash = ? AND deleted_at IS NULL ORDER BY page`,
    [document.id, document.contentHash],
  );
}

async function addPdf(core: Core, name: string, pdf: Uint8Array): Promise<Document> {
  const sources = await createSourceFolder();
  const [document] = await addAndProcess(core, [await writeSourceFile(sources, name, pdf)]);
  if (!document) throw new Error("The PDF wasn't added.");
  return document;
}

describe("Scanned pages are read on this computer (#397)", { timeout: 60_000 }, () => {
  test("a scanned PDF's pages are read, with their progress, and become searchable", async () => {
    const dataDir = await createTempDataFolder();
    const { recognizer, read } = countingRecognizer({
      pages: { 1: { lines: linesOf("en-clean") }, 2: { lines: linesOf("zh-clean") } },
    });
    const core = startCore(dataDir, scanning(recognizer));
    const progress: Document["scan"][] = [];
    core.on("document.status", (document) => {
      if (document.scan) progress.push(document.scan);
    });

    const document = await addPdf(
      core,
      "Lease scan.pdf",
      buildPdf([{ scan: picture("en-clean") }, { scan: picture("zh-clean") }]),
    );

    expect(document.status).toBe("ready");
    expect(read).toEqual([1, 2]);
    expect(progress).toContainEqual({ pages: 2, read: 0 });
    expect(progress).toContainEqual({ pages: 2, read: 1 });
    expect(progress.at(-1)).toEqual({ pages: 2, read: 2 });
    const [english] = await core.searchPassages("undisputed invoice", { mode: "keyword" });
    expect(english).toMatchObject({ documentId: document.id, pageFrom: 1 });
    const [chinese] = await core.searchPassages("押金", { mode: "keyword" });
    expect(chinese).toMatchObject({ documentId: document.id, pageFrom: 2 });

    // Each page records what read it, and each line where it is on the page.
    const units = unitsOf(dataDir, document);
    expect(units.map((unit) => JSON.parse(unit.label ?? "null"))).toEqual([
      { scan: { reader: READER } },
      { scan: { reader: READER } },
    ]);
    const scanned = await core.readScannedText(document.id);
    expect(scanned.pages.map((page) => page.page)).toEqual([1, 2]);
    const firstLine = scanned.pages[0]?.anchors[0];
    expect(firstLine).toMatchObject({ start: 0, target: "line1" });
    expect(scanned.pages[0]?.text.slice(firstLine?.start, firstLine?.end)).toBe(
      "MASTER SERVICES AGREEMENT",
    );
    expect(firstLine?.quad).toHaveLength(8);
  });

  test("a mixed PDF keeps its text pages as they were, and reads only its scanned one", async () => {
    const dataDir = await createTempDataFolder();
    const { recognizer, read } = countingRecognizer({
      pages: { 2: { lines: linesOf("en-noisy") } },
    });
    const core = startCore(dataDir, scanning(recognizer));
    const text = { lines: ["Survey of the harbour", "The harbour floor is mostly sand."] };
    const document = await addPdf(
      core,
      "Survey.pdf",
      buildPdf([text, { scan: picture("en-noisy") }]),
    );

    expect(read).toEqual([2]);
    const units = unitsOf(dataDir, document);
    expect(units[0]).toEqual({
      page: 1,
      text: "Survey of the harbour\nThe harbour floor is mostly sand.",
      label: null,
      anchors: null,
    });
    expect(units[1]?.text).toContain("Mean water temperature rose to 9.4 degrees Celsius");
    expect(JSON.parse(units[1]?.label ?? "null")).toEqual({ scan: { reader: READER } });

    // Without a model to read them, a scanned page is left as it was before.
    const plainDir = await createTempDataFolder();
    const plain = startCore(plainDir);
    const left = await addPdf(plain, "Survey.pdf", buildPdf([text, { scan: picture("en-noisy") }]));
    expect(unitsOf(plainDir, left).map((unit) => unit.text)).toEqual([units[0]?.text, ""]);
  });

  test("an Answer cites a scanned page, and the Citation check finds its quote", async () => {
    const quote = "Either party may terminate this Agreement for convenience";
    const model = citingModel({
      query: "terminate for convenience",
      records: (passages) => [
        {
          marker: 1,
          passage: passages[0]?.id ?? "",
          pageFrom: 1,
          pageTo: 1,
          quote,
        },
      ],
      answer: "Either party can end it on 90 days' notice [^1].",
    });
    const { recognizer } = countingRecognizer({
      pages: { 1: { lines: linesOf("en-clean"), unsure: ["convenience"] } },
    });
    const { core, client, mind, documents } = await setUpWithDocuments(
      model,
      [{ name: "Agreement.pdf", contents: buildPdf([{ scan: picture("en-clean") }]) }],
      scanning(recognizer),
    );
    const { answerId } = await askAndFinish(core, client, mind.id, "How can it be ended?");
    const citation = onlyCitation(client, answerId);
    expect(citation).toMatchObject({
      documentId: documents[0]?.id,
      pageFrom: 1,
      quote,
      check: "found",
    });
    // The card says it was read from a scan, and which of its words the reading wasn't sure of.
    expect(
      await core.readScannedQuote({
        documentId: citation.documentId as string,
        contentHash: citation.contentHash ?? null,
        pageFrom: 1,
        pageTo: 1,
        quote,
      }),
    ).toEqual({ unsure: ["convenience"] });
    expect(
      await core.readScannedQuote({
        documentId: citation.documentId as string,
        contentHash: citation.contentHash ?? null,
        pageFrom: 1,
        pageTo: 1,
        quote: "Business Days and certify the deletion",
      }),
    ).toEqual({ unsure: [] });
  });

  test("pages that couldn't be read are named, and Retry reads only them", async () => {
    const dataDir = await createTempDataFolder();
    const failing = new Set([2]);
    const { recognizer, read } = countingRecognizer(
      { pages: { 1: { lines: linesOf("en-clean") }, 2: { lines: linesOf("en-noisy") } } },
      failing,
    );
    const core = startCore(dataDir, scanning(recognizer));
    const pdf = buildPdf([{ scan: picture("en-clean") }, { scan: picture("en-noisy") }]);
    const document = await addPdf(core, "Two scans.pdf", pdf);

    expect(document.status).toBe("failed");
    expect(document.failure).toMatchObject({ reason: "pages-unread", pages: [2] });
    expect(document.failure?.message).toContain("The model failed on this page.");
    // The page that was read is searched meanwhile.
    expect(await core.searchPassages("undisputed invoice", { mode: "keyword" })).toHaveLength(1);

    failing.clear();
    read.length = 0;
    await core.retryDocument(document.id);
    const [retried] = await waitForProcessing(core, [document.id]);
    expect(retried?.status).toBe("ready");
    expect(retried?.failure).toBeNull();
    expect(read).toEqual([2]);
    const [found] = await core.searchPassages("oystercatchers", { mode: "keyword" });
    expect(found).toMatchObject({ documentId: document.id, pageFrom: 2 });
  });

  test("re-reading after a reader's upgrade keeps the pages a Citation quotes as they were read", async () => {
    const quote = "Either party may terminate this Agreement for convenience";
    const model = citingModel({
      query: "terminate",
      records: (passages) => [
        { marker: 1, passage: passages[0]?.id ?? "", pageFrom: 1, pageTo: 1, quote },
      ],
      answer: "It can be ended [^1].",
    });
    const first = countingRecognizer({
      pages: { 1: { lines: linesOf("en-clean") }, 2: { lines: linesOf("en-noisy") } },
    });
    const pdf = buildPdf([{ scan: picture("en-clean") }, { scan: picture("en-noisy") }]);
    const { core, client, mind, dataDir, documents } = await setUpWithDocuments(
      model,
      [{ name: "Agreement.pdf", contents: pdf }],
      scanning(first.recognizer),
    );
    const document = documents[0] as Document;
    const { answerId } = await askAndFinish(core, client, mind.id, "Can it be ended?");
    expect(onlyCitation(client, answerId).check).toBe("found");
    await client.settled();
    core.close();

    // As an older reader would have left it: its pages read by "PP-OCRv4 mobile/1".
    const db = openDatabase(join(dataDir, DATABASE_FILE));
    try {
      db.run(
        `UPDATE document_pages SET label = json_set(label, '$.scan.reader', 'PP-OCRv4 mobile/1')
         WHERE document_id = ?`,
        [document.id],
      );
      db.run("UPDATE documents SET processing_version = 7 WHERE id = ?", [document.id]);
    } finally {
      db.close();
    }
    // The new reader reads both pages differently.
    const upgraded = countingRecognizer({
      pages: {
        1: { lines: linesOf("en-clean").map((line) => line.replaceAll("Agreement", "Agreernent")) },
        2: { lines: linesOf("en-noisy").map((line) => line.replaceAll("estuary", "estuarine")) },
      },
    });
    const restarted = startCore(dataDir, {
      ...scanning(upgraded.recognizer),
      createChatModel: () => {
        throw new Error("No Answers here.");
      },
    });
    const [reread] = await waitForProcessing(restarted, [document.id]);
    expect(reread?.status).toBe("ready");
    // Page 1, which the Citation quotes, is kept as the older reader read it; page 2 is read again.
    expect(upgraded.read).toEqual([2]);
    const units = unitsOf(dataDir, reread as Document);
    expect(units[0]?.text).toContain(quote);
    expect(JSON.parse(units[0]?.label ?? "{}").scan.reader).toBe("PP-OCRv4 mobile/1");
    expect(units[1]?.text).toContain("northern estuarine");
    expect(JSON.parse(units[1]?.label ?? "{}").scan.reader).toBe(READER);
    const recheck = await restarted.recheckCitation({
      documentId: document.id,
      quote,
      pageFrom: 1,
      pageTo: 1,
    });
    expect(recheck.check).toBe("found");
  });

  test("pausing a Linked folder stops reading its scan, and resuming goes on from where it stopped", async () => {
    const dataDir = await createTempDataFolder();
    const pages = [1, 2, 3, 4].map((page) => [page, { lines: [`Page ${page} of the minutes.`] }]);
    const fake = createFakeTextRecognizer({
      script: { pages: Object.fromEntries(pages) },
      delayMs: 150,
    });
    const read: number[] = [];
    const recognizer: TextRecognizer = {
      load: (files) => fake.load(files),
      read: (picture, page): Promise<RecognizedLine[]> => {
        read.push(page);
        return fake.read(picture, page);
      },
      close: () => fake.close(),
    };
    const core = startCore(dataDir, scanning(recognizer));
    const folder = await createSourceFolder();
    const scan = { scan: picture("table") };
    await writeSourceFile(folder, "Minutes.pdf", buildPdf([scan, scan, scan, scan]));
    const linked = await core.addLinkedFolder(folder);
    const midway = new Promise<void>((resolve) => {
      const stop = core.on("document.status", (document: CoreEvents["document.status"]) => {
        if ((document.scan?.read ?? 0) >= 2) {
          stop();
          resolve();
        }
      });
    });
    await midway;
    await core.setLinkedFolderPaused(linked.id, true);
    const readWhenPaused = read.length;
    await new Promise((resolve) => setTimeout(resolve, 600));
    // At most the page under way when it was paused is read after.
    expect(read.length).toBeLessThanOrEqual(readWhenPaused + 1);

    await core.setLinkedFolderPaused(linked.id, false);
    const [document] = await core.listDocuments();
    const [done] = await waitForProcessing(core, [document?.id as string]);
    expect(done?.status).toBe("ready");
    // The pages read before the pause weren't read again; only the one under way could be.
    expect(new Set(read)).toEqual(new Set([1, 2, 3, 4]));
    expect(read.filter((page) => page <= 2)).toEqual([1, 2]);
  });
});
