/**
 * Importing references (#287): a CSL JSON, BibTeX or RIS export (from
 * Zotero, EndNote, Mendeley) is matched to Documents by DOI, then arXiv id,
 * then title, and written into their reference records; what matches nothing
 * is kept as a record with no Document; the User's edits are never replaced;
 * importing again updates; Undo takes an import back. Through the core's
 * public interface, with fixture exports: no network.
 */
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { describe, expect, onTestFinished, test } from "vitest";
import type { Core, Document, ReferenceImportResult, ReferenceRecord } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import {
  addAndProcess,
  createSourceFolder,
  waitForDocuments,
  writeSourceFile,
} from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";
import { recordedAnswer, startReferenceServices } from "../helpers/referenceServices";

const BOWLER_TITLE =
  "A systematic review of evidence for the added benefits to health of exposure to natural environments";
const EXPORTS = join(import.meta.dirname, "..", "fixtures", "references", "exports");

/** The fixture exports, copied beside a PDF the export points to (`files/rust-book.pdf`). */
async function exportsFolder(): Promise<string> {
  const folder = await createSourceFolder();
  for (const name of ["library.bib", "library.json", "library.ris", "edge.bib"]) {
    await copyFile(join(EXPORTS, name), join(folder, name));
  }
  await mkdir(join(folder, "files"));
  await writeFile(
    join(folder, "files", "rust-book.pdf"),
    buildPdf([{ lines: ["The Rust Programming Language", "Chapter 1. Getting started"] }]),
  );
  return folder;
}

const bowlerPdf = () =>
  buildPdf([{ lines: ["Background", "https://doi.org/10.1186/1471-2458-10-456"] }]);
const chinesePdf = () =>
  buildPdf([
    { chineseLines: ["城市绿地对热岛效应的缓解作用"], lines: ["DOI: 10.5555/chinese-heat-island"] },
  ]);
const attentionPdf = () =>
  buildPdf([{ lines: ["arXiv:1706.03762v7 [cs.CL] 2 Aug 2023", "Attention Is All You Need"] }]);
const halpernPdf = () =>
  buildPdf([{ lines: ["Chapter 1. Introduction"] }], {
    info: { Title: "Reasoning about Uncertainty", Author: "Joseph Halpern" },
  });
const invoicePdf = () => buildPdf([{ lines: ["INVOICE 2041"] }]);

/** Adds the four papers an export matches, and an invoice it doesn't, by file name. */
async function addPapers(core: Core) {
  const sources = await createSourceFolder();
  const files = {
    "bowler.pdf": bowlerPdf(),
    "chinese.pdf": chinesePdf(),
    "attention.pdf": attentionPdf(),
    "halpern.pdf": halpernPdf(),
    "invoice.pdf": invoicePdf(),
  };
  const paths: Record<string, string> = {};
  for (const [name, contents] of Object.entries(files)) {
    paths[name] = await writeSourceFile(sources, name, contents);
  }
  const documents = await addAndProcess(core, Object.values(paths));
  const byName = Object.fromEntries(
    Object.entries(paths).map(([name, path]) => [
      name,
      documents.find((document) => document.path === path) as Document,
    ]),
  );
  return { documents: byName, sources };
}

const imported = (result: ReferenceImportResult) => {
  if (result.outcome !== "imported") throw new Error(`Import failed: ${result.message}`);
  return result;
};

const documentOf = async (core: Core, id: string) =>
  (await core.listDocuments()).find((document) => document.id === id) as Document;

const byTitle = (records: readonly ReferenceRecord[]) =>
  Object.fromEntries(records.map((record) => [record.fields.title, record]));

describe("Importing references: the formats", () => {
  test("CSL JSON, BibTeX and RIS give the same fields for the same works", async () => {
    const folder = await exportsFolder();
    const read: Record<string, Record<string, ReferenceRecord>> = {};
    for (const name of ["library.json", "library.bib", "library.ris"]) {
      const core = startCore(await createTempDataFolder());
      const result = imported(await core.importReferences(join(folder, name)));
      expect(result.total).toBe(5);
      expect(result.matched).toBe(0);
      expect(result.unmatched).toBe(5);
      expect(result.problems).toEqual([]);
      read[name] = byTitle(await core.listUnmatchedReferences());
    }
    const fieldsOf = (name: string) =>
      Object.fromEntries(
        Object.entries(read[name] ?? {}).map(([title, record]) => [title, record.fields]),
      );
    expect(fieldsOf("library.bib")).toEqual(fieldsOf("library.json"));
    expect(fieldsOf("library.ris")).toEqual(fieldsOf("library.json"));

    const bowler = read["library.bib"]?.[BOWLER_TITLE];
    expect(bowler?.fields).toMatchObject({
      type: "article-journal",
      authors: [
        { family: "Bowler", given: "Diana E." },
        { family: "Buyung-Ali", given: "Lisette M." },
        { family: "Knight", given: "Teri M." },
        { family: "Pullin", given: "Andrew S." },
      ],
      issued: "2010-08",
      containerTitle: "BMC Public Health",
      volume: "10",
      issue: "1",
      pages: "456",
      doi: "10.1186/1471-2458-10-456",
    });
    expect(bowler?.sources).toMatchObject({ title: "import", authors: "import", doi: "import" });
    // Chinese names stay whole, family name first, however the export split them.
    const chinese = read["library.ris"]?.["城市绿地对热岛效应的缓解作用"];
    expect(chinese?.fields.authors).toEqual([{ literal: "张三" }, { literal: "李四" }]);
    expect(chinese?.fields.language).toBe("zh");
    // A preprint: an arXiv address and no DOI.
    expect(read["library.bib"]?.["Attention Is All You Need"]?.fields).toMatchObject({
      type: "article",
      url: "https://arxiv.org/abs/1706.03762",
    });
  });

  test("a RIS file with Windows line endings and a byte order mark reads the same", async () => {
    const folder = await exportsFolder();
    const text = await readFile(join(folder, "library.ris"), "utf8");
    const path = join(folder, "windows.ris");
    await writeFile(path, `﻿${text.replace(/\n/g, "\r\n")}`);
    const core = startCore(await createTempDataFolder());
    expect(imported(await core.importReferences(path)).total).toBe(5);
  });

  test("BibTeX: LaTeX escapes, organisations, macros, comments, and entries that can't be read", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    const result = imported(await core.importReferences(join(folder, "edge.bib")));

    // dup twice with one work is one reference; the same key for another work is another.
    expect(result.merged).toBe(1);
    expect(result.total).toBe(4 + 1);
    expect(result.problems).toEqual([
      { label: "broken", reason: "broken" },
      { label: "notitle", reason: "no-title" },
    ]);

    const records = byTitle(await core.listUnmatchedReferences());
    const muller = records["Deep Learning for Möbius Strips: Naïve & Fast"];
    expect(muller?.fields).toMatchObject({
      authors: [
        { family: "Müller", given: "Hans" },
        { family: "Dvořák", given: "Anton" },
        { literal: "World Health Organization" },
      ],
      containerTitle: "Journal of Applied Studies",
      pages: "12–34",
      doi: "10.1000/muller.1",
      url: "https://example.org/a_b",
      issued: "2019",
    });
    expect(records["A paper whose journal is a macro"]?.fields.containerTitle).toBe(
      "Journal of Computational Physics",
    );
    // The duplicate's later copy filled what the first lacked.
    expect(records["Duplicate Entry One"]?.fields.pages).toBe("1–9");
    expect(records["A Different Work With The Same Key"]).toBeDefined();
    // The entry after the broken one was still read.
    expect(records["The entry after the broken one"]).toBeDefined();
  });

  test("a file that is not references, or holds none, or can't be read, says so and changes nothing", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    const notes = await writeSourceFile(folder, "notes.txt", "Buy milk.\nCall the bank.\n");
    const empty = await writeSourceFile(folder, "empty.json", "[]");
    const garbled = await writeSourceFile(folder, "garbled.json", "[{");
    expect(await core.importReferences(notes)).toMatchObject({
      outcome: "failed",
      reason: "unknown-format",
    });
    expect(await core.importReferences(empty)).toMatchObject({
      outcome: "failed",
      reason: "empty",
    });
    expect(await core.importReferences(garbled)).toMatchObject({
      outcome: "failed",
      reason: "unknown-format",
    });
    expect(await core.importReferences(join(folder, "missing.bib"))).toMatchObject({
      outcome: "failed",
      reason: "unreadable",
    });
    expect(await core.listUnmatchedReferences()).toEqual([]);
  });
});

describe("Importing references: matching Documents", () => {
  test("matches by DOI, then arXiv id, then title; the rest is kept with no Document", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    const { documents } = await addPapers(core);

    const result = imported(await core.importReferences(join(folder, "library.bib")));
    expect(result).toMatchObject({ total: 5, matched: 4, unmatched: 1, changed: 5, merged: 0 });
    const how = Object.fromEntries(result.entries.map((entry) => [entry.title, entry.matchedBy]));
    expect(how).toEqual({
      [BOWLER_TITLE]: "doi",
      城市绿地对热岛效应的缓解作用: "doi",
      "Attention Is All You Need": "arxiv",
      "Reasoning about Uncertainty": "title",
      "The Rust Programming Language": null,
    });

    // What the import added sits in the Document's record, next to what its file gave.
    const halpern = (await documentOf(core, documents["halpern.pdf"]?.id as string)).reference;
    expect(halpern?.fields).toMatchObject({
      title: "Reasoning about Uncertainty",
      publisher: "MIT Press",
      issued: "2003",
      type: "book",
    });
    expect(halpern?.sources.publisher).toBe("import");
    const bowler = (await documentOf(core, documents["bowler.pdf"]?.id as string)).reference;
    expect(bowler?.fields.containerTitle).toBe("BMC Public Health");
    expect(bowler?.fields.authors).toHaveLength(4);
    // A Document nothing matched is left alone.
    expect(
      (await documentOf(core, documents["invoice.pdf"]?.id as string)).reference?.fields.type,
    ).toBe("document");

    const unmatched = await core.listUnmatchedReferences();
    expect(unmatched.map((record) => record.fields.title)).toEqual([
      "The Rust Programming Language",
    ]);
  });

  test("a field the User edited is never replaced, and one they cleared stays clear", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    const { documents } = await addPapers(core);
    const id = documents["bowler.pdf"]?.id as string;
    imported(await core.importReferences(join(folder, "library.bib")));
    await core.updateReference(id, { title: "My own title for it", volume: "", pages: "1–2" });

    // The export changes, and is imported again.
    const text = await readFile(join(folder, "library.bib"), "utf8");
    await writeFile(
      join(folder, "library.bib"),
      text.replace("BMC Public Health", "BMC Pub. Health"),
    );
    imported(await core.importReferences(join(folder, "library.bib")));

    const record = (await documentOf(core, id)).reference;
    expect(record?.fields).toMatchObject({
      title: "My own title for it",
      volume: "",
      pages: "1–2",
    });
    expect(record?.sources).toMatchObject({ title: "user", volume: "user", pages: "user" });
    // What nobody edited was updated.
    expect(record?.fields.containerTitle).toBe("BMC Pub. Health");
    expect(record?.sources.containerTitle).toBe("import");
  });

  test("a field Look up filled is kept too", async () => {
    const services = await startReferenceServices({
      crossref: { "10.5555/chinese-heat-island": recordedAnswer("crossref-chinese.json") },
    });
    onTestFinished(() => services.close());
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder(), {
      referenceLookup: { crossref: services.url, arxiv: services.url },
    });
    const { documents } = await addPapers(core);
    const id = documents["chinese.pdf"]?.id as string;
    core.on("consent.requested", (request) => void core.respondToConsent(request.requestId, true));
    await core.lookUpReference(id);
    // Crossref says 2020; the export says 2021.
    expect((await documentOf(core, id)).reference?.fields.issued).toBe("2020-01");

    imported(await core.importReferences(join(folder, "library.bib")));

    const record = (await documentOf(core, id)).reference;
    expect(record?.fields.issued).toBe("2020-01");
    expect(record?.sources.issued).toBe("lookup");
    expect(record?.fields.authors).toHaveLength(3);
    expect(record?.sources.authors).toBe("lookup");
  });

  test("importing again updates instead of repeating", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    await addPapers(core);

    const first = imported(await core.importReferences(join(folder, "library.bib")));
    const second = imported(await core.importReferences(join(folder, "library.bib")));
    expect(second).toMatchObject({ total: 5, matched: 4, unmatched: 1, changed: 0 });
    expect((await core.listUnmatchedReferences()).map((record) => record.id)).toEqual(
      first.entries.filter((entry) => entry.documentId === null).map((entry) => entry.recordId),
    );
    // The same works in another format are the same references.
    const third = imported(await core.importReferences(join(folder, "library.json")));
    expect(third).toMatchObject({ total: 5, matched: 4, unmatched: 1 });
    expect(await core.listUnmatchedReferences()).toHaveLength(1);
  });

  test("a changed export updates what an import gave", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    await core.importReferences(join(folder, "library.bib"));
    const text = await readFile(join(folder, "library.bib"), "utf8");
    await writeFile(
      join(folder, "library.bib"),
      text.replace("No Starch Press", "No Starch Press, Inc."),
    );

    const again = imported(await core.importReferences(join(folder, "library.bib")));
    expect(again.changed).toBe(1);
    const records = byTitle(await core.listUnmatchedReferences());
    expect(records["The Rust Programming Language"]?.fields.publisher).toBe(
      "No Starch Press, Inc.",
    );
  });

  test("two references never take the same Document", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    await addPapers(core);
    const path = join(folder, "twice.json");
    await writeFile(
      path,
      JSON.stringify([
        {
          id: "a",
          type: "book",
          title: "Reasoning about Uncertainty",
          issued: { "date-parts": [[2003]] },
        },
        {
          id: "b",
          type: "book",
          title: "Reasoning about Uncertainty",
          issued: { "date-parts": [[2012]] },
          publisher: "Second edition",
        },
      ]),
    );
    const result = imported(await core.importReferences(path));
    // Different years are different works: one matches the Document, one is kept apart.
    expect(result).toMatchObject({ total: 2, matched: 1, unmatched: 1 });
  });

  test("a different DOI is a different work, whatever the title says", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    await addPapers(core);
    const path = join(folder, "published.json");
    await writeFile(
      path,
      JSON.stringify([
        { id: "a", type: "article-journal", title: BOWLER_TITLE, DOI: "10.9999/another.version" },
      ]),
    );
    expect(imported(await core.importReferences(path))).toMatchObject({ matched: 0, unmatched: 1 });
  });
});

describe("Importing references: Undo, files and progress", () => {
  test("Undo takes an import back, but not what the User edited since", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    const { documents } = await addPapers(core);
    const halpern = documents["halpern.pdf"]?.id as string;
    const bowler = documents["bowler.pdf"]?.id as string;

    const result = imported(await core.importReferences(join(folder, "library.bib")));
    await core.updateReference(bowler, { containerTitle: "BMC Public Health (edited)" });
    await core.undoReferenceImport(result.importId);

    expect(await core.listUnmatchedReferences()).toEqual([]);
    expect((await documentOf(core, halpern)).reference?.fields.publisher).toBe("");
    const record = (await documentOf(core, bowler)).reference;
    expect(record?.fields.authors).toEqual([]);
    expect(record?.fields.containerTitle).toBe("BMC Public Health (edited)");
    await expect(core.undoReferenceImport(result.importId)).rejects.toThrow();
  });

  test("references.changed carries the Documents an import changed", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    const { documents } = await addPapers(core);
    const changed: string[] = [];
    core.on("references.changed", (list) => changed.push(...list.map((document) => document.id)));

    await core.importReferences(join(folder, "library.bib"));

    expect(new Set(changed)).toEqual(
      new Set(
        ["bowler.pdf", "chinese.pdf", "attention.pdf", "halpern.pdf"].map(
          (name) => documents[name]?.id,
        ),
      ),
    );
  });

  test("files the export attaches are offered, and added as Documents with their references", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());

    const result = imported(await core.importReferences(join(folder, "library.json")));
    // Only the PDF that exists is offered (the HTML snapshot in BibTeX's list isn't a Document type).
    expect(result.files).toBe(1);
    const added = await core.addImportedReferenceFiles(result.importId);
    expect(added.skipped).toEqual([]);
    expect(added.documents).toHaveLength(1);
    const [document] = await waitForDocuments(core, (documents) => {
      expect(documents.find((each) => each.name === "rust-book")?.status).toBe("ready");
    });
    expect(document).toBeDefined();

    const rust = (await core.listDocuments()).find((each) => each.name === "rust-book");
    expect(rust?.reference?.fields.title).toBe("The Rust Programming Language");
    expect(rust?.reference?.fields.publisher).toBe("No Starch Press");
    // Its reference is no longer without a Document.
    expect(
      (await core.listUnmatchedReferences()).map((record) => record.fields.title),
    ).not.toContain("The Rust Programming Language");
  });

  test("Add the PDF gives a reference its Document, and the file's own reading fills only what is empty", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    imported(await core.importReferences(join(folder, "library.bib")));
    const record = byTitle(await core.listUnmatchedReferences())["The Rust Programming Language"];
    expect(record).toBeDefined();
    const pdf = await writeSourceFile(
      await createSourceFolder(),
      "Rust.pdf",
      buildPdf([{ lines: ["Chapter 1"] }], {
        info: { Title: "Some other title", Author: "Someone" },
      }),
    );

    const document = await core.addReferenceFile((record as ReferenceRecord).id, pdf);
    await waitForDocuments(core, (documents) => {
      expect(documents.find((each) => each.id === document.id)?.status).toBe("ready");
    });

    // The Document has the record the export gave; its own reading left the export's title alone.
    const reference = (await documentOf(core, document.id)).reference;
    expect(reference?.id).toBe((record as ReferenceRecord).id);
    expect(reference?.fields.title).toBe("The Rust Programming Language");
    expect(reference?.sources.title).toBe("import");
    expect(reference?.fields.authors).toEqual([
      { family: "Klabnik", given: "Steve" },
      { family: "Nichols", given: "Carol" },
    ]);
    expect((await core.listUnmatchedReferences()).map((each) => each.id)).not.toContain(
      (record as ReferenceRecord).id,
    );
  });

  test("importing again links a reference whose Document was added since", async () => {
    const folder = await exportsFolder();
    const core = startCore(await createTempDataFolder());
    imported(await core.importReferences(join(folder, "library.bib")));
    expect(await core.listUnmatchedReferences()).toHaveLength(5);

    const { documents } = await addPapers(core);
    const again = imported(await core.importReferences(join(folder, "library.bib")));

    expect(again).toMatchObject({ matched: 4, unmatched: 1 });
    expect(await core.listUnmatchedReferences()).toHaveLength(1);
    expect(
      (await documentOf(core, documents["halpern.pdf"]?.id as string)).reference?.fields.publisher,
    ).toBe("MIT Press");
    // Undo puts the four loose references back.
    await core.undoReferenceImport(again.importId);
    expect(await core.listUnmatchedReferences()).toHaveLength(5);
  });

  test("a 500-item BibTeX file imports with progress, and says how many matched", async () => {
    const folder = await createSourceFolder();
    const core = startCore(await createTempDataFolder());
    const { documents } = await addPapers(core);
    const entries = Array.from({ length: 500 }, (_, index) =>
      [
        `@article{key${index},`,
        `  author = {Family${index}, Given},`,
        `  title = {A generated paper number ${index} on subject ${index * 7}},`,
        `  journal = {Journal ${index % 10}},`,
        `  year = {${1990 + (index % 30)}},`,
        `  doi = {10.7777/generated.${index}},`,
        "}",
      ].join("\n"),
    );
    // Five of them are the papers the User has.
    entries.push(
      "@article{b, title = {x}, doi = {10.1186/1471-2458-10-456}, author = {Bowler, Diana E.}, year = {2010}}",
    );
    const path = await writeSourceFile(folder, "big.bib", entries.join("\n\n"));
    const progress: { done: number; total: number }[] = [];
    core.on("references.importProgress", (each) => progress.push(each));

    const result = imported(await core.importReferences(path));

    expect(result).toMatchObject({ total: 501, matched: 1, unmatched: 500, problems: [] });
    expect(progress.length).toBeGreaterThan(2);
    expect(progress.at(-1)).toEqual({ done: 501, total: 501 });
    expect(progress.map((each) => each.done)).toEqual(
      [...progress.map((each) => each.done)].sort((a, b) => a - b),
    );
    expect(await core.listUnmatchedReferences()).toHaveLength(500);
    expect(documents["bowler.pdf"]).toBeDefined();
    expect(dirname(path)).toBe(folder);
  });
});
