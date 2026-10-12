import { constants } from "node:fs";
import { copyFile, link, mkdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { describe, expect, test } from "vitest";
import { DATABASE_FILE, type Document, type Folder, type WatchListener } from "../../src/core";
import { isInside, type StatIdentity } from "../../src/core/documents/files";
import { openDatabase } from "../../src/core/storage";
import { citationState } from "../../src/shared/citations";
import { askAndFinish, citingModel, onlyCitation, setUpWithDocuments } from "../helpers/citations";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import {
  addAndProcess,
  createSourceFolder,
  documentAt,
  linkAndProcess,
  sha256,
  waitForDocuments,
  writeSourceFile,
} from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { buildPdf } from "../helpers/pdf";

/*
 * A file moved and edited where IncarnaMind didn't see it is still its
 * Document (#389): on the same disk by its device, file number and creation
 * time; from another disk as the one clear candidate. Two different files
 * are never one Document. Other disks and file systems are stood in for by
 * the stat the core reads them with, since a test has one disk.
 */

const SPRING = "Spring tides happen at new moon and at full moon.";

/** Three pages: the quote is on page 2. */
const V1 = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", SPRING] },
  { lines: ["Tide tables", "Harbours publish the times of high water every year."] },
]);
/** Edited: page 2 rewritten, so the quote is in version 1 only. */
const V2 = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides follow the new and the full moon."] },
  { lines: ["Tide tables", "Harbours publish the times of high water every year."] },
]);

const NOTES = "# Notes\n\nRivers carry silt down to the delta.\n";
const PAPER = "# Paper\n\nNeap tides are the smallest of the month.\n";
const OTHER = "# Other\n\nEstuaries mix fresh water with the sea.\n";
const EDIT = "\nAdded later: the delta grows each year.\n";

type Identity = Awaited<ReturnType<StatIdentity>>;

/** The core's stat, with what `as` makes of each file's real identity. A file that isn't there isn't. */
const statAs =
  (as: (path: string, real: Identity) => Identity): StatIdentity =>
  async (path) => {
    const real = await stat(path, { bigint: true });
    return as(path, { dev: real.dev, ino: real.ino, birthtimeNs: real.birthtimeNs });
  };

/** The files in `drive` are on another disk: another device, and other file numbers. */
const onAnotherDisk = (drive: string) =>
  statAs((path, real) =>
    isInside(drive, path) ? { ...real, dev: real.dev + 1n, ino: real.ino + 1_000_000n } : real,
  );

const folderAt = (folders: readonly Folder[], relativePath: string) => {
  const folder = folders.find((each) => each.relativePath === relativePath);
  if (!folder) throw new Error(`No Folder at "${relativePath}".`);
  return folder;
};

const byId = (documents: readonly Document[], id: string | undefined) => {
  const document = documents.find((each) => each.id === id);
  if (!document) throw new Error(`No Document ${id}.`);
  return document;
};

/** The page rows kept of each version of a Document, by content hash. */
const pagesByVersion = (dataDir: string, documentId: string) =>
  Object.fromEntries(
    queryDatabase<{ content_hash: string; pages: number }>(
      dataDir,
      `SELECT content_hash, count(*) AS pages FROM document_pages
       WHERE document_id = ? AND deleted_at IS NULL GROUP BY content_hash`,
      [documentId],
    ).map((row) => [row.content_hash, row.pages]),
  );

/** A Linked folder holding Tides.pdf, cited on page 2 by an Answer, with a Tag and a name the User chose. */
async function citedInLinkedFolder() {
  const model = citingModel({
    query: "spring tides",
    records: (passages) => [
      {
        marker: 1,
        passage: passages.find((passage) => passage.text.includes(SPRING))?.id,
        pageFrom: 2,
        pageTo: 2,
        quote: SPRING,
      },
    ],
    answer: "Spring tides come at new and full moon [^1].",
  });
  const setup = await setUpWithDocuments(model, []);
  const { core, client, mind } = setup;
  const library = await createSourceFolder();
  const path = await writeSourceFile(library, "Tides.pdf", V1);
  await writeSourceFile(library, "Notes.md", NOTES);
  const documents = await linkAndProcess(core, library);
  const tides = documentAt(documents, path);
  const tag = await core.createTag({ name: "Oceans" });
  await core.addDocumentTag(tides.id, tag.id);
  await core.renameDocument(tides.id, "Tide tables");
  const { answerId } = await askAndFinish(core, client, mind.id, "When are spring tides?");
  const cited = onlyCitation(client, answerId);
  expect(cited).toMatchObject({ documentId: tides.id, contentHash: sha256(V1), check: "found" });
  return { ...setup, library, path, tides, tag, answerId, cited };
}

describe("A file moved and edited where IncarnaMind didn't see it", { timeout: 60_000 }, () => {
  test("while the app was closed, keeps its id, Folder, Tags, name, Citations and Search scopes, and the cited version's text stays", async () => {
    const { core, dataDir, mind, library, path, tides, tag, answerId, cited } =
      await citedInLinkedFolder();
    const ids = (await core.listDocuments()).map((each) => each.id).sort();
    await core.close();
    const moved = join(library, "Archive", "Tides.pdf");
    await mkdir(join(library, "Archive"));
    await rename(path, moved);
    await writeFile(moved, V2);

    const second = startCore(dataDir);
    await second.reconcileDocuments();
    const documents = await waitForDocuments(second, (current) => {
      expect(byId(current, tides.id)).toMatchObject({ contentHash: sha256(V2), status: "ready" });
    });

    expect(documents.map((each) => each.id).sort()).toEqual(ids);
    expect(documents.map((each) => each.fileStatus)).toEqual(["available", "available"]);
    expect(byId(documents, tides.id)).toMatchObject({
      path: moved,
      name: "Tide tables",
      folderId: folderAt(await second.listFolders(), "Archive").id,
      tags: expect.arrayContaining([expect.objectContaining({ tagId: tag.id, source: "user" })]),
    });
    // Search scopes naming it search its new version only.
    expect(
      await second.searchPassages("follow", { mode: "keyword", documentIds: [tides.id] }),
    ).toHaveLength(1);
    expect(await second.searchPassages("happen", { mode: "keyword" })).toEqual([]);
    // The Citation still quotes version 1, found, and says the Document changed after it.
    const client = await connectToMind(second, mind.id);
    const citation = onlyCitation(client, answerId);
    expect(citation).toEqual(cited);
    expect(citationState(citation, documents)).toEqual({
      check: "found",
      reason: null,
      documentId: tides.id,
      changedAfterCited: true,
    });
    await second.close();

    // The text it quotes outlives the clean-up at the next start.
    startCore(dataDir);
    expect(Object.keys(pagesByVersion(dataDir, tides.id)).sort()).toEqual(
      [sha256(V1), sha256(V2)].sort(),
    );
  });

  test("while the app runs, in a Linked folder and as a file added on its own", async () => {
    const library = await createSourceFolder();
    const downloads = await createSourceFolder();
    const desk = await createSourceFolder();
    const notesPath = await writeSourceFile(library, "A/notes.md", NOTES);
    const core = startCore(await createTempDataFolder());
    const [notes] = await linkAndProcess(core, library);
    const [paper] = await addAndProcess(core, [
      await writeSourceFile(downloads, "paper.md", PAPER),
      await writeSourceFile(desk, "desk.md", OTHER),
    ]);
    const tag = await core.createTag({ name: "Coast" });
    await core.addDocumentTag(notes?.id as string, tag.id);
    await core.addDocumentTag(paper?.id as string, tag.id);

    const movedNotes = join(library, "B", "notes.md");
    await mkdir(join(library, "B"));
    await rename(notesPath, movedNotes);
    await writeFile(movedNotes, `${NOTES}${EDIT}`);
    const movedPaper = join(desk, "paper.md");
    await rename(join(downloads, "paper.md"), movedPaper);
    await writeFile(movedPaper, `${PAPER}${EDIT}`);

    const documents = await waitForDocuments(core, (current) => {
      expect(byId(current, notes?.id)).toMatchObject({
        path: movedNotes,
        contentHash: sha256(`${NOTES}${EDIT}`),
        status: "ready",
      });
      expect(byId(current, paper?.id)).toMatchObject({
        path: movedPaper,
        contentHash: sha256(`${PAPER}${EDIT}`),
        status: "ready",
      });
    });
    expect(documents).toHaveLength(3);
    expect(documents.map((each) => each.fileStatus)).toEqual([
      "available",
      "available",
      "available",
    ]);
    expect(byId(documents, notes?.id).folderId).toBe(folderAt(await core.listFolders(), "B").id);
    for (const id of [notes?.id, paper?.id]) {
      expect(byId(documents, id).tags).toEqual([expect.objectContaining({ tagId: tag.id })]);
      expect(
        await core.searchPassages("grows", { mode: "keyword", documentIds: [id as string] }),
      ).toHaveLength(1);
    }
  });

  test("indexed before IncarnaMind knew files on their disks, a Document is known by its file once it is seen again", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const path = await writeSourceFile(library, "notes.md", NOTES);
    const first = startCore(dataDir);
    const [notes] = await linkAndProcess(first, library);
    await first.close();
    const db = openDatabase(join(dataDir, DATABASE_FILE));
    try {
      db.exec(`
        DROP INDEX documents_by_file_number;
        ALTER TABLE documents DROP COLUMN file_device;
        ALTER TABLE documents DROP COLUMN file_number;
        ALTER TABLE documents DROP COLUMN file_created_ns;
        DELETE FROM schema_migrations WHERE version = 68;
      `);
    } finally {
      db.close();
    }
    const second = startCore(dataDir);
    await second.reconcileDocuments();
    await second.close();

    const moved = join(library, "Sub", "notes.md");
    await mkdir(join(library, "Sub"));
    await rename(path, moved);
    await writeFile(moved, `${NOTES}${EDIT}`);
    const third = startCore(dataDir);
    await third.reconcileDocuments();

    const documents = await waitForDocuments(third, (current) => {
      expect(current).toEqual([
        expect.objectContaining({
          id: notes?.id,
          path: moved,
          contentHash: sha256(`${NOTES}${EDIT}`),
          status: "ready",
        }),
      ]);
    });
    expect(documents[0]?.fileStatus).toBe("available");
  });
});

describe("A file moved to another disk and edited", { timeout: 60_000 }, () => {
  test("is the Document it was when that is the one clear candidate", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const drive = await createSourceFolder();
    const path = await writeSourceFile(library, "notes.md", NOTES);
    await writeSourceFile(library, "other.md", OTHER);
    const linkedFolders = { stat: onAnotherDisk(drive) };
    const first = startCore(dataDir, { linkedFolders });
    const notes = documentAt(await linkAndProcess(first, library), path);
    const onDrive = await first.addLinkedFolder(drive);
    await first.close();
    const moved = join(drive, "notes.md");
    await rename(path, moved);
    await writeSourceFile(drive, "notes.md", `${NOTES}${EDIT}`);

    const second = startCore(dataDir, { linkedFolders });
    await second.reconcileDocuments();

    const documents = await waitForDocuments(second, (current) => {
      expect(byId(current, notes.id)).toMatchObject({
        path: moved,
        linkedFolderId: onDrive.id,
        contentHash: sha256(`${NOTES}${EDIT}`),
        fileStatus: "available",
        status: "ready",
      });
    });
    expect(documents).toHaveLength(2);
  });

  test("isn't matched when two Documents fit, two files have its name, or its name changed", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const drive = await createSourceFolder();
    const first = startCore(dataDir, { linkedFolders: { stat: onAnotherDisk(drive) } });
    const files = {
      a: await writeSourceFile(library, "A/notes.md", NOTES),
      b: await writeSourceFile(library, "B/notes.md", `${NOTES}More notes.\n`),
      summary: await writeSourceFile(library, "summary.md", PAPER),
      report: await writeSourceFile(library, "report.md", OTHER),
    };
    const before = await linkAndProcess(first, library);
    await first.addLinkedFolder(drive);
    await first.close();
    for (const file of Object.values(files)) await rm(file);
    // Two Documents named notes.md went; two files named summary.md came; report.md was renamed.
    const arrived = [
      await writeSourceFile(drive, "notes.md", `${NOTES}${EDIT}`),
      await writeSourceFile(drive, "X/summary.md", `${PAPER}${EDIT}`),
      await writeSourceFile(drive, "Y/summary.md", `${PAPER}${EDIT}Y.\n`),
      await writeSourceFile(drive, "report v2.md", `${OTHER}${EDIT}`),
    ];

    const second = startCore(dataDir, { linkedFolders: { stat: onAnotherDisk(drive) } });
    await second.reconcileDocuments();
    const documents = await waitForDocuments(second, (current) => {
      for (const path of arrived) expect(documentAt(current, path).status).toBe("ready");
    });

    const old = new Set(before.map((each) => each.id));
    for (const path of arrived) expect(old.has(documentAt(documents, path).id)).toBe(false);
    for (const document of before) {
      expect(byId(documents, document.id)).toMatchObject({
        path: document.path,
        fileStatus: "missing",
      });
    }
  });

  test("is matched to a Document found missing in the last pass, not to one missing for longer", async () => {
    const library = await createSourceFolder();
    const drive = await createSourceFolder();
    const notesPath = await writeSourceFile(library, "notes.md", NOTES);
    const otherPath = await writeSourceFile(library, "other.md", OTHER);
    const listeners = new Map<string, WatchListener>();
    const core = startCore(await createTempDataFolder(), {
      linkedFolders: {
        stat: onAnotherDisk(drive),
        // Reports only what the test says changed, one pass at a time.
        watch: (root, listener) => {
          listeners.set(root, listener);
          return { close: () => {} };
        },
      },
    });
    const before = await linkAndProcess(core, library);
    await linkAndProcess(core, drive);
    const report = (root: string, path: string) => listeners.get(root)?.onChange([path]);
    const notes = documentAt(before, notesPath);
    const other = documentAt(before, otherPath);

    await rm(notesPath);
    report(library, notesPath);
    await waitForDocuments(core, (current) => {
      expect(byId(current, notes.id).fileStatus).toBe("missing");
    });
    const movedNotes = await writeSourceFile(drive, "notes.md", `${NOTES}${EDIT}`);
    report(drive, movedNotes);
    await waitForDocuments(core, (current) => {
      expect(byId(current, notes.id)).toMatchObject({ path: movedNotes, status: "ready" });
    });

    await rm(otherPath);
    report(library, otherPath);
    await waitForDocuments(core, (current) => {
      expect(byId(current, other.id).fileStatus).toBe("missing");
    });
    const unrelated = await writeSourceFile(library, "unrelated.md", PAPER);
    report(library, unrelated);
    await waitForDocuments(core, (current) => {
      expect(documentAt(current, unrelated).status).toBe("ready");
    });
    const movedOther = await writeSourceFile(drive, "other.md", `${OTHER}${EDIT}`);
    report(drive, movedOther);
    const documents = await waitForDocuments(core, (current) => {
      expect(documentAt(current, movedOther).status).toBe("ready");
    });

    expect(documentAt(documents, movedOther).id).not.toBe(other.id);
    expect(byId(documents, other.id)).toMatchObject({ path: otherPath, fileStatus: "missing" });
  });
});

describe("Two different files are never one Document", { timeout: 60_000 }, () => {
  test("a new file given a deleted file's number is told apart by its creation time", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const deleted = await writeSourceFile(library, "notes.md", NOTES);
    const kept = await writeSourceFile(library, "kept.md", PAPER);
    const identities = new Map<string, Identity>([
      [deleted, { dev: 7n, ino: 42n, birthtimeNs: 1_000n }],
      [kept, { dev: 7n, ino: 43n, birthtimeNs: 1_000n }],
    ]);
    const linkedFolders = { stat: statAs((path, real) => identities.get(path) ?? real) };
    const first = startCore(dataDir, { linkedFolders });
    const before = await linkAndProcess(first, library);
    await first.close();
    await rm(deleted);
    // On the same disk: the new file gets the deleted one's number, at a later creation time.
    const reused = await writeSourceFile(library, "Sub/notes.md", `${NOTES}${EDIT}`);
    identities.set(reused, { dev: 7n, ino: 42n, birthtimeNs: 2_000n });
    // The control: moved and edited, keeping its number and creation time.
    const moved = join(library, "Sub", "kept.md");
    await rename(kept, moved);
    await writeFile(moved, `${PAPER}${EDIT}`);
    identities.set(moved, { dev: 7n, ino: 43n, birthtimeNs: 1_000n });

    const second = startCore(dataDir, { linkedFolders });
    await second.reconcileDocuments();
    const documents = await waitForDocuments(second, (current) => {
      expect(documentAt(current, reused).status).toBe("ready");
      expect(documentAt(current, moved).status).toBe("ready");
    });

    expect(documentAt(documents, moved).id).toBe(documentAt(before, kept).id);
    expect(documentAt(documents, reused).id).not.toBe(documentAt(before, deleted).id);
    expect(byId(documents, documentAt(before, deleted).id).fileStatus).toBe("missing");
  });

  test("a Save As copy, a Duplicate and a second link to a file are new Documents, even once the original is gone", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const report = await writeSourceFile(library, "report.md", NOTES);
    const paper = await writeSourceFile(library, "paper.md", PAPER);
    const tides = await writeSourceFile(library, "tides.md", OTHER);
    const first = startCore(dataDir);
    const before = await linkAndProcess(first, library);
    await first.close();
    // Save As into another folder, with changes, then the original deleted.
    const savedAs = await writeSourceFile(library, "Archive/report.md", `${NOTES}${EDIT}`);
    await rm(report);
    // Duplicated (cloned where the file system can), the copy edited, the original deleted.
    const duplicate = join(library, "paper copy.md");
    await copyFile(paper, duplicate, constants.COPYFILE_FICLONE);
    await writeFile(duplicate, `${PAPER}${EDIT}`);
    await rm(paper);
    // A second link to a file that stays.
    const linked = join(library, "Archive", "tides.md");
    await link(tides, linked);

    const second = startCore(dataDir);
    await second.reconcileDocuments();
    const documents = await waitForDocuments(second, (current) => {
      for (const path of [savedAs, duplicate, linked]) {
        expect(documentAt(current, path).status).toBe("ready");
      }
    });

    const old = new Set(before.map((each) => each.id));
    for (const path of [savedAs, duplicate, linked]) {
      expect(old.has(documentAt(documents, path).id)).toBe(false);
    }
    expect(byId(documents, documentAt(before, report).id)).toMatchObject({
      path: report,
      fileStatus: "missing",
    });
    expect(byId(documents, documentAt(before, paper).id)).toMatchObject({
      path: paper,
      fileStatus: "missing",
    });
    expect(byId(documents, documentAt(before, tides).id)).toMatchObject({
      path: tides,
      fileStatus: "available",
    });
  });
});

describe("Files on any system", { timeout: 60_000 }, () => {
  test("a file system without file numbers or creation times falls back to the content alone", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const linkedFolders = {
      stat: statAs((path, real) => ({
        dev: real.dev,
        ino: basename(path).startsWith("unnumbered") ? 0n : real.ino,
        birthtimeNs: basename(path).startsWith("untimed") ? 0n : real.birthtimeNs,
      })),
    };
    const paths = {
      unnumbered: await writeSourceFile(library, "unnumbered.md", NOTES),
      untimed: await writeSourceFile(library, "untimed.md", PAPER),
      unedited: await writeSourceFile(library, "unnumbered unedited.md", OTHER),
    };
    const first = startCore(dataDir, { linkedFolders });
    const before = await linkAndProcess(first, library);
    await first.close();
    await mkdir(join(library, "Sub"));
    const moved = {
      unnumbered: join(library, "Sub", "unnumbered.md"),
      untimed: join(library, "Sub", "untimed.md"),
      unedited: join(library, "Sub", "unnumbered unedited.md"),
    };
    for (const key of ["unnumbered", "untimed", "unedited"] as const) {
      await rename(paths[key], moved[key]);
    }
    await writeFile(moved.unnumbered, `${NOTES}${EDIT}`);
    await writeFile(moved.untimed, `${PAPER}${EDIT}`);

    const second = startCore(dataDir, { linkedFolders });
    await second.reconcileDocuments();
    const documents = await waitForDocuments(second, (current) => {
      for (const path of Object.values(moved)) {
        expect(documentAt(current, path).status).toBe("ready");
      }
    });

    // Moved without changes, still found by its content.
    expect(documentAt(documents, moved.unedited).id).toBe(documentAt(before, paths.unedited).id);
    // Moved and edited: a new Document, as before IncarnaMind read file numbers.
    for (const key of ["unnumbered", "untimed"] as const) {
      const old = documentAt(before, paths[key]).id;
      expect(documentAt(documents, moved[key]).id).not.toBe(old);
      expect(byId(documents, old).fileStatus).toBe("missing");
    }
  });

  test("file numbers too big for a JavaScript number, as Windows' are, are compared exactly", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    // 2^60 and up: neighbouring numbers are one JavaScript number. The creation times are alike.
    const linkedFolders = {
      stat: statAs((_path, real) => ({
        dev: real.dev,
        ino: (1n << 60n) + real.ino,
        birthtimeNs: 1_790_000_000_000_000_000n,
      })),
    };
    const deleted = await writeSourceFile(library, "a.md", NOTES);
    const kept = await writeSourceFile(library, "b.md", PAPER);
    const first = startCore(dataDir, { linkedFolders });
    const before = await linkAndProcess(first, library);
    await first.close();
    await rm(deleted);
    const moved = join(library, "Sub", "b.md");
    await mkdir(join(library, "Sub"));
    await rename(kept, moved);
    await writeFile(moved, `${PAPER}${EDIT}`);

    const second = startCore(dataDir, { linkedFolders });
    await second.reconcileDocuments();
    const documents = await waitForDocuments(second, (current) => {
      expect(documentAt(current, moved).status).toBe("ready");
    });

    expect(documentAt(documents, moved).id).toBe(documentAt(before, kept).id);
    expect(byId(documents, documentAt(before, deleted).id).fileStatus).toBe("missing");
  });
});
