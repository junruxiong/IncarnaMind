import type { ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { chmod, lstat, mkdir, open, readdir, readFile, rename, rm } from "node:fs/promises";
import { basename, join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  type Core,
  type Document,
  type Folder,
  InvalidInputError,
  NotFoundError,
  type WatchListener,
} from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import {
  addAndProcess,
  createSourceFolder,
  documentAt,
  linkAndProcess,
  sha256,
  waitForDocuments,
  waitForProcessing,
  writeSourceFile,
} from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

const TIDES = "Spring tides happen at new moon and at full moon.\n";
const NEAP = "Neap tides are the smallest of the month, at the quarter moons.\n";
const RIVERS = "Rivers carry silt down to the delta.\n";

/** The Folder at a relative path in a Linked folder. */
const folderAt = (folders: readonly Folder[], relativePath: string) => {
  const folder = folders.find((each) => each.relativePath === relativePath);
  if (!folder) throw new Error(`No Folder at "${relativePath}".`);
  return folder;
};

const pathsOf = (documents: readonly Document[]) => documents.map((each) => each.path).sort();

/** Every file and folder under `root`, with its size, modified time and content hash. */
async function snapshot(root: string): Promise<string[]> {
  const entries: string[] = [];
  const visit = async (folder: string) => {
    for (const name of (await readdir(folder)).sort()) {
      const path = join(folder, name);
      const info = await lstat(path);
      if (info.isDirectory()) {
        entries.push(`${path}/ ${info.mtimeMs}`);
        await visit(path);
      } else {
        const hash = createHash("sha256")
          .update(await readFile(path))
          .digest("hex");
        entries.push(`${path} ${info.size} ${info.mtimeMs} ${info.mode} ${hash}`);
      }
    }
  };
  await visit(root);
  return entries;
}

describe("Linking a folder", { timeout: 30_000 }, () => {
  test("indexes every supported file in it, at any depth, where it is, with its folders as Folders", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const top = await writeSourceFile(library, "Top.md", `# Top\n\n${TIDES}`);
    const deep = await writeSourceFile(library, "Papers/2026/Deep.txt", NEAP);
    const shallow = await writeSourceFile(library, "Papers/Shallow.md", RIVERS);
    // Left out: unsupported, hidden, .git and node_modules.
    await writeSourceFile(library, "Papers/figure.png", "not a Document");
    await writeSourceFile(library, ".hidden/Secret.md", "hidden");
    await writeSourceFile(library, "Papers/.draft.md", "a hidden file");
    await writeSourceFile(library, ".git/notes.txt", "git");
    await writeSourceFile(library, "code/node_modules/pkg/README.md", "a package");
    const core = startCore(dataDir);

    const linked = await core.addLinkedFolder(library);
    expect(linked).toMatchObject({
      path: library,
      status: "scanning",
      folderId: expect.any(String),
    });
    const documents = await linkAndProcess(core, library);

    expect(pathsOf(documents)).toEqual([deep, shallow, top].sort());
    for (const document of documents) {
      expect(document).toMatchObject({
        linkedFolderId: linked.id,
        fileStatus: "available",
        status: "ready",
      });
    }
    expect(documentAt(documents, deep)).toMatchObject({ name: "Deep", contentHash: sha256(NEAP) });
    // Its folders are Folders, as on disk; only those holding a Document.
    const folders = await core.listFolders();
    expect(folders.map((folder) => folder.relativePath).sort()).toEqual([
      "",
      "Papers",
      "Papers/2026",
    ]);
    const root = folderAt(folders, "");
    const papers = folderAt(folders, "Papers");
    const year = folderAt(folders, "Papers/2026");
    expect(root).toMatchObject({ id: linked.folderId, name: basename(library), parentId: null });
    expect(papers).toMatchObject({ name: "Papers", parentId: root.id, linkedFolderId: linked.id });
    expect(year).toMatchObject({ name: "2026", parentId: papers.id });
    for (const folder of folders) expect(folder.id).toMatch(UUID);
    expect(documentAt(documents, top).folderId).toBe(root.id);
    expect(documentAt(documents, deep).folderId).toBe(year.id);
    const inPapers = await core.listDocuments({ folderId: papers.id, includeSubfolders: true });
    expect(pathsOf(inPapers)).toEqual([deep, shallow].sort());
    expect(await core.listDocuments({ linkedFolderId: null })).toEqual([]);
    expect(pathsOf(await core.listDocuments({ linkedFolderId: linked.id }))).toHaveLength(3);
    // Nothing is copied into the data folder.
    expect(existsSync(join(dataDir, "documents"))).toBe(false);
    expect(await core.listLinkedFolders()).toEqual([
      expect.objectContaining({
        id: linked.id,
        status: "watching",
        layout: "tree",
        progress: { files: 3, indexed: 3 },
        onlineOnly: { files: 0, bytes: 0, downloading: false },
      }),
    ]);
    expect((await core.searchPassages("Neap", { mode: "keyword" }))[0]?.documentId).toBe(
      documentAt(documents, deep).id,
    );
  });

  test("a Folder's id is derived from its Linked folder and path, so it is the same after a restart", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    await writeSourceFile(library, "A/B/notes.md", TIDES);
    const first = startCore(dataDir);
    await linkAndProcess(first, library);
    const before = await first.listFolders();
    first.close();

    const second = startCore(dataDir);
    await second.reconcileDocuments();

    expect(await second.listFolders()).toEqual(before);
  });

  test("a preview counts the files and bytes, and guesses the time and the layout, without indexing anything", async () => {
    const library = await createSourceFolder();
    // Zotero's storage: one folder per item, each with one file (and its hidden cache).
    const pdf = buildPdf([{ lines: ["A paper about tides."] }]);
    for (const key of ["ABCD1234", "EFGH5678", "IJKL9012", "MNOP3456"]) {
      await writeSourceFile(library, `storage/${key}/paper.pdf`, pdf);
      await writeSourceFile(library, `storage/${key}/.zotero-ft-cache`, "cache");
    }
    const core = startCore(await createTempDataFolder());

    const preview = await core.previewLinkedFolder(join(library, "storage"));

    expect(preview).toEqual({
      path: join(library, "storage"),
      files: 4,
      bytes: pdf.byteLength * 4,
      onlineOnly: { files: 0, bytes: 0 },
      estimatedSeconds: expect.any(Number),
      layout: "flat",
      insideLinkedFolderId: null,
      containsLinkedFolderIds: [],
    });
    expect(preview.estimatedSeconds).toBeGreaterThan(0);
    expect(await core.listDocuments()).toEqual([]);
    expect(await core.listLinkedFolders()).toEqual([]);

    // Linked, it gets that layout, and the User can change it.
    const linked = await core.addLinkedFolder(join(library, "storage"));
    await core.reconcileDocuments();
    expect((await core.listLinkedFolders())[0]?.layout).toBe("flat");
    expect(await core.setLinkedFolderLayout(linked.id, "tree")).toMatchObject({ layout: "tree" });
    await expect(core.setLinkedFolderLayout(linked.id, "grid" as never)).rejects.toThrow(
      InvalidInputError,
    );
    await expect(core.previewLinkedFolder(join(library, "nowhere"))).rejects.toThrow(NotFoundError);
    await expect(core.addLinkedFolder("relative/path")).rejects.toThrow(InvalidInputError);
  });

  test("a folder inside a Linked folder is linked already; one around Linked folders takes them in, their Documents keeping their ids", async () => {
    const outer = await createSourceFolder();
    const inner = join(outer, "Inner");
    await writeSourceFile(inner, "a.md", TIDES);
    await writeSourceFile(outer, "b.md", NEAP);
    const core = startCore(await createTempDataFolder());
    const [a] = await linkAndProcess(core, inner);
    const innerFolder = (await core.listLinkedFolders())[0];
    if (!a || !innerFolder) throw new Error("Nothing was linked.");

    // Inside: nothing new.
    await mkdir(join(inner, "Deeper"));
    expect((await core.addLinkedFolder(join(inner, "Deeper"))).id).toBe(innerFolder.id);
    expect(await core.previewLinkedFolder(inner)).toMatchObject({
      insideLinkedFolderId: innerFolder.id,
    });
    expect(await core.previewLinkedFolder(outer)).toMatchObject({
      containsLinkedFolderIds: [innerFolder.id],
    });

    // Around: merged into one.
    const documents = await linkAndProcess(core, outer);
    const linked = await core.listLinkedFolders();
    expect(linked.map((each) => each.path)).toEqual([outer]);
    const outerId = linked[0]?.id;
    expect(documentAt(documents, a.path)).toMatchObject({ id: a.id, linkedFolderId: outerId });
    const folders = await core.listFolders();
    expect(folders.map((folder) => folder.relativePath).sort()).toEqual(["", "Inner"]);
    expect(documentAt(documents, a.path).folderId).toBe(folderAt(folders, "Inner").id);
  });

  test("a file added on its own is in no Folder, and becomes a Linked folder's Document, keeping its id, once it is in one", async () => {
    const library = await createSourceFolder();
    const elsewhere = await createSourceFolder();
    const core = startCore(await createTempDataFolder());
    const [single] = await addAndProcess(core, [await writeSourceFile(library, "notes.md", TIDES)]);
    if (!single) throw new Error("Nothing was added.");
    expect(single).toMatchObject({ linkedFolderId: null, folderId: null, fileStatus: "available" });
    expect(await core.listDocuments({ linkedFolderId: null })).toEqual([single]);

    const documents = await linkAndProcess(core, library);

    const linked = (await core.listLinkedFolders())[0];
    expect(documents).toEqual([
      expect.objectContaining({
        id: single.id,
        linkedFolderId: linked?.id,
        folderId: linked?.folderId,
      }),
    ]);
    expect(await core.listDocuments({ linkedFolderId: null })).toEqual([]);
    // Added on its own from inside a Linked folder, a file is that folder's Document.
    const other = await writeSourceFile(elsewhere, "other.md", NEAP);
    const inside = await writeSourceFile(library, "Sub/inside.md", RIVERS);
    const { documents: added } = await core.addDocuments([inside, other]);
    expect(added.map((each) => each.linkedFolderId)).toEqual([linked?.id, null]);
  });

  test("unlinking a folder removes its Documents and Folders from the index", async () => {
    const library = await createSourceFolder();
    await writeSourceFile(library, "A/a.md", TIDES);
    await writeSourceFile(library, "b.md", NEAP);
    const core = startCore(await createTempDataFolder());
    const documents = await linkAndProcess(core, library);
    const linked = (await core.listLinkedFolders())[0];
    if (!linked) throw new Error("Nothing was linked.");
    const removed: string[][] = [];
    core.on("documents.removed", (ids) => removed.push(ids));

    await core.removeLinkedFolder(linked.id);

    expect(await core.listDocuments()).toEqual([]);
    expect(await core.listFolders()).toEqual([]);
    expect(await core.listLinkedFolders()).toEqual([]);
    expect(await core.searchPassages("tides", { mode: "keyword" })).toEqual([]);
    expect(removed.flat().sort()).toEqual(documents.map((each) => each.id).sort());
    await expect(core.removeLinkedFolder(linked.id)).rejects.toThrow(NotFoundError);
    // Linked again, its files are indexed again.
    expect(pathsOf(await linkAndProcess(core, library))).toEqual(pathsOf(documents));
  });

  test("linking, reconciling, restarting and unlinking leave the folder byte for byte as it was", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    await writeSourceFile(library, "Report.pdf", buildPdf([{ lines: ["Quarterly tides"] }]));
    await writeSourceFile(library, "Notes/notes.md", TIDES);
    await writeSourceFile(library, "Notes/.hidden.md", "hidden");
    await writeSourceFile(library, "Notes/data.csv", "a,b\n1,2\n");
    await writeSourceFile(library, ".git/config", "[core]\n");
    const before = await snapshot(library);

    const first = startCore(dataDir);
    await linkAndProcess(first, library);
    await first.reconcileDocuments();
    const opened = await first.openDocumentFile((await first.listDocuments())[0]?.id as string);
    await opened.stream.cancel();
    first.close();
    const second = startCore(dataDir);
    await second.reconcileDocuments();
    const linked = (await second.listLinkedFolders())[0];
    await second.removeLinkedFolder(linked?.id as string);
    second.close();

    expect(await snapshot(library)).toEqual(before);
  });
});

describe("While the app runs, a Linked folder is watched", { timeout: 30_000 }, () => {
  test("a new file is indexed, an edited one indexed again as a new version, and a deleted one goes missing", async () => {
    const library = await createSourceFolder();
    const tides = await writeSourceFile(library, "tides.md", TIDES);
    const rivers = await writeSourceFile(library, "Sub/rivers.md", RIVERS);
    const core = startCore(await createTempDataFolder());
    const linked = await linkAndProcess(core, library);
    const before = { tides: documentAt(linked, tides), rivers: documentAt(linked, rivers) };

    // New.
    const neap = await writeSourceFile(library, "Sub/neap.md", NEAP);
    await waitForDocuments(core, (documents) => {
      expect(documentAt(documents, neap)).toMatchObject({
        status: "ready",
        contentHash: sha256(NEAP),
      });
    });
    // Edited: a new version, processed, replaces the old one in search.
    const edited = `${TIDES}Perigean spring tides are the highest.\n`;
    await writeSourceFile(library, "tides.md", edited);
    await waitForDocuments(core, (documents) => {
      expect(documentAt(documents, tides)).toMatchObject({
        id: before.tides.id,
        status: "ready",
        contentHash: sha256(edited),
      });
    });
    expect(await core.searchPassages("Perigean", { mode: "keyword" })).toHaveLength(1);
    // Deleted: missing, kept, left out of new searches.
    await rm(rivers);
    const afterDelete = await waitForDocuments(core, (documents) => {
      expect(documentAt(documents, rivers).fileStatus).toBe("missing");
    });
    expect(documentAt(afterDelete, rivers)).toMatchObject({
      id: before.rivers.id,
      status: "ready",
    });
    expect(await core.searchPassages("silt", { mode: "keyword" })).toEqual([]);
    expect(
      await core.searchPassages("silt", { mode: "keyword", documentIds: [before.rivers.id] }),
    ).toEqual([]);
    expect((await core.readDocumentText(before.rivers.id)).pages).toEqual([
      { page: 1, text: RIVERS.trim(), kind: "section", label: { path: [] } },
    ]);
    await expect(core.openDocumentFile(before.rivers.id)).rejects.toThrow(NotFoundError);
    // Back with the same content: not missing any more.
    await writeSourceFile(library, "Sub/rivers.md", RIVERS);
    await waitForDocuments(core, (documents) => {
      expect(documentAt(documents, rivers)).toMatchObject({
        id: before.rivers.id,
        fileStatus: "available",
      });
    });
    expect(await core.searchPassages("silt", { mode: "keyword" })).toHaveLength(1);
  });

  test("when a watcher fails or overflows, the whole folder is reconciled", async () => {
    const library = await createSourceFolder();
    await writeSourceFile(library, "a.md", TIDES);
    const listeners: WatchListener[] = [];
    // A watcher that reports nothing until it fails.
    const core = startCore(await createTempDataFolder(), {
      linkedFolders: {
        watch: (_root, listener) => {
          listeners.push(listener);
          return { close: () => {} };
        },
      },
    });
    await linkAndProcess(core, library);
    const missed = await writeSourceFile(library, "b.md", NEAP);
    await core
      .listDocuments()
      .then((documents) => expect(pathsOf(documents)).not.toContain(missed));

    listeners.at(-1)?.onError(new Error("Too many changes: events were dropped."));

    await waitForDocuments(core, (documents) => {
      expect(documentAt(documents, missed).status).toBe("ready");
    });
  });
});

describe("Reconciling at a start", { timeout: 30_000 }, () => {
  test("changes made while the app was closed are found: new, edited and deleted files", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const tides = await writeSourceFile(library, "tides.md", TIDES);
    const rivers = await writeSourceFile(library, "rivers.md", RIVERS);
    const first = startCore(dataDir);
    const before = await linkAndProcess(first, library);
    first.close();

    const neap = await writeSourceFile(library, "Sub/neap.md", NEAP);
    const edited = `${TIDES}King tides flood the quay.\n`;
    await writeSourceFile(library, "tides.md", edited);
    await rm(rivers);
    const second = startCore(dataDir);
    await second.reconcileDocuments();
    const after = await waitForProcessing(
      second,
      (await second.listDocuments()).map((each) => each.id),
    );

    expect(documentAt(after, tides)).toMatchObject({
      id: documentAt(before, tides).id,
      contentHash: sha256(edited),
      status: "ready",
    });
    expect(documentAt(after, rivers)).toMatchObject({
      id: documentAt(before, rivers).id,
      fileStatus: "missing",
    });
    expect(documentAt(after, neap).status).toBe("ready");
    expect(await second.searchPassages("King", { mode: "keyword" })).toHaveLength(1);
  });

  test("a file whose size and modified time are unchanged isn't read again", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    await writeSourceFile(library, "tides.md", TIDES);
    const first = startCore(dataDir);
    const [before] = await linkAndProcess(first, library);
    first.close();

    const second = startCore(dataDir);
    const seen: Document[] = [];
    second.on("document.status", (document) => seen.push(document));
    await second.reconcileDocuments();

    expect(seen).toEqual([]);
    expect(await second.listDocuments()).toEqual([before]);
  });

  test("files added on their own are missing when gone from their folder, unavailable when their folder is gone too", async () => {
    const dataDir = await createTempDataFolder();
    const kept = await createSourceFolder();
    const gone = await createSourceFolder();
    const first = startCore(dataDir);
    const [x, y] = await addAndProcess(first, [
      await writeSourceFile(kept, "x.md", TIDES),
      await writeSourceFile(gone, "y.md", NEAP),
    ]);
    first.close();
    await rm(join(kept, "x.md"));
    await rm(gone, { recursive: true });

    const second = startCore(dataDir);
    await second.reconcileDocuments();

    const documents = await second.listDocuments();
    expect(documents.find((each) => each.id === x?.id)?.fileStatus).toBe("missing");
    expect(documents.find((each) => each.id === y?.id)?.fileStatus).toBe("unavailable");
    // An unavailable Document stays searchable; a missing one doesn't.
    expect(
      (await second.searchPassages("tides", { mode: "keyword" })).map((each) => each.documentId),
    ).toEqual([y?.id]);
  });

  test("a Linked folder that can't be reached keeps its Documents searchable, and comes back by itself", async () => {
    const dataDir = await createTempDataFolder();
    const drive = await createSourceFolder();
    const library = join(drive, "Library");
    await writeSourceFile(library, "tides.md", TIDES);
    const first = startCore(dataDir);
    const [tides] = await linkAndProcess(first, library);
    first.close();
    // The drive is unplugged.
    await rename(library, join(drive, "Elsewhere"));

    const second = startCore(dataDir);
    await second.reconcileDocuments();

    expect(await second.listDocuments()).toEqual([
      expect.objectContaining({ id: tides?.id, fileStatus: "unavailable", status: "ready" }),
    ]);
    expect((await second.listLinkedFolders())[0]?.status).toBe("unavailable");
    expect(await second.searchPassages("tides", { mode: "keyword" })).toHaveLength(1);
    await expect(second.openDocumentFile(tides?.id as string)).rejects.toThrow(NotFoundError);

    // Plugged in again: found by the next try.
    await rename(join(drive, "Elsewhere"), library);
    await waitForDocuments(second, (documents) => {
      expect(documents[0]?.fileStatus).toBe("available");
    });
    await second.reconcileDocuments();
    expect((await second.listLinkedFolders())[0]?.status).toBe("watching");
  });
});

describe("Moves and renames", { timeout: 30_000 }, () => {
  test("a file moved or renamed is the same Document: its id, Tags and name follow it", async () => {
    const library = await createSourceFolder();
    const original = await writeSourceFile(library, "Papers/old name.md", TIDES);
    const core = startCore(await createTempDataFolder());
    const [document] = await linkAndProcess(core, library);
    if (!document) throw new Error("Nothing was linked.");
    const tag = await core.createTag({ name: "Oceans" });
    await core.addDocumentTag(document.id, tag.id);
    const moves: Document[][] = [];
    core.on("documents.moved", (moved) => moves.push(moved));

    const moved = join(library, "Archive", "new name.md");
    await mkdir(join(library, "Archive"));
    await rename(original, moved);

    const after = await waitForDocuments(core, (documents) => {
      expect(documents).toEqual([expect.objectContaining({ id: document.id, path: moved })]);
    });
    const folders = await core.listFolders();
    expect(after[0]).toMatchObject({
      name: "new name",
      fileStatus: "available",
      folderId: folderAt(folders, "Archive").id,
      tags: [expect.objectContaining({ tagId: tag.id })],
      status: "ready",
    });
    // The Papers Folder held nothing else: it has gone.
    expect(folders.map((folder) => folder.relativePath).sort()).toEqual(["", "Archive"]);
    expect(moves.flat().map((each) => each.id)).toContain(document.id);
    // Its Passages, and so its Citations and Search scopes, follow it.
    expect(
      (await core.searchPassages("Spring", { documentIds: [document.id] }))[0]?.documentId,
    ).toBe(document.id);
    expect(
      await core.recheckCitation({
        documentId: document.id,
        quote: "Spring tides happen at new moon",
        pageFrom: null,
        pageTo: null,
      }),
    ).toMatchObject({ check: "found", contentHash: document.contentHash });

    // A name the User chose stays when the file is renamed.
    await core.renameDocument(document.id, "Tide notes");
    const renamed = join(library, "Archive", "renamed again.md");
    await rename(moved, renamed);
    await waitForDocuments(core, (documents) => {
      expect(documents).toEqual([
        expect.objectContaining({ id: document.id, path: renamed, name: "Tide notes" }),
      ]);
    });
  });

  test("a file moved while the app was closed is found again by its content", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    await writeSourceFile(library, "A/tides.md", TIDES);
    const first = startCore(dataDir);
    const [document] = await linkAndProcess(first, library);
    first.close();
    await rename(join(library, "A"), join(library, "B"));

    const second = startCore(dataDir);
    await second.reconcileDocuments();

    expect(await second.listDocuments()).toEqual([
      expect.objectContaining({
        id: document?.id,
        path: join(library, "B", "tides.md"),
        fileStatus: "available",
      }),
    ]);
  });

  test("a file added on its own and moved into a Linked folder is the same Document", async () => {
    const library = await createSourceFolder();
    const downloads = await createSourceFolder();
    const core = startCore(await createTempDataFolder());
    await linkAndProcess(core, library);
    const [single] = await addAndProcess(core, [
      await writeSourceFile(downloads, "paper.md", NEAP),
    ]);

    await rename(join(downloads, "paper.md"), join(library, "paper.md"));

    await waitForDocuments(core, (documents) => {
      expect(documents).toEqual([
        expect.objectContaining({
          id: single?.id,
          path: join(library, "paper.md"),
          linkedFolderId: expect.any(String),
        }),
      ]);
    });
  });
});

describe("Duplicates, removals and pausing", { timeout: 30_000 }, () => {
  test("the same file in two places is two Documents, and search shows each Passage once, from the copy in the scope", async () => {
    const library = await createSourceFolder();
    const first = await writeSourceFile(library, "A/notes.md", TIDES);
    const second = await writeSourceFile(library, "B/copy of notes.md", TIDES);
    const core = startCore(await createTempDataFolder());

    const documents = await linkAndProcess(core, library);

    expect(documents).toHaveLength(2);
    const [a, b] = [documentAt(documents, first), documentAt(documents, second)];
    expect(a.contentHash).toBe(b.contentHash);
    const all = await core.searchPassages("Spring tides");
    expect(all).toHaveLength(1);
    expect(
      (await core.searchPassages("Spring tides", { documentIds: [b.id] }))[0]?.documentId,
    ).toBe(b.id);
    expect(
      (await core.searchPassages("Spring tides", { documentIds: [a.id] }))[0]?.documentId,
    ).toBe(a.id);
  });

  test("a Document the User removes from a Linked folder stays out of the index while its file is there", async () => {
    const dataDir = await createTempDataFolder();
    const library = await createSourceFolder();
    const unwanted = await writeSourceFile(library, "unwanted.md", NEAP);
    await writeSourceFile(library, "wanted.md", TIDES);
    const first = startCore(dataDir);
    const documents = await linkAndProcess(first, library);

    await first.deleteDocument(documentAt(documents, unwanted).id);
    await first.reconcileDocuments();
    expect(pathsOf(await first.listDocuments())).not.toContain(unwanted);
    first.close();
    const second = startCore(dataDir);
    await second.reconcileDocuments();

    expect(pathsOf(await second.listDocuments())).toEqual([join(library, "wanted.md")]);
    // Its file wasn't touched.
    expect(await readFile(unwanted, "utf8")).toBe(NEAP);
  });

  test("pausing a Linked folder holds its indexing until it is resumed", async () => {
    const library = await createSourceFolder();
    await writeSourceFile(library, "a.md", TIDES);
    const core = startCore(await createTempDataFolder());
    await linkAndProcess(core, library);
    const linked = (await core.listLinkedFolders())[0];
    if (!linked) throw new Error("Nothing was linked.");

    expect(await core.setLinkedFolderPaused(linked.id, true)).toMatchObject({ status: "paused" });
    const later = await writeSourceFile(library, "b.md", NEAP);
    await core.reconcileDocuments();
    expect(pathsOf(await core.listDocuments())).not.toContain(later);

    expect(await core.setLinkedFolderPaused(linked.id, false)).toMatchObject({
      status: "scanning",
    });
    await core.reconcileDocuments();
    await waitForDocuments(core, (documents) => {
      expect(documentAt(documents, later).status).toBe("ready");
    });
    expect((await core.listLinkedFolders())[0]).toMatchObject({
      status: "watching",
      progress: { files: 2, indexed: 2 },
    });
    await expect(core.setLinkedFolderPaused(linked.id, "yes" as never)).rejects.toThrow(
      InvalidInputError,
    );
  });
});

describe("Files that can't be read now", { timeout: 30_000 }, () => {
  test.skipIf(process.platform === "win32" || process.getuid?.() === 0)(
    "a file that can't be read waits, and is indexed once it can be",
    async () => {
      const library = await createSourceFolder();
      await writeSourceFile(library, "open.md", TIDES);
      const locked = await writeSourceFile(library, "locked.md", NEAP);
      await chmod(locked, 0o000);
      const core = startCore(await createTempDataFolder());

      const documents = await linkAndProcess(core, library);
      expect(pathsOf(documents)).toEqual([join(library, "open.md")]);

      await chmod(locked, 0o644);
      await waitForDocuments(core, (listed) => {
        expect(documentAt(listed, locked).status).toBe("ready");
      });
    },
  );

  test("online-only files are counted, never read, until the User asks to download them", async (context) => {
    const library = await createSourceFolder();
    await writeSourceFile(library, "local.md", TIDES);
    // A file whose size is set but none of whose blocks are stored: a dataless placeholder.
    const placeholder = join(library, "Online.txt");
    const handle = await open(placeholder, "w");
    await handle.truncate(40_000);
    await handle.close();
    if ((await lstat(placeholder)).blocks !== 0) context.skip(); // no sparse files here
    // iCloud Drive's stub for a file that isn't downloaded.
    await writeSourceFile(library, ".Stubbed.md.icloud", "plist");
    const commands: string[][] = [];
    const core = startCore(await createTempDataFolder(), {
      linkedFolders: { detectDatalessFiles: true },
      // iCloud Drive's command-line tool, which downloads a stub's file on macOS.
      processes: {
        spawn: async (command, args) => {
          commands.push([command, ...args]);
          const child = new EventEmitter() as ChildProcess;
          queueMicrotask(() => child.emit("exit", 0));
          return child;
        },
      },
    });

    expect(await core.previewLinkedFolder(library)).toMatchObject({
      files: 1,
      onlineOnly: { files: 2, bytes: 40_000 },
    });
    const documents = await linkAndProcess(core, library);
    expect(pathsOf(documents)).toEqual([join(library, "local.md")]);
    const linked = (await core.listLinkedFolders())[0];
    expect(linked?.onlineOnly).toEqual({ files: 2, bytes: 40_000, downloading: false });

    await core.downloadOnlineOnlyFiles(linked?.id as string);
    await core.reconcileDocuments();

    // Read now (and downloaded, for a real placeholder). The stub's file isn't there yet:
    // on macOS, iCloud Drive was asked to download it.
    expect(pathsOf(await core.listDocuments())).toContain(placeholder);
    expect((await core.listLinkedFolders())[0]?.onlineOnly).toMatchObject({ files: 1 });
    if (process.platform === "darwin") {
      expect(commands).toEqual([["brctl", "download", join(library, "Stubbed.md")]]);
    }
  });
});

describe("Documents' files", { timeout: 30_000 }, () => {
  test("open in the default app and show in the folder through the shell, at the file's real path", async () => {
    const library = await createSourceFolder();
    const path = await writeSourceFile(library, "Report.md", TIDES);
    const opened: string[] = [];
    const shown: string[] = [];
    const core: Core = startCore(await createTempDataFolder(), {
      shell: {
        openPath: async (each) => {
          opened.push(each);
        },
        showItemInFolder: (each) => shown.push(each),
      },
    });
    const [document] = await linkAndProcess(core, library);
    if (!document) throw new Error("Nothing was linked.");

    await core.openDocumentInApp(document.id);
    await core.showDocumentInFolder(document.id);

    expect(opened).toEqual([path]);
    expect(shown).toEqual([path]);
    // A missing file can't be opened or shown.
    await rm(path);
    await core.reconcileDocuments();
    await expect(core.openDocumentInApp(document.id)).rejects.toThrow(NotFoundError);
    await expect(core.showDocumentInFolder(document.id)).rejects.toThrow(NotFoundError);
    expect(opened).toHaveLength(1);
  });
});
