import { randomUUID } from "node:crypto";
import { describe, expect, test, vi } from "vitest";
import {
  type Core,
  type Document,
  type Folder,
  InvalidInputError,
  NotFoundError,
} from "../../src/core";
import { createTempDataFolder, queryDatabase, startCore, tickingClock } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

const names = (folders: readonly { name: string }[]) => folders.map((folder) => folder.name);

/** Adds one small text Document per name, each with its own text, and waits until they are ready. */
async function addDocuments(core: Core, ...documentNames: string[]): Promise<Document[]> {
  const sources = await createTempDataFolder();
  const paths = await Promise.all(
    documentNames.map((name) =>
      writeSourceFile(sources, `${name}.txt`, `Notes about ${name}, kept for later.`),
    ),
  );
  return addAndProcess(core, paths);
}

/** Creates a chain of Folders, each inside the one before, and returns them top first. */
async function createChain(core: Core, ...folderNames: string[]): Promise<Folder[]> {
  const chain: Folder[] = [];
  for (const name of folderNames) {
    chain.push(await core.createFolder({ name, parentId: chain.at(-1)?.id ?? null }));
  }
  return chain;
}

describe("Folders", () => {
  test("a new data folder has no Folders", async () => {
    const core = startCore(await createTempDataFolder());

    expect(await core.listFolders()).toEqual([]);
  });

  test("creating a Folder returns it with a random UUID, its name trimmed, and timestamps", async () => {
    const core = startCore(await createTempDataFolder(), {
      now: () => new Date("2026-10-06T12:00:00Z"),
    });

    const folder = await core.createFolder({ name: "  Projects  " });

    expect(folder).toEqual({
      id: expect.stringMatching(UUID_V4),
      name: "Projects",
      parentId: null,
      createdAt: "2026-10-06T12:00:00.000Z",
      updatedAt: "2026-10-06T12:00:00.000Z",
    });
    expect(await core.listFolders()).toEqual([folder]);
  });

  test("Folders nest with no depth limit", async () => {
    const core = startCore(await createTempDataFolder());
    const depth = 30;

    const chain = await createChain(
      core,
      ...Array.from({ length: depth }, (_, level) => `Level ${level}`),
    );

    const listed = new Map((await core.listFolders()).map((folder) => [folder.id, folder]));
    expect(listed.size).toBe(depth);
    expect(chain[0]?.parentId).toBeNull();
    for (const [level, folder] of chain.entries()) {
      expect(listed.get(folder.id)).toEqual(folder);
      if (level > 0) expect(folder.parentId).toBe(chain[level - 1]?.id);
    }
  });

  test("Folders are listed by name, ignoring case, whatever their depth", async () => {
    const core = startCore(await createTempDataFolder());
    const papers = await core.createFolder({ name: "papers" });
    await core.createFolder({ name: "Contracts" });
    await core.createFolder({ name: "Archive", parentId: papers.id });
    await core.createFolder({ name: "Books" });

    expect(names(await core.listFolders())).toEqual(["Archive", "Books", "Contracts", "papers"]);
  });

  test("a Folder needs a name, and a parent that exists", async () => {
    const core = startCore(await createTempDataFolder());
    const deleted = await core.createFolder({ name: "Deleted" });
    await core.deleteFolder(deleted.id);

    await expect(core.createFolder({ name: "   " })).rejects.toThrow(InvalidInputError);
    await expect(core.createFolder({ name: 42 } as never)).rejects.toThrow(InvalidInputError);
    await expect(core.createFolder("Projects" as never)).rejects.toThrow(InvalidInputError);
    await expect(core.createFolder({ name: "Lost", parentId: randomUUID() })).rejects.toThrow(
      NotFoundError,
    );
    await expect(core.createFolder({ name: "Lost", parentId: deleted.id })).rejects.toThrow(
      NotFoundError,
    );
    expect(await core.listFolders()).toEqual([]);
  });
});

describe("Renaming Folders", () => {
  test("renaming a Folder changes its name, trimmed, and moves its updatedAt", async () => {
    const core = startCore(await createTempDataFolder(), { now: tickingClock() });
    const [projects, nested] = await createChain(core, "Projects", "Nested");
    if (!projects || !nested) throw new Error("The Folders weren't created.");

    const renamed = await core.renameFolder(nested.id, "  2026 reports  ");

    expect(renamed).toEqual({ ...nested, name: "2026 reports", updatedAt: renamed.updatedAt });
    expect(renamed.updatedAt > nested.updatedAt).toBe(true);
    expect(await core.listFolders()).toEqual([renamed, projects]);
  });

  test("an empty name, or a Folder that doesn't exist, is refused", async () => {
    const core = startCore(await createTempDataFolder());
    const folder = await core.createFolder({ name: "Kept" });

    await expect(core.renameFolder(folder.id, "  ")).rejects.toThrow(InvalidInputError);
    await expect(core.renameFolder(folder.id, 42 as never)).rejects.toThrow(InvalidInputError);
    await expect(core.renameFolder(randomUUID(), "Lost")).rejects.toThrow(NotFoundError);
    expect(await core.listFolders()).toEqual([folder]);
  });
});

describe("Moving Folders", () => {
  test("a Folder moves into another Folder, with everything in it, and back to the top level", async () => {
    const core = startCore(await createTempDataFolder(), { now: tickingClock() });
    const [projects, reports] = await createChain(core, "Projects", "Reports");
    const archive = await core.createFolder({ name: "Archive" });
    if (!projects || !reports) throw new Error("The Folders weren't created.");

    const moved = await core.moveFolder(projects.id, archive.id);

    expect(moved).toEqual({ ...projects, parentId: archive.id, updatedAt: moved.updatedAt });
    expect(moved.updatedAt > projects.updatedAt).toBe(true);
    const listed = await core.listFolders();
    expect(listed.find((folder) => folder.id === projects.id)?.parentId).toBe(archive.id);
    // Reports came along: it is still inside Projects.
    expect(listed.find((folder) => folder.id === reports.id)?.parentId).toBe(projects.id);

    const back = await core.moveFolder(projects.id, null);
    expect(back.parentId).toBeNull();
    expect((await core.listFolders()).find((folder) => folder.id === projects.id)).toEqual(back);
  });

  test("moving a Folder into itself or into one of its own sub-Folders is refused", async () => {
    const core = startCore(await createTempDataFolder());
    const [top, middle, bottom] = await createChain(core, "Top", "Middle", "Bottom");
    if (!top || !middle || !bottom) throw new Error("The Folders weren't created.");
    const before = await core.listFolders();

    await expect(core.moveFolder(top.id, top.id)).rejects.toThrow(InvalidInputError);
    await expect(core.moveFolder(top.id, middle.id)).rejects.toThrow(InvalidInputError);
    await expect(core.moveFolder(top.id, bottom.id)).rejects.toThrow(InvalidInputError);
    await expect(core.moveFolder(middle.id, bottom.id)).rejects.toThrow(InvalidInputError);

    expect(await core.listFolders()).toEqual(before);
    // Moving the other way, up the tree, is fine.
    expect((await core.moveFolder(bottom.id, top.id)).parentId).toBe(top.id);
  });

  test("moving to or from a Folder that doesn't exist is refused", async () => {
    const core = startCore(await createTempDataFolder());
    const folder = await core.createFolder({ name: "Kept" });
    const deleted = await core.createFolder({ name: "Deleted" });
    await core.deleteFolder(deleted.id);

    await expect(core.moveFolder(folder.id, randomUUID())).rejects.toThrow(NotFoundError);
    await expect(core.moveFolder(folder.id, deleted.id)).rejects.toThrow(NotFoundError);
    await expect(core.moveFolder(deleted.id, null)).rejects.toThrow(NotFoundError);
    await expect(core.moveFolder(folder.id, 42 as never)).rejects.toThrow(InvalidInputError);
    expect(await core.listFolders()).toEqual([folder]);
  });
});

describe("Filing Documents in Folders", { timeout: 30_000 }, () => {
  test("a new Document is unfiled", async () => {
    const core = startCore(await createTempDataFolder());

    const [document] = await addDocuments(core, "Paper");

    expect(document?.folderId).toBeNull();
    expect((await core.listDocuments())[0]?.folderId).toBeNull();
  });

  test("a Document is in at most one Folder: moving it files it in the new one only", async () => {
    const core = startCore(await createTempDataFolder(), { now: tickingClock() });
    const [paper] = await addDocuments(core, "Paper");
    const projects = await core.createFolder({ name: "Projects" });
    const archive = await core.createFolder({ name: "Archive" });
    if (!paper) throw new Error("Nothing was added.");

    const filed = await core.moveDocument(paper.id, projects.id);
    expect(filed).toEqual({ ...paper, folderId: projects.id, updatedAt: filed.updatedAt });
    expect(filed.updatedAt > paper.updatedAt).toBe(true);

    const refiled = await core.moveDocument(paper.id, archive.id);
    expect(refiled.folderId).toBe(archive.id);
    expect(await core.listDocuments()).toEqual([refiled]);
    expect(await core.listDocuments({ folderId: projects.id })).toEqual([]);
    expect(await core.listDocuments({ folderId: archive.id })).toEqual([refiled]);

    const unfiled = await core.moveDocument(paper.id, null);
    expect(unfiled.folderId).toBeNull();
    expect(await core.listDocuments()).toEqual([unfiled]);
    expect(await core.listDocuments({ folderId: archive.id })).toEqual([]);
  });

  test("moving a Document to a Folder that doesn't exist, or moving one that doesn't, is refused", async () => {
    const core = startCore(await createTempDataFolder());
    const [paper, gone] = await addDocuments(core, "Paper", "Gone");
    const deleted = await core.createFolder({ name: "Deleted" });
    await core.deleteFolder(deleted.id);
    if (!paper || !gone) throw new Error("Nothing was added.");
    await core.deleteDocument(gone.id);
    const folder = await core.createFolder({ name: "Kept" });

    await expect(core.moveDocument(paper.id, randomUUID())).rejects.toThrow(NotFoundError);
    await expect(core.moveDocument(paper.id, deleted.id)).rejects.toThrow(NotFoundError);
    await expect(core.moveDocument(gone.id, folder.id)).rejects.toThrow(NotFoundError);
    await expect(core.moveDocument(randomUUID(), folder.id)).rejects.toThrow(NotFoundError);
    await expect(core.moveDocument(paper.id, undefined as never)).rejects.toThrow(
      InvalidInputError,
    );
    expect(await core.listDocuments()).toEqual([paper]);
  });
});

describe("Filtering Documents by Folder", { timeout: 30_000 }, () => {
  test("a Folder's Documents, with or without those in its sub-Folders at any depth", async () => {
    const core = startCore(await createTempDataFolder());
    const [unfiled, inTop, inMiddle, inBottom, inOther] = await addDocuments(
      core,
      "Unfiled",
      "In top",
      "In middle",
      "In bottom",
      "In other",
    );
    const [top, middle, bottom] = await createChain(core, "Top", "Middle", "Bottom");
    const other = await core.createFolder({ name: "Other" });
    if (!unfiled || !inTop || !inMiddle || !inBottom || !inOther || !top || !middle || !bottom) {
      throw new Error("Setting up failed.");
    }
    await core.moveDocument(inTop.id, top.id);
    await core.moveDocument(inMiddle.id, middle.id);
    await core.moveDocument(inBottom.id, bottom.id);
    await core.moveDocument(inOther.id, other.id);
    const listedNames = async (options?: Parameters<Core["listDocuments"]>[0]) =>
      names(await core.listDocuments(options));

    // Most recently added first, as in the full list.
    expect(await listedNames({ folderId: top.id, includeSubfolders: true })).toEqual([
      "In bottom",
      "In middle",
      "In top",
    ]);
    expect(await listedNames({ folderId: middle.id, includeSubfolders: true })).toEqual([
      "In bottom",
      "In middle",
    ]);
    expect(await listedNames({ folderId: top.id })).toEqual(["In top"]);
    expect(await listedNames({ folderId: top.id, includeSubfolders: false })).toEqual(["In top"]);
    expect(await listedNames({ folderId: other.id, includeSubfolders: true })).toEqual([
      "In other",
    ]);
    expect(await listedNames()).toEqual([
      "In other",
      "In bottom",
      "In middle",
      "In top",
      "Unfiled",
    ]);
    expect(await listedNames({})).toEqual(await listedNames());
  });

  test("the filter follows Folder moves", async () => {
    const core = startCore(await createTempDataFolder());
    const [report] = await addDocuments(core, "Report");
    const [projects, reports] = await createChain(core, "Projects", "Reports");
    const archive = await core.createFolder({ name: "Archive" });
    if (!report || !projects || !reports) throw new Error("Setting up failed.");
    await core.moveDocument(report.id, reports.id);
    const inFolder = async (folderId: string) =>
      names(await core.listDocuments({ folderId, includeSubfolders: true }));

    expect(await inFolder(projects.id)).toEqual(["Report"]);

    await core.moveFolder(reports.id, archive.id);

    expect(await inFolder(projects.id)).toEqual([]);
    expect(await inFolder(archive.id)).toEqual(["Report"]);
  });

  test("filtering by a Folder that doesn't exist, or with malformed options, is refused", async () => {
    const core = startCore(await createTempDataFolder());
    const deleted = await core.createFolder({ name: "Deleted" });
    await core.deleteFolder(deleted.id);

    await expect(core.listDocuments({ folderId: randomUUID() })).rejects.toThrow(NotFoundError);
    await expect(core.listDocuments({ folderId: deleted.id })).rejects.toThrow(NotFoundError);
    await expect(core.listDocuments({ folderId: 42 } as never)).rejects.toThrow(InvalidInputError);
    await expect(
      core.listDocuments({ folderId: deleted.id, includeSubfolders: "yes" } as never),
    ).rejects.toThrow(InvalidInputError);
    await expect(core.listDocuments("all" as never)).rejects.toThrow(InvalidInputError);
  });
});

describe("Deleting Folders", { timeout: 30_000 }, () => {
  test("deleting a Folder deletes its sub-Folders and leaves their Documents unfiled", async () => {
    const core = startCore(await createTempDataFolder(), { now: tickingClock() });
    const [inTop, inBottom, inKept] = await addDocuments(core, "In top", "In bottom", "In kept");
    const [top, middle, bottom] = await createChain(core, "Top", "Middle", "Bottom");
    const kept = await core.createFolder({ name: "Kept" });
    const keptChild = await core.createFolder({ name: "Kept child", parentId: kept.id });
    if (!inTop || !inBottom || !inKept || !top || !middle || !bottom) {
      throw new Error("Setting up failed.");
    }
    await core.moveDocument(inTop.id, top.id);
    await core.moveDocument(inBottom.id, bottom.id);
    const keptDocument = await core.moveDocument(inKept.id, keptChild.id);

    await core.deleteFolder(top.id);

    expect(await core.listFolders()).toEqual([kept, keptChild]);
    const documents = await core.listDocuments();
    expect(names(documents)).toEqual(["In kept", "In bottom", "In top"]);
    expect(documents.find((each) => each.id === inTop.id)?.folderId).toBeNull();
    expect(documents.find((each) => each.id === inBottom.id)?.folderId).toBeNull();
    expect(documents.find((each) => each.id === inKept.id)).toEqual(keptDocument);
    // The Documents are still searchable: deleting a Folder never deletes Documents.
    expect(
      (await core.searchPassages("In bottom", { mode: "keyword" })).map(
        (result) => result.documentId,
      ),
    ).toEqual([inBottom.id]);
  });

  test("a deleted Folder, and its sub-Folders, can't be renamed, moved, used or deleted again", async () => {
    const core = startCore(await createTempDataFolder());
    const [top, child] = await createChain(core, "Top", "Child");
    const other = await core.createFolder({ name: "Other" });
    const [paper] = await addDocuments(core, "Paper");
    if (!top || !child || !paper) throw new Error("Setting up failed.");
    await core.deleteFolder(top.id);

    for (const deleted of [top, child]) {
      await expect(core.renameFolder(deleted.id, "Back")).rejects.toThrow(NotFoundError);
      await expect(core.moveFolder(deleted.id, other.id)).rejects.toThrow(NotFoundError);
      await expect(core.moveFolder(other.id, deleted.id)).rejects.toThrow(NotFoundError);
      await expect(core.moveDocument(paper.id, deleted.id)).rejects.toThrow(NotFoundError);
      await expect(core.deleteFolder(deleted.id)).rejects.toThrow(NotFoundError);
    }
    expect(await core.listFolders()).toEqual([other]);
  });
});

describe("Folder events", { timeout: 30_000 }, () => {
  test("the core pushes the list of Folders whenever it changes", async () => {
    const core = startCore(await createTempDataFolder(), { now: tickingClock() });
    const listener = vi.fn();
    core.on("folders.changed", listener);

    const projects = await core.createFolder({ name: "Projects" });
    expect(listener).toHaveBeenLastCalledWith([projects]);

    const reports = await core.createFolder({ name: "Reports", parentId: projects.id });
    expect(listener).toHaveBeenLastCalledWith([projects, reports]);

    const renamed = await core.renameFolder(projects.id, "Work");
    expect(listener).toHaveBeenLastCalledWith([reports, renamed]);

    const moved = await core.moveFolder(reports.id, null);
    expect(listener).toHaveBeenLastCalledWith([moved, renamed]);

    await core.deleteFolder(renamed.id);
    expect(listener).toHaveBeenLastCalledWith([moved]);
    expect(listener).toHaveBeenCalledTimes(5);
  });

  test("the core pushes the Documents that move, including those a Folder deletion unfiles", async () => {
    const core = startCore(await createTempDataFolder(), { now: tickingClock() });
    const [paper, report, notes] = await addDocuments(core, "Paper", "Report", "Notes");
    const [top, child] = await createChain(core, "Top", "Child");
    if (!paper || !report || !notes || !top || !child) throw new Error("Setting up failed.");
    const listener = vi.fn();
    core.on("documents.moved", listener);

    const filedPaper = await core.moveDocument(paper.id, top.id);
    expect(listener).toHaveBeenLastCalledWith([filedPaper]);
    await core.moveDocument(report.id, child.id);

    // Moving a Document to where it already is changes nothing, so pushes nothing.
    await core.moveDocument(notes.id, null);
    expect(listener).toHaveBeenCalledTimes(2);

    await core.deleteFolder(top.id);
    expect(listener).toHaveBeenCalledTimes(3);
    const unfiled: Document[] = listener.mock.lastCall?.[0] ?? [];
    const listed = await core.listDocuments();
    expect(unfiled.map((each) => each.id).sort()).toEqual([paper.id, report.id].sort());
    for (const document of unfiled) {
      expect(document.folderId).toBeNull();
      expect(listed.find((each) => each.id === document.id)).toEqual(document);
    }
  });
});

describe("Folders after a restart", { timeout: 30_000 }, () => {
  test("Folders, their nesting, and where each Document is filed survive a restart", async () => {
    const dataDir = await createTempDataFolder();
    const before = startCore(dataDir, { now: tickingClock() });
    const [paper, report] = await addDocuments(before, "Paper", "Report");
    const [projects, reports] = await createChain(before, "Projects", "Reports");
    const gone = await before.createFolder({ name: "Gone", parentId: projects?.id });
    if (!paper || !report || !projects || !reports) throw new Error("Setting up failed.");
    await before.renameFolder(projects.id, "Work");
    await before.moveDocument(paper.id, reports.id);
    await before.deleteFolder(gone.id);
    const folders = await before.listFolders();
    const documents = await before.listDocuments();
    const filtered = await before.listDocuments({ folderId: projects.id, includeSubfolders: true });
    before.close();

    const after = startCore(dataDir);

    expect(await after.listFolders()).toEqual(folders);
    expect(names(folders)).toEqual(["Reports", "Work"]);
    expect(await after.listDocuments()).toEqual(documents);
    expect(await after.listDocuments({ folderId: projects.id, includeSubfolders: true })).toEqual(
      filtered,
    );
    expect(names(filtered)).toEqual(["Paper"]);
  });
});

describe("Folders follow the sync-ready rules (ADR-0003)", { timeout: 30_000 }, () => {
  test("ids are random UUIDs, every row has timestamps, and deletes are soft", async () => {
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir, { now: tickingClock() });
    const [paper] = await addDocuments(core, "Paper");
    const [top, child] = await createChain(core, "Top", "Child");
    if (!paper || !top || !child) throw new Error("Setting up failed.");
    const filed = await core.moveDocument(paper.id, child.id);

    await core.deleteFolder(top.id);

    const rows = queryDatabase<{
      id: string;
      parent_id: string | null;
      name: string;
      created_at: string;
      updated_at: string;
      deleted_at: string | null;
    }>(dataDir, "SELECT id, parent_id, name, created_at, updated_at, deleted_at FROM folders");
    // Both rows stay, marked deleted at the same moment.
    expect(rows).toHaveLength(2);
    const deletedAt = rows[0]?.deleted_at;
    expect(deletedAt).toEqual(expect.any(String));
    for (const row of rows) {
      expect(row.id).toMatch(UUID_V4);
      expect(row.created_at).toEqual(expect.any(String));
      expect(row.deleted_at).toBe(deletedAt);
      expect(row.updated_at).toBe(deletedAt);
    }
    expect(rows.find((row) => row.id === child.id)?.parent_id).toBe(top.id);

    // The Document's row records the change too: unfiled, with a later updated_at.
    const [document] = queryDatabase<{ folder_id: string | null; updated_at: string }>(
      dataDir,
      "SELECT folder_id, updated_at FROM documents WHERE id = ?",
      [paper.id],
    );
    expect(document).toEqual({ folder_id: null, updated_at: deletedAt });
    expect((deletedAt ?? "") > filed.updatedAt).toBe(true);
  });
});
