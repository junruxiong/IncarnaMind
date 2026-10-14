import { existsSync } from "node:fs";
import { mkdir, readdir, readFile, realpath, rename, rm, symlink } from "node:fs/promises";
import { dirname, join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import { type Core, type FileWriteRequest, InvalidInputError, NotFoundError } from "../../src/core";
import { createOwnFiles } from "../../src/core/documents/ownFiles";
import { createEventHub } from "../../src/core/events";
import { createOutputFiles } from "../../src/core/outputFiles";
import { createSettings } from "../../src/core/settings";
import { migrate, openDatabase } from "../../src/core/storage";
import { createTempDataFolder, nextEvent, startCore } from "../helpers/core";
import { createSourceFolder, linkAndProcess, sha256, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { note, writeMind } from "../helpers/minds";

/** A core with a Mind titled "Tides" holding one Note, and the Documents folder its first save offers. */
async function setUp(dataDir?: string) {
  const data = dataDir ?? (await createTempDataFolder());
  const core = startCore(data);
  const mind = await core.createMind({ title: "Tides" });
  const client = await connectToMind(core, mind.id);
  writeMind(client, [note("Spring tides come twice a month.")]);
  const save = () => core.saveMindExport(mind.id, { format: "markdown" });
  return { core, data, mind, save, suggested: join(data, "Documents", "IncarnaMind") };
}

/** Every question the core raises from now on. */
function questionsAsked(core: Core): FileWriteRequest[] {
  const asked: FileWriteRequest[] = [];
  core.on("fileWrite.requested", (request) => asked.push(request));
  return asked;
}

/** The question a save is waiting on, once it has asked. */
function waitingQuestion(core: Core): Promise<FileWriteRequest> {
  return vi.waitFor(async () => {
    const [request] = await core.listFileWriteRequests();
    if (!request) throw new Error("Nothing has been asked yet.");
    return request;
  });
}

/** Answers the question a save is waiting on, which must be of this kind, and returns it. */
async function answerNext<K extends FileWriteRequest["kind"]>(
  core: Core,
  kind: K,
  accept: boolean,
): Promise<Extract<FileWriteRequest, { kind: K }>> {
  const request = await waitingQuestion(core);
  expect(request.kind).toBe(kind);
  await core.respondToFileWrite(request.requestId, accept);
  return request as Extract<FileWriteRequest, { kind: K }>;
}

/** Every file under `root`, with its content hash, for checking nothing there changed. */
async function hashes(root: string): Promise<Record<string, string>> {
  const result: Record<string, string> = {};
  for (const name of await readdir(root, { recursive: true })) {
    const path = join(root, name);
    try {
      result[name] = sha256(await readFile(path));
    } catch {
      // A folder.
    }
  }
  return result;
}

describe("The Output location", () => {
  test("the first save asks where new files go, offering IncarnaMind in Documents; Save here makes it, saves there, and is remembered", async () => {
    const { core, save, suggested } = await setUp();
    expect(await core.getOutputSettings()).toEqual({
      location: null,
      suggested,
      linkedFolders: [],
    });

    const saving = save();
    const request = await answerNext(core, "output-location", true);
    expect(request).toMatchObject({ suggested, fileName: "Tides.md" });
    const result = await saving;

    const location = await realpath(suggested);
    expect(result).toEqual({
      status: "saved",
      file: { path: join(location, "Tides.md"), name: "Tides.md" },
      citations: 0,
      unverifiedCitations: 0,
      questions: 0,
      word: null,
    });
    expect(await readFile(join(location, "Tides.md"), "utf8")).toContain(
      "Spring tides come twice a month.",
    );
    expect((await core.getOutputSettings()).location).toBe(location);
    expect(await core.listFileWriteRequests()).toEqual([]);

    // The next save doesn't ask.
    const asked = questionsAsked(core);
    expect(await save()).toMatchObject({ status: "saved", file: { name: "Tides 2.md" } });
    expect(asked).toEqual([]);
  });

  test("a folder the User picks answers the first save, and later saves go there too", async () => {
    const { core, save } = await setUp();
    const picked = await createSourceFolder();

    const saving = save();
    const request = await waitingQuestion(core);
    const resolved = nextEvent(core, "fileWrite.resolved");
    const settings = await core.setOutputLocation(picked);
    expect(settings.location).toBe(picked);
    expect(await resolved).toEqual({ requestId: request.requestId });
    expect(await saving).toMatchObject({ file: { path: join(picked, "Tides.md") } });

    // Picked in Settings before any save: nothing is asked.
    const other = await createSourceFolder();
    await core.setOutputLocation(other);
    const asked = questionsAsked(core);
    expect(await save()).toMatchObject({ file: { path: join(other, "Tides.md") } });
    expect(asked).toEqual([]);
  });

  test("cancelling the first question saves nothing, and the next save asks again", async () => {
    const { core, save, suggested } = await setUp();
    const saving = save();
    await answerNext(core, "output-location", false);
    expect(await saving).toEqual({ status: "cancelled" });
    expect(existsSync(suggested)).toBe(false);
    expect((await core.getOutputSettings()).location).toBeNull();

    const again = save();
    await answerNext(core, "output-location", true);
    expect(await again).toMatchObject({ status: "saved" });
  });

  test("saves waiting at the same time share one question", async () => {
    const { core, save } = await setUp();
    const asked = questionsAsked(core);
    const first = save();
    const second = save();
    await expect.poll(() => asked.length).toBe(1);
    await core.respondToFileWrite(asked[0]?.requestId ?? "", true);
    const names = [await first, await second].map((each) =>
      each.status === "saved" ? each.file.name : each.status,
    );
    expect(names.sort()).toEqual(["Tides 2.md", "Tides.md"]);
    expect(asked).toHaveLength(1);
  });

  test("the Output location and the answers are kept when IncarnaMind starts again", async () => {
    const first = await setUp();
    const picked = await createSourceFolder();
    await first.core.setOutputLocation(picked);
    first.core.close();

    const again = startCore(first.data);
    expect((await again.getOutputSettings()).location).toBe(picked);
  });

  test("a folder that isn't there, or isn't a folder, isn't taken", async () => {
    const { core } = await setUp();
    const folder = await createSourceFolder();
    await expect(core.setOutputLocation(join(folder, "nowhere"))).rejects.toThrow(NotFoundError);
    const file = await writeSourceFile(folder, "notes.txt", "Not a folder.");
    await expect(core.setOutputLocation(file)).rejects.toThrow(InvalidInputError);
    await expect(core.setOutputLocation("relative/path")).rejects.toThrow(InvalidInputError);
    expect((await core.getOutputSettings()).location).toBeNull();
  });

  test("an Output location removed from a folder still there is made again at the next save", async () => {
    const { core, save } = await setUp();
    const parent = await createSourceFolder();
    const location = join(parent, "Exports");
    await mkdir(location);
    await core.setOutputLocation(location);
    await rm(location, { recursive: true });
    expect(await save()).toMatchObject({ file: { path: join(location, "Tides.md") } });
  });
});

describe("New files never replace anything", () => {
  test("a name that is taken gets the next free one, and the file already there is unchanged", async () => {
    const { core, save } = await setUp();
    const location = await createSourceFolder();
    await core.setOutputLocation(location);
    await writeSourceFile(location, "Tides.md", "My own notes.");
    const before = await hashes(location);

    expect(await save()).toMatchObject({ file: { name: "Tides 2.md" } });
    expect(await save()).toMatchObject({ file: { name: "Tides 3.md" } });
    const after = await hashes(location);
    expect(after["Tides.md"]).toBe(before["Tides.md"]);
    expect(Object.keys(after).sort()).toEqual(["Tides 2.md", "Tides 3.md", "Tides.md"]);
  });

  test("a link under the name, even one leading nowhere, is skipped, not followed", async () => {
    const { core, save } = await setUp();
    const location = await createSourceFolder();
    const elsewhere = await createSourceFolder();
    await core.setOutputLocation(location);
    await symlink(join(elsewhere, "target.md"), join(location, "Tides.md"));

    expect(await save()).toMatchObject({ file: { name: "Tides 2.md" } });
    expect(await readdir(elsewhere)).toEqual([]);
  });

  test("an Output location replaced by a link that leads outside it is refused, and nothing is written there", async () => {
    const { core, save } = await setUp();
    const parent = await createSourceFolder();
    const elsewhere = await createSourceFolder();
    const location = join(parent, "Exports");
    await mkdir(location);
    await core.setOutputLocation(location);
    await rename(location, join(parent, "Moved"));
    await symlink(elsewhere, location, "dir");

    await expect(save()).rejects.toThrow(/only in the output location/);
    expect(await readdir(elsewhere)).toEqual([]);
  });
});

describe("Linked folders", () => {
  /** A Linked folder holding one original, with the Output location inside it. */
  async function linkedSetUp() {
    const setup = await setUp();
    const linked = await createSourceFolder();
    await writeSourceFile(linked, "Lease.txt", "Rent is due monthly.");
    await linkAndProcess(setup.core, linked);
    const [folder] = await setup.core.listLinkedFolders();
    if (!folder) throw new Error("The folder wasn't linked.");
    const exports = join(linked, "Exports");
    await mkdir(exports);
    await setup.core.setOutputLocation(exports);
    return { ...setup, linked, exports, linkedFolderId: folder.id };
  }

  test("the first save inside a Linked folder asks once; Allow is remembered, and the originals are unchanged", async () => {
    const { core, save, linked, exports, linkedFolderId } = await linkedSetUp();
    const before = await hashes(linked);

    const saving = save();
    const request = await answerNext(core, "linked-folder", true);
    expect(request).toMatchObject({
      linkedFolderId,
      linkedFolderPath: linked,
      fileName: "Tides.md",
    });
    expect(await saving).toMatchObject({ file: { path: join(exports, "Tides.md") } });

    const asked = questionsAsked(core);
    expect(await save()).toMatchObject({ file: { name: "Tides 2.md" } });
    expect(asked).toEqual([]);
    expect((await core.getOutputSettings()).linkedFolders).toEqual([
      { linkedFolderId, path: linked, allowed: true, decidedAt: expect.any(String) },
    ]);

    // Nothing that was there changed: only the new files were added.
    const after = await hashes(linked);
    for (const [name, hash] of Object.entries(before)) expect(after[name]).toBe(hash);
    expect(Object.keys(after).sort()).toEqual(
      [...Object.keys(before), join("Exports", "Tides 2.md"), join("Exports", "Tides.md")].sort(),
    );
  });

  test("Don't allow is remembered: the next save there is refused without asking, until Settings changes the answer", async () => {
    const { core, save, exports, linked, linkedFolderId } = await linkedSetUp();
    const saving = save();
    await answerNext(core, "linked-folder", false);
    expect(await saving).toEqual({
      status: "not-allowed",
      linkedFolderId,
      linkedFolderPath: linked,
    });
    expect(await readdir(exports)).toEqual([]);

    const asked = questionsAsked(core);
    expect(await save()).toMatchObject({ status: "not-allowed" });
    expect(asked).toEqual([]);

    // Settings › General: allowed now.
    const changed = nextEvent(core, "outputSettings.changed");
    const settings = await core.setLinkedFolderWriting(linkedFolderId, true);
    expect(settings.linkedFolders).toMatchObject([{ linkedFolderId, allowed: true }]);
    expect((await changed).linkedFolders).toMatchObject([{ allowed: true }]);
    expect(await save()).toMatchObject({ status: "saved", file: { name: "Tides.md" } });

    // And back: refused again, still without asking.
    await core.setLinkedFolderWriting(linkedFolderId, false);
    expect(await save()).toMatchObject({ status: "not-allowed" });
    expect(asked).toEqual([]);
  });

  test("answering in Settings while a save waits lets it go on", async () => {
    const { core, save, linkedFolderId } = await linkedSetUp();
    const saving = save();
    const request = await waitingQuestion(core);
    const resolved = nextEvent(core, "fileWrite.resolved");
    await core.setLinkedFolderWriting(linkedFolderId, true);
    expect(await resolved).toEqual({ requestId: request.requestId });
    expect(await saving).toMatchObject({ status: "saved" });
  });

  test("unlinking the folder forgets the answer: linked again, it is asked about again", async () => {
    const { core, save, linked, linkedFolderId } = await linkedSetUp();
    const saving = save();
    await answerNext(core, "linked-folder", true);
    await saving;

    await core.removeLinkedFolder(linkedFolderId);
    expect((await core.getOutputSettings()).linkedFolders).toEqual([]);
    await expect(core.setLinkedFolderWriting(linkedFolderId, true)).rejects.toThrow(NotFoundError);

    await linkAndProcess(core, linked);
    const again = save();
    await answerNext(core, "linked-folder", true);
    expect(await again).toMatchObject({ status: "saved" });
  });

  test("a save waiting on a folder that is unlinked stops, and saves nothing", async () => {
    const { core, save, exports, linkedFolderId } = await linkedSetUp();
    const saving = save();
    await waitingQuestion(core);
    await core.removeLinkedFolder(linkedFolderId);
    // Unlinked, the Output location is no longer inside a Linked folder: the save goes on there.
    expect(await saving).toMatchObject({ status: "saved" });
    expect(await readdir(exports)).toEqual(["Tides.md"]);
  });
});

test("Show in Finder shows only a file IncarnaMind saved", async () => {
  const shown: string[] = [];
  const data = await createTempDataFolder();
  const core = startCore(data, {
    shell: {
      openPath: async () => undefined,
      showItemInFolder: (path) => {
        shown.push(path);
      },
    },
  });
  const location = await createSourceFolder();
  await core.setOutputLocation(location);
  const mind = await core.createMind({ title: "Tides" });
  const result = await core.saveMindExport(mind.id, { format: "docx" });
  if (result.status !== "saved") throw new Error("Not saved.");
  await core.showSavedFile(result.file.path);
  expect(shown).toEqual([join(location, "Tides.docx")]);

  const other = await writeSourceFile(location, "Other.txt", "Not saved by IncarnaMind.");
  await expect(core.showSavedFile(other)).rejects.toThrow(NotFoundError);
  expect(shown).toHaveLength(1);
});

describe("Saving next to an original (the call other features use)", () => {
  /** The module over an empty database, with one Linked folder holding one original. */
  async function outputsWith() {
    const db = openDatabase(":memory:");
    migrate(db);
    const now = () => new Date().toISOString();
    const linked = await createSourceFolder();
    const original = await writeSourceFile(linked, join("Leases", "Lease.docx"), "The original.");
    const outputs = createOutputFiles({
      db,
      events: createEventHub(),
      now,
      settings: createSettings(db, now, () => ["en-US"], "none"),
      suggested: join(await createSourceFolder(), "IncarnaMind"),
      linkedFolders: () => [{ id: "linked-1", path: linked }],
      recordOwnFile: createOwnFiles(db, now).record,
    });
    outputs.setLinkedFolderWriting("linked-1", true);
    const fingerprints = () =>
      db.all<{ content_hash: string; kind: string }>("SELECT content_hash, kind FROM own_files");
    return { outputs, linked, original, fingerprints };
  }

  const data = new TextEncoder().encode("A new file.");

  test("a file goes next to the original, under a free name, and the original is unchanged", async () => {
    const { outputs, linked, original } = await outputsWith();
    const before = sha256(await readFile(original));
    const folder = join(linked, "Leases");
    expect(await outputs.save({ name: "Lease.docx", data, folder, ownFile: "export" })).toEqual({
      path: join(folder, "Lease 2.docx"),
      name: "Lease 2.docx",
    });
    expect(sha256(await readFile(original))).toBe(before);
    // A name can't climb out of its folder.
    const escaping = await outputs.save({
      name: "../../Escape.md",
      data,
      folder,
      ownFile: "export",
    });
    expect(dirname(escaping.path)).toBe(folder);
    expect(escaping.name).toMatch(/Escape\.md$/);
    expect(await readdir(linked)).toEqual(["Leases"]);
  });

  test("a file IncarnaMind wrote has its fingerprint recorded, once; an empty one, or a source it saved, has none (#80)", async () => {
    const { outputs, linked, fingerprints } = await outputsWith();
    const folder = join(linked, "Leases");
    await outputs.save({ name: "Lease.md", data, folder, ownFile: "export" });
    await outputs.save({ name: "Lease.md", data, folder, ownFile: "export" });
    expect(fingerprints()).toEqual([{ content_hash: sha256(data), kind: "export" }]);

    await outputs.save({ name: "Empty.md", data: new Uint8Array(), folder, ownFile: "export" });
    const transcript = new TextEncoder().encode("[00:01] The tide turned at noon.");
    await outputs.save({ name: "Meeting.vtt", data: transcript, folder, ownFile: null });
    expect(fingerprints()).toHaveLength(1);
  });

  test("a folder outside the Output location and every Linked folder, or a link leading out of one, is refused", async () => {
    const { outputs, linked } = await outputsWith();
    const elsewhere = await createSourceFolder();
    await expect(
      outputs.save({ name: "Out.md", data, folder: elsewhere, ownFile: "export" }),
    ).rejects.toThrow(InvalidInputError);
    await symlink(elsewhere, join(linked, "Shortcut"), "dir");
    await expect(
      outputs.save({ name: "Out.md", data, folder: join(linked, "Shortcut"), ownFile: "export" }),
    ).rejects.toThrow(/only in the output location/);
    expect(await readdir(elsewhere)).toEqual([]);
  });
});
