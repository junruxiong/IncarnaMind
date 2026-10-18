/**
 * Tagging beside Organize: with Organize allowed and Folders in place, automatic
 * tagging tags new Documents unless Organize files them by itself, and only the
 * moves into Folders wait for Organize. Through the core's public interface,
 * with a scripted model: no network.
 */
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import { type Core, DATABASE_FILE, type Document } from "../../src/core";
import { openDatabase } from "../../src/core/storage";
import { createTempDataFolder, startCore } from "../helpers/core";
import { createSourceFolder, writeSourceFile } from "../helpers/documents";
import { scriptedModels } from "../helpers/models";
import { docxOf, xlsxOf } from "../helpers/office";
import { buildPdf } from "../helpers/pdf";
import { type ScriptedChat, scriptedChat } from "../helpers/scriptedChat";
import { tagNamed, tagNamesOf, waitForTagging } from "../helpers/tags";

const LOCAL = { kind: "ollama", modelId: "local-model" } as const;
const startOn = (dataDir: string, chat: ScriptedChat) =>
  startCore(dataDir, { createChatModel: scriptedModels(chat.model).createChatModel });

/** A core with a Folder and Organize allowed, filing new Documents by itself or not. */
async function start(automatic: boolean, dataDir?: string) {
  const directory = dataDir ?? (await createTempDataFolder());
  const chat = scriptedChat();
  const core = startOn(directory, chat);
  const provider = await core.saveChatProvider(LOCAL);
  const reports = await core.createLibraryGroup({ name: "Reports", description: "Reports" });
  const save = (filesByItself: boolean) =>
    core.saveLibrarySettings({
      classifier: { kind: "chat", choice: { providerId: provider.id, modelId: LOCAL.modelId } },
      automatic: filesByItself,
    });
  await save(automatic);
  return { core, chat, reports, dataDir: directory, save };
}

async function add(core: Core, files: Record<string, string | Uint8Array>): Promise<Document[]> {
  const folder = await createSourceFolder();
  const paths: string[] = [];
  for (const [name, contents] of Object.entries(files)) {
    paths.push(await writeSourceFile(folder, name, contents));
  }
  const { documents } = await core.addDocuments(paths);
  return waitForTagging(
    core,
    documents.map((each) => each.id),
    (each) => each.status === "ready",
  );
}

const assignmentsOf = async (core: Core) => (await core.getLibrary()).assignments;

/** Waits until Organize has filed each Document in `folderId`. */
const filed = (core: Core, ids: readonly string[], folderId: string) =>
  vi.waitFor(async () => {
    const assignments = await assignmentsOf(core);
    for (const id of ids) {
      expect(assignments.find((each) => each.documentId === id)).toMatchObject({
        groupId: folderId,
        status: "classified",
      });
    }
  });

describe("with Organize not moving new Documents by itself", () => {
  test("a PDF, a Word file and a sheet get Tags and none moves into a Folder", async () => {
    const { core, chat } = await start(false);

    const added = await add(core, {
      "Annual report.pdf": buildPdf([
        { lines: ["Annual report", "Revenue grew in every region."] },
      ]),
      "Board minutes.docx": docxOf([{ text: "Minutes of the board meeting in March." }]),
      "Sales.xlsx": xlsxOf([
        {
          name: "Sales",
          rows: [
            ["Region", "Total"],
            ["North", 12],
          ],
        },
      ]),
    });
    const ids = added.map((each) => each.id);
    await waitForTagging(core, ids);

    const report = await tagNamed(core, "Report");
    for (const document of await core.listDocuments()) {
      expect(document.tags.map((link) => link.tagId)).toEqual([report.id]);
    }
    expect(chat.calls.map((call) => call.purpose)).toEqual(["tags", "tags", "tags"]);
    expect(await assignmentsOf(core)).toEqual([]);
  });

  test("Organize, run afterwards, moves them and keeps a Tag the User added by hand", async () => {
    const { core, chat, reports } = await start(false);
    const [first, second] = await add(core, {
      "One.txt": "The first report, about revenue.",
      "Two.txt": "The second report, about costs.",
    });
    if (!first || !second) throw new Error("Nothing was added.");
    await waitForTagging(core, [first.id, second.id]);
    const book = await tagNamed(core, "Book");
    const report = await tagNamed(core, "Report");
    await core.addDocumentTag(first.id, book.id);
    chat.organizeTags([report.id]);

    await core.classifyDocuments();
    await filed(core, [first.id, second.id], reports.id);

    expect((await tagNamesOf(core, first.id)).sort()).toEqual(["Book", "Report"]);
    expect(await tagNamesOf(core, second.id)).toEqual(["Report"]);
  });

  test("Undo of that Organize run leaves the Tags", async () => {
    const { core, chat, reports } = await start(false);
    const [document] = await add(core, { "One.txt": "The first report, about revenue." });
    if (!document) throw new Error("Nothing was added.");
    await waitForTagging(core, [document.id]);
    chat.organizeTags([(await tagNamed(core, "Contract")).id]);
    await core.classifyDocuments([document.id]);
    await filed(core, [document.id], reports.id);
    expect(await tagNamesOf(core, document.id)).toEqual(["Contract"]);

    await core.restoreAssignment(document.id, null);

    expect(await assignmentsOf(core)).toEqual([]);
    expect(await tagNamesOf(core, document.id)).toEqual(["Contract"]);
  });

  test("Re-tag works the Tags out again and doesn't move the Document", async () => {
    const { core, chat } = await start(false);
    const [document] = await add(core, { "One.txt": "The first report, about revenue." });
    if (!document) throw new Error("Nothing was added.");
    await waitForTagging(core, [document.id]);

    await core.retagDocuments([document.id]);
    await vi.waitFor(() => expect(chat.calls).toHaveLength(2));
    await waitForTagging(core, [document.id]);

    expect(chat.calls.map((call) => call.purpose)).toEqual(["tags", "tags"]);
    expect(await assignmentsOf(core)).toEqual([]);
  });

  test("a Document Organize was asked to organize isn't tagged a second time by automatic tagging", async () => {
    const { core, chat, reports } = await start(false);
    chat.hold();
    const [first, second] = await add(core, {
      "One.txt": "The first report, about revenue.",
      "Two.txt": "The second report, about costs.",
    });
    if (!first || !second) throw new Error("Nothing was added.");
    await vi.waitFor(() => expect(chat.purposesFor("first report")).toEqual(["tags"]));

    await core.classifyDocuments([second.id]);
    chat.release();
    await filed(core, [second.id], reports.id);
    await waitForTagging(core, [first.id, second.id]);

    expect(chat.purposesFor("second report")).toEqual(["organize"]);
  });

  test("Documents the old behaviour left pending are tagged at the next start", async () => {
    const first = await start(false);
    const [document] = await add(first.core, { "One.txt": "The first report, about revenue." });
    if (!document) throw new Error("Nothing was added.");
    await waitForTagging(first.core, [document.id]);
    first.core.close();
    const db = openDatabase(join(first.dataDir, DATABASE_FILE));
    try {
      db.run("UPDATE documents SET tagging_status = 'pending'");
    } finally {
      db.close();
    }

    const chat = scriptedChat();
    const second = startOn(first.dataDir, chat);

    await waitForTagging(second, [document.id]);
    expect(chat.purposesFor("first report")).toEqual(["tags"]);
    expect(await assignmentsOf(second)).toEqual([]);
  });

  test("quitting halfway through tagging picks up at the next start", async () => {
    const first = await start(false);
    first.chat.hold();
    const [document] = await add(first.core, { "One.txt": "The first report, about revenue." });
    if (!document) throw new Error("Nothing was added.");
    await vi.waitFor(() => expect(first.chat.calls).toHaveLength(1));
    first.core.close();
    first.chat.release();

    const chat = scriptedChat();
    const second = startOn(first.dataDir, chat);

    await waitForTagging(second, [document.id]);
    expect(await tagNamesOf(second, document.id)).toEqual(["Report"]);
  });
});

describe("with Organize moving new Documents by itself", () => {
  test("Organize tags and files them, and automatic tagging sends nothing", async () => {
    const { core, chat, reports } = await start(true);

    const [document] = await add(core, { "One.txt": "The first report, about revenue." });
    if (!document) throw new Error("Nothing was added.");
    await filed(core, [document.id], reports.id);
    await waitForTagging(core, [document.id]);

    expect(chat.calls.map((call) => call.purpose)).toEqual(["organize"]);
  });
});

describe("turning on Organize filing new Documents by itself", () => {
  test("Documents automatic tagging holds are still tagged, and stay where they are", async () => {
    const { core, chat, save } = await start(false);
    chat.hold();
    const added = await add(core, {
      "One.txt": "The first report, about revenue.",
      "Two.txt": "The second report, about costs.",
      "Three.txt": "The third report, about staff.",
    });
    await vi.waitFor(() => expect(chat.calls).toHaveLength(1));

    await save(true);
    chat.release();

    await waitForTagging(
      core,
      added.map((each) => each.id),
    );
    expect(chat.calls.map((call) => call.purpose)).toEqual(["tags", "tags", "tags"]);
    expect(await assignmentsOf(core)).toEqual([]);
  });

  test("so are those still waiting after a restart", async () => {
    const first = await start(false);
    const [document] = await add(first.core, { "One.txt": "The first report, about revenue." });
    if (!document) throw new Error("Nothing was added.");
    await waitForTagging(first.core, [document.id]);
    await first.save(true);
    first.core.close();
    const db = openDatabase(join(first.dataDir, DATABASE_FILE));
    try {
      db.run("UPDATE documents SET tagging_status = 'pending'");
    } finally {
      db.close();
    }

    const chat = scriptedChat();
    const second = startOn(first.dataDir, chat);

    await waitForTagging(second, [document.id]);
    expect(chat.purposesFor("first report")).toEqual(["tags"]);
  });
});
