import { access, mkdir, readdir, readFile, rm, utimes, writeFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import { InvalidInputError, NotFoundError } from "../../src/core";
import { OPEN_COPIES_FOLDER } from "../../src/core/documents/copies";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, storedFile, writeSourceFile } from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";

const REPORT = buildPdf([{ lines: ["Quarterly report"] }, { lines: ["Results"] }]);

const exists = (path: string) =>
  access(path).then(
    () => true,
    () => false,
  );

/** A core whose temporary folder is a fresh one, with the Documents added from files of these names. */
async function setUp(files: Record<string, string | Uint8Array>) {
  const dataDir = await createTempDataFolder();
  const tempDir = await createTempDataFolder();
  const sources = await createTempDataFolder();
  const core = startCore(dataDir, { paths: { dataDir, tempDir } });
  const paths = await Promise.all(
    Object.entries(files).map(([name, contents]) => writeSourceFile(sources, name, contents)),
  );
  const documents = await addAndProcess(core, paths);
  return { core, dataDir, tempDir, sources, documents };
}

describe("getting a Document's original file back", { timeout: 30_000 }, () => {
  test("a copy is named after the Document, with its kind's extension, never its hash", async () => {
    const {
      core,
      documents: [report, notes, field],
    } = await setUp({
      "Report.pdf": REPORT,
      "notes.txt": "Some notes.\n",
      "Field notes.markdown": "# Field notes\n",
    });
    if (!report || !notes || !field) throw new Error("Not everything was added.");
    const nameOf = async (id: string) => (await core.documentCopyName(id)).fileName;

    expect(await nameOf(report.id)).toBe("Report.pdf");
    expect(await nameOf(notes.id)).toBe("notes.txt");
    expect(await nameOf(field.id)).toBe("Field notes.md");
    expect((await core.documentCopyName(report.id)).document).toEqual(report);

    // Renamed: the new name, without what file systems refuse, and one extension.
    await core.renameDocument(report.id, 'Q3/Q4: the "plan"?');
    expect(await nameOf(report.id)).toBe("Q3 Q4 the plan.pdf");
    await core.renameDocument(notes.id, "draft.TXT");
    expect(await nameOf(notes.id)).toBe("draft.TXT");
    await core.renameDocument(notes.id, "con");
    expect(await nameOf(notes.id)).toBe("con_.txt");
    await core.renameDocument(field.id, "???");
    expect(await nameOf(field.id)).toBe("Document.md");
  });

  test("opening one elsewhere makes a temporary copy named after it, in a folder of its own", async () => {
    const {
      core,
      tempDir,
      documents: [report],
    } = await setUp({ "Report.pdf": REPORT });
    if (!report) throw new Error("Nothing was added.");

    const first = await core.temporaryDocumentCopy(report.id);
    const second = await core.temporaryDocumentCopy(report.id);

    for (const path of [first, second]) {
      expect(basename(path)).toBe("Report.pdf");
      expect(dirname(dirname(path))).toBe(join(tempDir, OPEN_COPIES_FOLDER));
      expect(path).not.toContain(report.contentHash);
      expect(new Uint8Array(await readFile(path))).toEqual(REPORT);
    }
    // Opened twice, it is copied twice: a copy that is open is never replaced.
    expect(dirname(first)).not.toBe(dirname(second));
  });

  test("saving a copy writes the original's bytes where the host says, replacing a file there", async () => {
    const {
      core,
      sources,
      documents: [report],
    } = await setUp({ "Report.pdf": REPORT });
    if (!report) throw new Error("Nothing was added.");
    const target = join(sources, "Saved", "My report.pdf");
    await mkdir(dirname(target));
    await writeFile(target, "An older file.");

    await core.saveDocumentCopy(report.id, target);

    expect(new Uint8Array(await readFile(target))).toEqual(REPORT);
  });

  test("deleted Documents are refused, even when a live Document shares the file", async () => {
    const {
      core,
      sources,
      tempDir,
      documents: [first],
    } = await setUp({ "notes.txt": "Shared text.\n" });
    if (!first) throw new Error("Nothing was added.");
    await core.deleteDocument(first.id);
    // The same file added again is a new Document with the same stored copy.
    const [second] = await addAndProcess(core, [join(sources, "notes.txt")]);
    if (!second) throw new Error("Nothing was added.");
    const target = join(sources, "copy.txt");

    await expect(core.documentCopyName(first.id)).rejects.toThrow(NotFoundError);
    await expect(core.saveDocumentCopy(first.id, target)).rejects.toThrow(NotFoundError);
    await expect(core.temporaryDocumentCopy(first.id)).rejects.toThrow(NotFoundError);
    expect(await exists(target)).toBe(false);
    expect(await exists(join(tempDir, OPEN_COPIES_FOLDER))).toBe(false);

    await core.saveDocumentCopy(second.id, target);
    expect(await readFile(target, "utf8")).toBe("Shared text.\n");
    expect(await readFile(await core.temporaryDocumentCopy(second.id), "utf8")).toBe(
      "Shared text.\n",
    );
  });

  test("unknown ids, paths that aren't absolute and missing files are refused, leaving nothing behind", async () => {
    const {
      core,
      dataDir,
      sources,
      tempDir,
      documents: [report],
    } = await setUp({ "Report.pdf": REPORT });
    if (!report) throw new Error("Nothing was added.");
    const target = join(sources, "copy.pdf");

    await expect(core.documentCopyName("no-such-document")).rejects.toThrow(NotFoundError);
    await expect(core.temporaryDocumentCopy("")).rejects.toThrow(InvalidInputError);
    await expect(core.saveDocumentCopy(report.id, "copy.pdf")).rejects.toThrow(InvalidInputError);
    await expect(core.saveDocumentCopy(report.id, 42 as unknown as string)).rejects.toThrow(
      InvalidInputError,
    );

    await rm(storedFile(dataDir, report.contentHash));
    await expect(core.saveDocumentCopy(report.id, target)).rejects.toThrow(NotFoundError);
    await expect(core.temporaryDocumentCopy(report.id)).rejects.toThrow(NotFoundError);
    expect(await exists(target)).toBe(false);
    expect(await readdir(join(tempDir, OPEN_COPIES_FOLDER))).toEqual([]);
  });

  test("at startup, copies to open made more than a day ago are removed; newer ones stay", async () => {
    const dataDir = await createTempDataFolder();
    const tempDir = await createTempDataFolder();
    const copies = join(tempDir, OPEN_COPIES_FOLDER);
    for (const folder of ["copy-old", "copy-new"]) {
      await mkdir(join(copies, folder), { recursive: true });
      await writeFile(join(copies, folder, "Report.pdf"), REPORT);
    }
    const twoDaysAgo = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000);
    await utimes(join(copies, "copy-old"), twoDaysAgo, twoDaysAgo);

    startCore(dataDir, { paths: { dataDir, tempDir } });

    await vi.waitFor(async () => expect(await readdir(copies)).toEqual(["copy-new"]));
  });
});
