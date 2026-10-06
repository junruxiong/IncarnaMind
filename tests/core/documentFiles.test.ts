import { rm } from "node:fs/promises";
import { describe, expect, test } from "vitest";
import { type DocumentFile, InvalidInputError, NotFoundError } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, storedFile, writeSourceFile } from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";

/** Reads an opened Document file's stream to the end. */
async function readAll(file: DocumentFile): Promise<Uint8Array> {
  return new Uint8Array(await new Response(file.stream).arrayBuffer());
}

describe("opening a Document's file for the viewer", { timeout: 30_000 }, () => {
  test("streams a live Document's stored copy, byte for byte, with the Document", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const pdf = buildPdf([{ lines: ["Page one"] }, { lines: ["Page two"] }]);
    const [document] = await addAndProcess(core, [
      await writeSourceFile(sources, "Report.pdf", pdf),
    ]);
    if (!document) throw new Error("Nothing was added.");

    const file = await core.openDocumentFile(document.id);

    expect(file.document).toEqual(document);
    expect(await readAll(file)).toEqual(pdf);
  });

  test("works while the Document is still being processed", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const {
      documents: [document],
    } = await core.addDocuments([await writeSourceFile(sources, "notes.md", "# Notes\n")]);
    if (!document) throw new Error("Nothing was added.");

    const file = await core.openDocumentFile(document.id);

    expect(file.document.id).toBe(document.id);
    expect(new TextDecoder().decode(await readAll(file))).toBe("# Notes\n");
  });

  test("refuses a deleted Document, even when another Document still uses the same file", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const path = await writeSourceFile(sources, "notes.txt", "Shared text.\n");
    const [first] = await addAndProcess(core, [path]);
    if (!first) throw new Error("Nothing was added.");
    await core.deleteDocument(first.id);
    // The same file added again is a new Document with the same stored copy.
    const [second] = await addAndProcess(core, [path]);
    if (!second) throw new Error("Nothing was added.");

    await expect(core.openDocumentFile(first.id)).rejects.toThrow(NotFoundError);
    expect(new TextDecoder().decode(await readAll(await core.openDocumentFile(second.id)))).toBe(
      "Shared text.\n",
    );
  });

  test("refuses unknown ids, and a live Document whose file has gone from the data folder", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const [document] = await addAndProcess(core, [
      await writeSourceFile(sources, "notes.txt", "Some text.\n"),
    ]);
    if (!document) throw new Error("Nothing was added.");

    await expect(core.openDocumentFile("no-such-document")).rejects.toThrow(NotFoundError);
    await expect(core.openDocumentFile("")).rejects.toThrow(InvalidInputError);
    await expect(core.openDocumentFile(42 as unknown as string)).rejects.toThrow(InvalidInputError);

    await rm(storedFile(dataDir, document.contentHash));
    await expect(core.openDocumentFile(document.id)).rejects.toThrow(NotFoundError);
  });
});
