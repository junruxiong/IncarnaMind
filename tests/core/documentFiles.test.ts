import { rm, writeFile } from "node:fs/promises";
import { describe, expect, test } from "vitest";
import { type DocumentFile, InvalidInputError, NotFoundError } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, waitForProcessing, writeSourceFile } from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";

/** Reads an opened Document file's stream to the end. */
async function readAll(file: DocumentFile): Promise<Uint8Array> {
  return new Uint8Array(await new Response(file.stream).arrayBuffer());
}

describe("opening a Document's file for the viewer", { timeout: 30_000 }, () => {
  test("streams the file where the User keeps it, byte for byte, with the Document and its size", async () => {
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
    expect(file.size).toBe(pdf.byteLength);
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

  test("an opened file is checked first: a changed one is indexed again, and served as it is now", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const path = await writeSourceFile(sources, "notes.md", "# Notes\n");
    const [document] = await addAndProcess(core, [path]);
    if (!document) throw new Error("Nothing was added.");

    await writeFile(path, "# Notes, edited\n\nWith a second paragraph.\n");
    const file = await core.openDocumentFile(document.id);

    expect(new TextDecoder().decode(await readAll(file))).toBe(
      "# Notes, edited\n\nWith a second paragraph.\n",
    );
    const [processed] = await waitForProcessing(core, [document.id]);
    expect(processed?.contentHash).not.toBe(document.contentHash);
    expect(await core.searchPassages("paragraph", { mode: "keyword" })).toHaveLength(1);
  });

  test("refuses a deleted Document, though a Document added again at its path is served", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const path = await writeSourceFile(sources, "notes.txt", "Shared text.\n");
    const [first] = await addAndProcess(core, [path]);
    if (!first) throw new Error("Nothing was added.");
    await core.deleteDocument(first.id);
    const [second] = await addAndProcess(core, [path]);
    if (!second) throw new Error("Nothing was added.");

    await expect(core.openDocumentFile(first.id)).rejects.toThrow(NotFoundError);
    expect(new TextDecoder().decode(await readAll(await core.openDocumentFile(second.id)))).toBe(
      "Shared text.\n",
    );
  });

  test("refuses unknown ids, and a Document whose file is missing, whose kept text is read instead", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    const path = await writeSourceFile(sources, "notes.txt", "Some text.\n");
    const [document] = await addAndProcess(core, [path]);
    if (!document) throw new Error("Nothing was added.");

    await expect(core.openDocumentFile("no-such-document")).rejects.toThrow(NotFoundError);
    await expect(core.openDocumentFile("")).rejects.toThrow(InvalidInputError);
    await expect(core.openDocumentFile(42 as unknown as string)).rejects.toThrow(InvalidInputError);

    await rm(path);
    await expect(core.openDocumentFile(document.id)).rejects.toThrow(NotFoundError);
    expect((await core.listDocuments())[0]?.fileStatus).toBe("missing");
    expect(await core.readDocumentText(document.id)).toEqual({
      documentId: document.id,
      contentHash: document.contentHash,
      fileStatus: "missing",
      pages: [{ page: null, text: "Some text.\n" }],
    });
    await expect(core.readDocumentText("no-such-document")).rejects.toThrow(NotFoundError);
  });
});
