import { truncate } from "node:fs/promises";
import { describe, expect, test, vi } from "vitest";
import { ExtractionError } from "../../src/core";
import { MAX_PACKAGE_BYTES } from "../../src/core/documents/formats/zip";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { docxOf } from "../helpers/office";

const reads = vi.hoisted(() => ({ openFile: vi.fn() }));
vi.mock("../../src/core/documents/files", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/core/documents/files")>();
  reads.openFile.mockImplementation(actual.openFile);
  return { ...actual, openFile: reads.openFile };
});

describe("opening an Office file for the viewer", { timeout: 30_000 }, () => {
  test("a Word, PowerPoint or Excel file over the limit is refused on its size, with nothing read; one under it opens", async () => {
    const sources = await createSourceFolder();
    const core = startCore(await createTempDataFolder());
    // Sparse: nothing is written on disk.
    const huge = await writeSourceFile(sources, "Huge.docx", "");
    await truncate(huge, MAX_PACKAGE_BYTES + 1);
    const normal = await writeSourceFile(sources, "Review.docx", docxOf([{ text: "Hello" }]));
    const [refused, indexed] = await addAndProcess(core, [huge, normal]);
    if (!refused || !indexed) throw new Error("Nothing was added.");
    reads.openFile.mockClear();

    const failure = await core.openDocumentFile(refused.id).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(ExtractionError);
    expect(failure).toMatchObject({
      reason: "too-large",
      message: "Too large to open (limit 500 MB).",
    });
    expect(reads.openFile).not.toHaveBeenCalled();

    const file = await core.openDocumentFile(indexed.id);
    await file.stream.cancel();
    expect(reads.openFile).toHaveBeenCalledOnce();
    expect(file.size).toBe(indexed.size);
    core.close();
  });
});
