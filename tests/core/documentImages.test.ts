/**
 * The pictures a Markdown Document shows from beside it (ADR-0011): the
 * viewer asks for "figures/map.png" as written in the file, and gets that
 * image from the Markdown file's folder, never from anywhere else.
 */
import { mkdir, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { NotFoundError } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";

const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "base64",
);

describe("an image beside a Markdown Document", { timeout: 30_000 }, () => {
  test("is served from the Markdown file's folder or below, with its type", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const core = startCore(dataDir);
    await mkdir(join(sources, "figures"));
    await writeFile(join(sources, "figures", "site map.png"), PNG);
    await writeFile(join(sources, "logo.svg"), "<svg xmlns='http://www.w3.org/2000/svg'/>");
    const [notes] = await addAndProcess(core, [
      await writeSourceFile(sources, "notes.md", "# Notes\n\n![Map](figures/site%20map.png)\n"),
    ]);
    if (!notes) throw new Error("Nothing was added.");

    const image = await core.openDocumentImage(notes.id, "figures/site map.png");
    expect(image.type).toBe("image/png");
    expect(image.size).toBe(PNG.length);
    expect(Buffer.from(await new Response(image.stream).arrayBuffer())).toEqual(PNG);
    const svg = await core.openDocumentImage(notes.id, "./logo.svg");
    expect(svg.type).toBe("image/svg+xml");
    await svg.stream.cancel();
  });

  test("anything else is refused: other folders, other kinds of file, other Documents", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createTempDataFolder();
    const outside = await createTempDataFolder();
    const core = startCore(dataDir);
    await mkdir(join(sources, "notes"));
    await writeFile(join(outside, "secret.png"), PNG);
    await writeFile(join(sources, "notes", "data.csv"), "a,b\n1,2\n");
    await symlink(join(outside, "secret.png"), join(sources, "notes", "linked.png"));
    const [notes, report] = await addAndProcess(core, [
      await writeSourceFile(sources, join("notes", "notes.md"), "# Notes\n"),
      await writeSourceFile(sources, "report.txt", "A report.\n"),
    ]);
    if (!notes || !report) throw new Error("Nothing was added.");

    for (const path of [
      "../../outside/secret.png",
      join(outside, "secret.png"),
      "linked.png",
      "data.csv",
      "missing.png",
      "",
    ]) {
      await expect(core.openDocumentImage(notes.id, path)).rejects.toBeInstanceOf(NotFoundError);
    }
    await writeFile(join(sources, "picture.png"), PNG);
    // A plain-text Document shows no images.
    await expect(core.openDocumentImage(report.id, "picture.png")).rejects.toBeInstanceOf(
      NotFoundError,
    );
  });
});
