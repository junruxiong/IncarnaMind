import { access, readdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { describe, expect, test, vi } from "vitest";
import { ANSWER_BLOCK, CITATION_NODE, type Core, QUESTION_BLOCK } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { readMind } from "../helpers/minds";

/** The example Documents the app ships. */
const EXAMPLES = resolve(__dirname, "../../resources/examples");

async function startWithExamples() {
  const dataDir = await createTempDataFolder();
  const core = startCore(dataDir, { paths: { dataDir, examples: EXAMPLES } });
  return { core, dataDir };
}

/** The example Mind as the editor would read it. */
async function exampleMind(core: Core, mindId: string): Promise<ProseMirrorNode> {
  const client = await connectToMind(core, mindId);
  return readMind(client);
}

/** Each Citation's attributes, in order. */
function citationsOf(doc: ProseMirrorNode): Record<string, unknown>[] {
  const found: Record<string, unknown>[] = [];
  doc.descendants((node) => {
    if (node.type.name === CITATION_NODE) found.push(node.attrs);
  });
  return found;
}

describe("the example Mind", { timeout: 30_000 }, () => {
  test("is made on a first run, once, and its Citations quote its Documents and are found", async () => {
    const { core } = await startWithExamples();

    const made = await core.offerExamples();
    expect(made).toMatchObject({ available: true, mindId: expect.any(String) });
    // Offered once only, even after the example Mind is gone.
    expect(await core.offerExamples()).toBeNull();

    const mindId = made?.mindId as string;
    const [mind] = await core.listMinds();
    expect(mind).toMatchObject({ id: mindId, title: "Where tea comes from" });
    // Its Documents appear once the folder is scanned.
    await vi.waitFor(async () => {
      const documents = await core.listDocuments({ linkedFolderId: made?.linkedFolderId });
      expect(documents.map((each) => each.name).sort()).toEqual([
        "Tea · Wikipedia",
        "茶 · 维基百科",
      ]);
    });

    // Its blocks: a Note, the Question (scoped to the examples' folder), its Answer, an empty line.
    const doc = await exampleMind(core, mindId);
    expect(doc.content.content.map((block) => block.type.name)).toEqual([
      "paragraph",
      QUESTION_BLOCK,
      ANSWER_BLOCK,
      "paragraph",
    ]);
    expect(doc.child(1).textContent).toBe(
      "Where does tea come from, and how did it spread around the world?",
    );

    // Once both Documents are read, both Citations are checked: found, in the right sections.
    await vi.waitFor(
      async () => {
        const checks = citationsOf(await exampleMind(core, mindId)).map((each) => each.check);
        expect(checks).toEqual(["found", "found"]);
      },
      { timeout: 20_000, interval: 100 },
    );
    const [english, chinese] = citationsOf(await exampleMind(core, mindId));
    expect(english).toMatchObject({
      documentName: "Tea · Wikipedia",
      quote:
        "Tea drinking may have begun in the region of Yunnan, where it was used for medicinal purposes.",
      location: { kind: "section", heading: "Early tea drinking" },
    });
    expect(chinese).toMatchObject({
      documentName: "茶 · 维基百科",
      location: { kind: "section", heading: "在世界各地的传播" },
    });
  });

  test("are removed with their Documents and the copies of their files, and can be made again", async () => {
    const { core, dataDir } = await startWithExamples();
    const made = await core.createExamples();
    const folder = join(dataDir, "Examples");
    expect((await readdir(folder)).sort()).toEqual([
      "LICENSE",
      "Tea · Wikipedia.md",
      "茶 · 维基百科.md",
    ]);

    await core.removeExamples();
    expect(await core.getExamples()).toEqual({
      available: true,
      mindId: null,
      linkedFolderId: null,
    });
    expect(await core.listMinds()).toEqual([]);
    expect(await core.listLinkedFolders()).toEqual([]);
    await expect(access(folder)).rejects.toThrow();

    const again = await core.createExamples();
    expect(again.mindId).not.toBe(made.mindId);
    expect(await core.listMinds()).toHaveLength(1);
  });

  test("aren't offered when there are Minds already, nor when the app ships none", async () => {
    const { core } = await startWithExamples();
    await core.createMind({ title: "Mine" });
    expect(await core.offerExamples()).toBeNull();
    expect(await core.listMinds()).toHaveLength(1);

    const dataDir = await createTempDataFolder();
    const without = startCore(dataDir);
    expect(await without.offerExamples()).toBeNull();
    expect(await without.getExamples()).toEqual({
      available: false,
      mindId: null,
      linkedFolderId: null,
    });
  });
});
