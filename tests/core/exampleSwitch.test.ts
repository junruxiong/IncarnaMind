import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { expect, test } from "vitest";
import * as Y from "yjs";
import type { Examples, LinkedFolder, Mind } from "../../src/core";
import { createExamples } from "../../src/core/examples";
import type { Language } from "../../src/core/language";
import { createTempDataFolder } from "../helpers/core";

const EXAMPLES = resolve(__dirname, "../../resources/examples");

/**
 * The example's pieces over fakes, where a linked folder's Documents are
 * reported at the first moment the event loop is free: the earliest the real
 * library can report them.
 */
async function exampleOverFakes() {
  const dataDir = await createTempDataFolder();
  const values = new Map<string, unknown>();
  const minds = new Map<string, { mind: Mind; blocks: Y.XmlFragment }>();
  const folders = new Set<string>();
  let language: Language = "en";
  let shown: Examples | undefined;
  const reports: { folder: string; shownFolder: string | null | undefined }[] = [];

  const examples = createExamples({
    source: EXAMPLES,
    dataDir,
    language: () => language,
    readValue: (key) => values.get(key),
    writeValue: (key, value) => void values.set(key, value),
    createMind: (title) => {
      const mind: Mind = {
        id: randomUUID(),
        title,
        autoTitle: "",
        kind: "mind",
        createdAt: "2026-10-16T09:00:00.000Z",
        updatedAt: "2026-10-16T09:00:00.000Z",
        folderId: null,
      };
      minds.set(mind.id, { mind, blocks: new Y.Doc().getXmlFragment("content") });
      return mind;
    },
    renameMind: (mindId, title) => {
      const held = minds.get(mindId);
      if (held) held.mind = { ...held.mind, title };
    },
    mindTitle: (mindId) => minds.get(mindId)?.mind.title ?? "",
    mindExists: (mindId) => minds.has(mindId),
    deleteMind: (mindId) => void minds.delete(mindId),
    editMind: (mindId, change) => {
      const held = minds.get(mindId);
      if (!held) throw new Error("No such Mind.");
      return change(held.blocks);
    },
    linkFolder: async (path) => {
      const id = randomUUID();
      folders.add(id);
      setImmediate(() => reports.push({ folder: id, shownFolder: shown?.linkedFolderId }));
      return {
        id,
        path,
        folderId: randomUUID(),
        status: "scanning",
        layout: "tree",
        progress: { files: 0, indexed: 0 },
        onlineOnly: { files: 0, bytes: 0, downloading: false },
        unsupportedFiles: 0,
        createdAt: "2026-10-16T09:00:00.000Z",
        updatedAt: "2026-10-16T09:00:00.000Z",
      } satisfies LinkedFolder;
    },
    unlinkFolder: (linkedFolderId) => void folders.delete(linkedFolderId),
    linkedFolderExists: (linkedFolderId) => folders.has(linkedFolderId),
    documentsIn: () => [],
    mindFolder: () => null,
    now: () => "2026-10-16T09:00:00.000Z",
    checksRelyOn: () => false,
    recheck: () => {
      throw new Error("No Document has been read.");
    },
    changed: (value) => {
      shown = value;
    },
  });

  return {
    examples,
    reports,
    switchTo: async (next: Language) => {
      language = next;
      await examples.followLanguage();
      await new Promise((resolve) => setImmediate(resolve));
    },
  };
}

// protects=the interface never holds an example's Documents as the User's own, while the example is made or follows the language; fails_when=an example's folder is linked and the core waits (for the old files to be deleted) before it says the folder is the example's, so the Documents are reported as the User's own; no_claim=what the renderer does with them.
test("an example's Documents are known to be the example's before they are reported, when it follows the language too", async () => {
  const { examples, reports, switchTo } = await exampleOverFakes();

  await examples.create();
  await switchTo("zh-CN");
  await switchTo("en");

  expect(reports).toHaveLength(3);
  for (const report of reports) expect(report.shownFolder).toBe(report.folder);
});
