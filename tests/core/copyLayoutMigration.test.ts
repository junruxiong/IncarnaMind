import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import * as Y from "yjs";
import {
  CITATION_NODE,
  DATABASE_FILE,
  MIND_CONTENT_FIELD,
  NotFoundError,
  QUESTION_BLOCK,
} from "../../src/core";
import { PROCESSING_VERSION } from "../../src/core/documents/processing";
import { migrate, openDatabase } from "../../src/core/storage";
import { migrations } from "../../src/core/storage/migrations";
import { createTempDataFolder, queryDatabase, startCore } from "../helpers/core";
import {
  createSourceFolder,
  sha256,
  storedFile,
  waitForDocuments,
  writeSourceFile,
} from "../helpers/documents";

const NOTES = "# Tide notes\n\nSpring tides happen at new moon and at full moon.\n";
const HASH = sha256(NOTES);
const DOCUMENT = "8a3e4c1e-2f8b-4d7a-9c1e-0b2a6d5e4f01";
const FOLDER = "5c7d2a90-6e14-4b3f-8a2d-1f0e9c8b7a62";
const TAG = "b1f0e2d3-4c5a-4b6e-8f7a-9d0c1b2a3e45";
const MIND = "e4d3c2b1-a0f9-4e8d-9c7b-6a5f4e3d2c10";
const QUESTION = "f0e1d2c3-b4a5-4968-8776-5a4b3c2d1e0f";
const AT = "2026-10-01T09:00:00.000Z";

/** A Mind with a Question scoped to the old Folder, and an Answer citing the Document. */
function mindContent(): Uint8Array {
  const doc = new Y.Doc();
  const blocks = doc.getXmlFragment(MIND_CONTENT_FIELD);
  const question = new Y.XmlElement(QUESTION_BLOCK);
  question.setAttribute("id", QUESTION);
  question.setAttribute("scopeFolderIds", [FOLDER] as unknown as string);
  question.insert(0, [new Y.XmlText("When are spring tides?")]);
  const paragraph = new Y.XmlElement("paragraph");
  const citation = new Y.XmlElement(CITATION_NODE);
  for (const [name, value] of Object.entries({
    passageId: "old-passage",
    documentId: DOCUMENT,
    documentName: "Tide notes",
    contentHash: HASH,
    quote: "Spring tides happen at new moon and at full moon.",
    check: "found",
  })) {
    citation.setAttribute(name, value);
  }
  paragraph.insert(0, [new Y.XmlText("At new and full moon "), citation]);
  blocks.insert(0, [question, paragraph]);
  return Y.encodeStateAsUpdate(doc);
}

/**
 * A data folder as the copy layout left it (before migration 21): one ready
 * Document copied into documents/ under its hash, filed in an in-app Folder,
 * with a Tag, and a Mind citing it.
 */
async function copyLayoutDataFolder(): Promise<string> {
  const dataDir = await createTempDataFolder();
  await mkdir(join(dataDir, "documents"));
  await writeFile(storedFile(dataDir, HASH), NOTES);
  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    migrate(
      db,
      migrations.filter((migration) => migration.version < 21),
    );
    db.run(
      `INSERT INTO folders (id, parent_id, name, created_at, updated_at) VALUES (?, NULL, 'Ocean', ?, ?)`,
      [FOLDER, AT, AT],
    );
    db.run(
      `INSERT INTO documents (id, content_hash, name, kind, size, status, page_count, folder_id,
         processing_version, tagging_status, created_at, updated_at)
       VALUES (?, ?, 'Tide notes', 'markdown', ?, 'ready', NULL, ?, ?, 'tagged', ?, ?)`,
      [DOCUMENT, HASH, Buffer.byteLength(NOTES), FOLDER, BigInt(PROCESSING_VERSION), AT, AT],
    );
    db.run(
      `INSERT INTO passages (id, document_id, position, page_from, page_to, window_from, window_to,
         text, created_at, updated_at)
       VALUES ('old-passage', ?, 0, NULL, NULL, 0, 0, ?, ?, ?)`,
      [DOCUMENT, NOTES, AT, AT],
    );
    db.run(
      `INSERT INTO document_pages (id, document_id, page, text, created_at, updated_at)
       VALUES ('old-page', ?, NULL, ?, ?, ?)`,
      [DOCUMENT, NOTES, AT, AT],
    );
    db.run(
      `INSERT INTO tags (id, name, description, preset, created_at, updated_at)
       VALUES (?, 'Oceans', '', NULL, ?, ?)`,
      [TAG, AT, AT],
    );
    db.run(
      `INSERT INTO document_tags (id, document_id, tag_id, source, created_at, updated_at)
       VALUES ('old-link', ?, ?, 'user', ?, ?)`,
      [DOCUMENT, TAG, AT, AT],
    );
    db.run(`INSERT INTO minds (id, title, created_at, updated_at) VALUES (?, 'Tides', ?, ?)`, [
      MIND,
      AT,
      AT,
    ]);
    db.run(
      `INSERT INTO mind_updates (id, mind_id, data, created_at, updated_at)
       VALUES ('old-update', ?, ?, ?, ?)`,
      [MIND, mindContent(), AT, AT],
    );
  } finally {
    db.close();
  }
  return dataDir;
}

/** The Citations' attributes in a Mind's state. */
function citationsIn(state: Uint8Array): Record<string, unknown>[] {
  const doc = new Y.Doc();
  Y.applyUpdate(doc, state);
  const found: Record<string, unknown>[] = [];
  const visit = (parent: Y.XmlFragment | Y.XmlElement) => {
    for (const child of parent.toArray()) {
      if (!(child instanceof Y.XmlElement)) continue;
      if (child.nodeName === CITATION_NODE) found.push(child.getAttributes());
      else visit(child);
    }
  };
  visit(doc.getXmlFragment(MIND_CONTENT_FIELD));
  return found;
}

describe("Migration 21: from copies in the data folder to Documents in place", {
  timeout: 30_000,
}, () => {
  test("copied Documents become files added on their own, in the data folder, keeping their Tags, Minds and Citations", async () => {
    const dataDir = await copyLayoutDataFolder();
    const legacy = join(dataDir, "documents", `${HASH}.md`);

    const core = startCore(dataDir);

    expect(await core.listDocuments()).toEqual([
      expect.objectContaining({
        id: DOCUMENT,
        name: "Tide notes",
        contentHash: HASH,
        // Named with its kind's extension, so another app can open it.
        path: legacy,
        fileStatus: "available",
        linkedFolderId: null,
        folderId: null,
        tags: [expect.objectContaining({ tagId: TAG, source: "user" })],
      }),
    ]);
    expect(existsSync(storedFile(dataDir, HASH))).toBe(false);
    expect(existsSync(legacy)).toBe(true);
    // Each Passage and page is of the version it was built from.
    expect(queryDatabase(dataDir, "SELECT content_hash FROM passages")).toEqual([
      { content_hash: HASH },
    ]);
    expect(queryDatabase(dataDir, "SELECT content_hash FROM document_pages")).toEqual([
      { content_hash: HASH },
    ]);
    // The Mind and its Citation are as they were, and the Citation still checks.
    const { state } = await core.openMind(MIND);
    expect(citationsIn(state)).toEqual([
      expect.objectContaining({ documentId: DOCUMENT, contentHash: HASH, check: "found" }),
    ]);
    expect(
      await core.recheckCitation({
        documentId: DOCUMENT,
        quote: "Spring tides happen at new moon and at full moon.",
        pageFrom: null,
        pageTo: null,
      }),
    ).toMatchObject({ check: "found", contentHash: HASH });
    const file = await core.openDocumentFile(DOCUMENT);
    expect(await new Response(file.stream).text()).toBe(NOTES);
  });

  test("the in-app Folders are dropped: their Documents are unfiled, and a Search scope naming one finds it deleted", async () => {
    const dataDir = await copyLayoutDataFolder();

    const core = startCore(dataDir);

    expect(await core.listFolders()).toEqual([]);
    await expect(core.listDocuments({ folderId: FOLDER })).rejects.toThrow(NotFoundError);
    expect(
      queryDatabase<{ deleted: number }>(
        dataDir,
        "SELECT deleted_at IS NOT NULL AS deleted FROM folders WHERE id = ?",
        [FOLDER],
      ),
    ).toEqual([{ deleted: 1 }]);
    // The Question still names the Folder; the editor shows it as a deleted Folder.
    const { state } = await core.openMind(MIND);
    const doc = new Y.Doc();
    Y.applyUpdate(doc, state);
    const question = doc.getXmlFragment(MIND_CONTENT_FIELD).get(0) as Y.XmlElement;
    expect(question.getAttribute("scopeFolderIds")).toEqual([FOLDER]);
  });

  test("linking the folder the original is in relinks the Document: same id, and the copy goes", async () => {
    const dataDir = await copyLayoutDataFolder();
    const papers = await createSourceFolder();
    const original = await writeSourceFile(papers, "Tide notes.md", NOTES);
    const core = startCore(dataDir);

    await core.addLinkedFolder(papers);
    await core.reconcileDocuments();

    const [document] = await waitForDocuments(core, (documents) => {
      expect(documents).toEqual([
        expect.objectContaining({
          id: DOCUMENT,
          path: original,
          linkedFolderId: expect.any(String),
          tags: [expect.objectContaining({ tagId: TAG })],
        }),
      ]);
    });
    expect(document?.contentHash).toBe(HASH);
    expect(existsSync(join(dataDir, "documents", `${HASH}.md`))).toBe(false);
    expect(citationsIn((await core.openMind(MIND)).state)[0]).toMatchObject({
      documentId: DOCUMENT,
    });
  });
});
