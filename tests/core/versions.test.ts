import { writeFile } from "node:fs/promises";
import { describe, expect, test } from "vitest";
import type { Core } from "../../src/core";
import { citationState } from "../../src/shared/citations";
import {
  answerEnded,
  askNew,
  citationsIn,
  citingModel,
  onlyCitation,
  setUpWithDocuments,
} from "../helpers/citations";
import { queryDatabase, startCore } from "../helpers/core";
import { sha256, waitForProcessing } from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";

const SPRING = "Spring tides happen at new moon and at full moon.";

/** The first version: the quote is on page 2. */
const V1 = buildPdf([
  { lines: ["Tides and the Moon", "The Moon's gravity raises two tidal bulges on the Earth."] },
  { lines: ["Spring and neap tides", SPRING] },
]);
/** The second: the sentence was rewritten, so the quote is nowhere. */
const V2 = buildPdf([
  { lines: ["Tides and the Moon", "The Moon's gravity raises two tidal bulges on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides follow the new and the full moon."] },
]);
/** The third: the quote is back, a page later. */
const V3 = buildPdf([
  { lines: ["Tides and the Moon", "The Moon's gravity raises two tidal bulges on the Earth."] },
  { lines: ["A note on the Sun", "The Sun's pull adds to the Moon's."] },
  { lines: ["Spring and neap tides", SPRING] },
]);

/** The page rows kept of each version of the Documents, by content hash. */
const pagesByVersion = (dataDir: string) =>
  Object.fromEntries(
    queryDatabase<{ content_hash: string; pages: number }>(
      dataDir,
      `SELECT content_hash, count(*) AS pages FROM document_pages
       WHERE deleted_at IS NULL GROUP BY content_hash`,
    ).map((row) => [row.content_hash, row.pages]),
  );

/** Changes a file added on its own, and waits for its new version to be processed. */
async function changeFile(core: Core, documentId: string, path: string, contents: Uint8Array) {
  await writeFile(path, contents);
  await core.reconcileDocuments();
  const [processed] = await waitForProcessing(core, [documentId]);
  expect(processed?.contentHash).toBe(sha256(contents));
}

/**
 * A Citation of version 1, made while the file changed to version 2 under
 * the Answer being written, and then version 3.
 */
async function citeThenChange() {
  let release = () => {};
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  const answer = "Spring tides come at new and full moon [^1].";
  const model = citingModel({
    query: "spring tides",
    records: (passages) => [
      { marker: 1, passage: passages[0]?.id, pageFrom: 2, pageTo: 2, quote: SPRING },
    ],
    answer,
    pause: { after: answer.length, until: released },
  });
  const setup = await setUpWithDocuments(model, [{ name: "Tides.pdf", contents: V1 }]);
  const { core, client, mind, documents } = setup;
  const document = documents[0];
  if (!document) throw new Error("Nothing was added.");

  const answerId = await askNew(core, client, mind.id, "When are spring tides?");
  await expect.poll(() => citationsIn(client, answerId).length).toBe(1);
  // The file changes before the Answer finishes, so the check runs once version 2 is current.
  await changeFile(core, document.id, document.path, V2);
  const ended = answerEnded(core, answerId);
  release();
  expect((await ended).event).toBe("finished");
  return { ...setup, document, answerId };
}

describe("Versions of a Document", { timeout: 30_000 }, () => {
  test("a Citation is checked against the version it quoted, and says when the Document changed after it was cited", async () => {
    const { core, client, document, answerId } = await citeThenChange();

    const citation = onlyCitation(client, answerId);
    expect(citation).toMatchObject({
      documentId: document.id,
      contentHash: sha256(V1),
      pageFrom: 2,
      pageTo: 2,
      check: "found",
      checkReason: null,
    });
    const documents = await core.listDocuments();
    expect(documents[0]?.contentHash).toBe(sha256(V2));
    expect(citationState(citation, documents)).toEqual({
      check: "found",
      reason: null,
      documentId: document.id,
      changedAfterCited: true,
    });
    // Search and new Answers use the current version only: "happen" was only in version 1.
    expect(await core.searchPassages("happen", { mode: "keyword" })).toEqual([]);
    expect(await core.searchPassages("follow", { mode: "keyword" })).toHaveLength(1);

    // Checked again against the current version, where the quote isn't.
    const input = { documentId: document.id, quote: SPRING, pageFrom: 2, pageTo: 2 };
    expect(await core.recheckCitation(input)).toEqual({
      check: "not-found",
      checkReason: "quote-not-on-pages",
      contentHash: sha256(V2),
      passageId: expect.any(String),
      pageFrom: 2,
      pageTo: 2,
    });
    // In version 3 the quote is back, on page 3: the recheck finds it there.
    await changeFile(core, document.id, document.path, V3);
    const rechecked = await core.recheckCitation(input);
    expect(rechecked).toEqual({
      check: "found",
      checkReason: null,
      contentHash: sha256(V3),
      passageId: expect.any(String),
      pageFrom: 3,
      pageTo: 3,
    });
    // Written onto the Citation, it no longer says the Document changed.
    expect(
      citationState({ ...citation, ...rechecked }, await core.listDocuments()).changedAfterCited,
    ).toBe(false);
    await expect(core.recheckCitation({ ...input, documentId: "gone" })).resolves.toMatchObject({
      check: "cant-check",
      checkReason: "document-removed",
    });
  });

  test("old versions' text is kept while a Citation quotes it, and goes at a start once none does", async () => {
    const { core, dataDir, document, mind } = await citeThenChange();
    await changeFile(core, document.id, document.path, V3);
    // Version 1 is cited, version 2 isn't, version 3 is current.
    expect(pagesByVersion(dataDir)).toEqual({
      [sha256(V1)]: 2,
      [sha256(V2)]: 2,
      [sha256(V3)]: 3,
    });
    core.close();

    const second = startCore(dataDir);
    expect(pagesByVersion(dataDir)).toEqual({ [sha256(V1)]: 2, [sha256(V3)]: 3 });
    expect(
      queryDatabase(dataDir, "SELECT 1 FROM passages WHERE content_hash = ?", [sha256(V2)]),
    ).toEqual([]);
    // The Citation goes with its Mind: nothing quotes version 1 any more.
    await second.deleteMind(mind.id);
    second.close();

    startCore(dataDir);
    expect(pagesByVersion(dataDir)).toEqual({ [sha256(V3)]: 3 });
  });
});
