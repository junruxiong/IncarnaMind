/**
 * A saved web page kept as it looks (#756): a PDF of each version, made in
 * the background only when the User asks, at the save or by the setting,
 * kept by its content beside the snapshots, served with the page, made again
 * after it failed or the app quit, and gone with the page. The window that
 * makes it stands in here.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { type Core, DATABASE_FILE } from "../../src/core";
import type { PageLook } from "../../src/core/adapters";
import type { Document, WebPageLookFailure } from "../../src/core/api";
import { openDatabase } from "../../src/core/storage";
import { createTempDataFolder, startCore } from "../helpers/core";
import { waitForDocuments, waitForProcessing } from "../helpers/documents";
import { startWeb } from "../helpers/web";

const REPORT = readFileSync(new URL("../fixtures/web/report.html", import.meta.url));

/** Makes "%PDF-" files that say which page and which time, or fails as told. */
function standInLook(fail: () => WebPageLookFailure | null = () => null) {
  const asked: string[] = [];
  const look: PageLook = {
    async render(address) {
      asked.push(address);
      const failure = fail();
      if (failure) return { failure, message: "As the stand-in was told." };
      return { pdf: new TextEncoder().encode(`%PDF-1.7 ${address} ${asked.length}`) };
    },
  };
  return { look, asked };
}

async function startSite() {
  let body = REPORT.toString("utf8");
  const web = await startWeb({ "/report": () => ({ status: 200, body }) });
  return {
    ...web,
    address: `http://127.0.0.1:${web.port}/report`,
    change: () => {
      body = body.replace("Harbour trade in 2026", "Harbour trade in 2026, revised");
    },
  };
}

async function saved(core: Core, input: { address: string; keepLook?: boolean }) {
  const result = await core.saveWebPage(input);
  if (result.outcome !== "saved") throw new Error(JSON.stringify(result));
  await waitForProcessing(core, [result.document.id]);
  return result.document.id;
}

const lookOf = async (core: Core, id: string) =>
  (await core.listDocuments()).find((document) => document.id === id)?.web?.look ?? null;

const settled = (core: Core, id: string) =>
  waitForDocuments(core, (documents) => {
    const look = documents.find((document) => document.id === id)?.web?.look;
    expect(look?.status === "saved" || look?.status === "failed").toBe(true);
  });

describe("A web page kept as it looks", { timeout: 30_000 }, () => {
  test("off by default: a page saved, fetched again or found by web search is never loaded for its look", async () => {
    const site = await startSite();
    const stand = standInLook();
    const core = startCore(await createTempDataFolder(), {
      ...site.adapters,
      pageLook: stand.look,
    });
    expect((await core.getSettings()).user.keepWebPageLook).toBe(false);

    const id = await saved(core, { address: site.address });
    site.change();
    await core.fetchWebPageAgain(id);
    await waitForProcessing(core, [id]);

    expect(await lookOf(core, id)).toBeNull();
    expect(stand.asked).toEqual([]);
  });

  test("asked at the save, the PDF is made in the background, kept by its content and served with the page", async () => {
    const site = await startSite();
    const stand = standInLook();
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir, { ...site.adapters, pageLook: stand.look });
    const statuses: (string | undefined)[] = [];
    core.on("document.status", (document: Document) => statuses.push(document.web?.look?.status));

    const id = await saved(core, { address: site.address, keepLook: true });
    const [page] = await settled(core, id);
    const look = (await core.listDocuments()).find((document) => document.id === id)?.web?.look;
    expect(statuses).toContain("saving");
    expect(look).toEqual({
      status: "saved",
      savedAt: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T.*Z$/),
      path: expect.stringMatching(/^looks\/[0-9a-f]{64}\.pdf$/),
    });
    expect(stand.asked).toEqual([site.address]);
    if (look?.status !== "saved" || !page) throw new Error("Not saved.");

    const file = join(dataDir, "Web pages", look.path);
    expect(readFileSync(file, "utf8")).toBe(`%PDF-1.7 ${site.address} 1`);
    const served = await core.openDocumentImage(id, look.path);
    expect(served).toMatchObject({ type: "application/pdf", size: readFileSync(file).byteLength });
    await served.stream.cancel();
    await expect(core.openDocumentImage(id, `looks/${"0".repeat(64)}.pdf`)).rejects.toThrow();
  });

  test("with the setting on, a save keeps it unless it says not to; fetched again, a new version gets its own and an unchanged one keeps its", async () => {
    const site = await startSite();
    const stand = standInLook();
    const core = startCore(await createTempDataFolder(), {
      ...site.adapters,
      pageLook: stand.look,
    });
    await core.updateSettings({ user: { keepWebPageLook: true } });

    const id = await saved(core, { address: site.address });
    await settled(core, id);
    const first = await lookOf(core, id);

    await core.fetchWebPageAgain(id);
    await waitForProcessing(core, [id]);
    expect(await lookOf(core, id)).toEqual(first);
    expect(stand.asked).toHaveLength(1);

    site.change();
    await core.fetchWebPageAgain(id);
    await waitForProcessing(core, [id]);
    await settled(core, id);
    const second = await lookOf(core, id);
    expect(second?.status).toBe("saved");
    expect(second).not.toEqual(first);
    expect(stand.asked).toHaveLength(2);

    const other = await startWeb({ "/other": () => ({ status: 200, body: REPORT }) });
    const otherId = await saved(core, {
      address: `http://127.0.0.1:${other.port}/other`,
      keepLook: false,
    });
    expect(await lookOf(core, otherId)).toBeNull();
    expect(stand.asked).toHaveLength(2);
  });

  test("one that failed says why and is made when the User tries again", async () => {
    const site = await startSite();
    let failure: WebPageLookFailure | null = "timed-out";
    const stand = standInLook(() => failure);
    const core = startCore(await createTempDataFolder(), {
      ...site.adapters,
      pageLook: stand.look,
    });

    const id = await saved(core, { address: site.address, keepLook: true });
    await settled(core, id);
    expect(await lookOf(core, id)).toEqual({ status: "failed", failure: "timed-out" });

    failure = null;
    await core.keepWebPageLook(id);
    await settled(core, id);
    expect((await lookOf(core, id))?.status).toBe("saved");
    expect(stand.asked).toHaveLength(2);
  });

  test("one being made when the app quit is made at the next start", async () => {
    const site = await startSite();
    const dataDir = await createTempDataFolder();
    const first = startCore(dataDir, site.adapters);
    const id = await saved(first, { address: site.address });
    first.close();
    const db = openDatabase(join(dataDir, DATABASE_FILE));
    try {
      db.run("UPDATE web_pages SET look_status = 'saving'");
    } finally {
      db.close();
    }

    const stand = standInLook();
    const second = startCore(dataDir, { ...site.adapters, pageLook: stand.look });
    await settled(second, id);
    expect((await lookOf(second, id))?.status).toBe("saved");
    expect(stand.asked).toEqual([site.address]);
  });

  test("deleted permanently, its PDF goes with it", async () => {
    const site = await startSite();
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir, { ...site.adapters, pageLook: standInLook().look });
    const id = await saved(core, { address: site.address, keepLook: true });
    await settled(core, id);
    const look = await lookOf(core, id);
    if (look?.status !== "saved") throw new Error("Not saved.");
    const file = join(dataDir, "Web pages", look.path);

    await core.deleteDocument(id);
    expect(existsSync(file)).toBe(true);
    await core.deletePermanently([{ kind: "document", id }]);
    await expect.poll(() => existsSync(file)).toBe(false);
  });
});
