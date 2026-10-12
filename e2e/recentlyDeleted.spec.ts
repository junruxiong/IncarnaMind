import { access, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  coreOf,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openDocumentMenu,
  pointAndClick,
  removeDataFolder,
  setWindowSize,
  sidebarFolder,
} from "./app";

/*
 * Recently deleted (#228): deleting a Mind, Document, Folder or Tag asks nothing and shows
 * the undo line; the Library's Recently deleted view lists them, newest first, with Restore
 * and Delete Permanently, the only question. Set INCARNAMIND_SCREENSHOTS to a folder to also
 * save screenshots there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** Reads the core through the bridge (or changes it, as from another window). */
function read<T>(window: Page, query: (core: CoreBridge) => Promise<T>): Promise<T>;
function read<T, A>(
  window: Page,
  query: (core: CoreBridge, arg: A) => Promise<T>,
  arg: A,
): Promise<T>;
async function read(
  window: Page,
  query: (core: CoreBridge, arg?: unknown) => Promise<unknown>,
  arg?: unknown,
): Promise<unknown> {
  return (await coreOf(window)).evaluate(query, arg);
}

/** What the core holds that deleting and restoring touch, to compare before and after. */
const stateOf = (window: Page) =>
  read(window, async (core) => {
    const library = await core.getLibrary();
    const byName = (a: string, b: string) => a.localeCompare(b);
    return {
      folders: library.groups.map((group) => `${group.id}:${group.name}`).sort(byName),
      minds: (await core.listMinds()).map((mind) => `${mind.title}:${mind.folderId}`).sort(byName),
      documents: (await core.listDocuments())
        .map(
          (document) =>
            `${document.name}:${
              library.assignments.find((item) => item.documentId === document.id)?.groupId ?? null
            }`,
        )
        .sort(byName),
      tags: (await core.listTags()).map((tag) => tag.name).sort(byName),
    };
  });

const deletedNames = (window: Page) =>
  read(window, async (core) => (await core.listDeleted()).map((item) => item.name));

const undoLine = (window: Page) => window.getByTestId("undo-line");
const noQuestion = (window: Page) => expect(window.locator("dialog[open]")).toHaveCount(0);
const rows = (window: Page) => window.getByTestId("deleted-item");
const row = (window: Page, name: string) => rows(window).filter({ hasText: name });

async function openDeleted(window: Page): Promise<Locator> {
  await pointAndClick(window, window.getByTestId("open-library"));
  await pointAndClick(window, window.getByTestId("library-recently-deleted"));
  const view = window.getByTestId("recently-deleted");
  await expect(view).toBeVisible();
  return view;
}

test("deleting a Mind, a Document, a Folder and a Tag asks nothing and lands in Recently deleted; Restore puts each back exactly", async () => {
  test.setTimeout(120_000);
  const invoice = join(sources, "Invoice 4471.txt");
  await writeFile(invoice, "Invoice 4471: the rent for October is due on the first.\n");
  const { window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const finance = await read(
    window,
    async (core) => (await core.createLibraryGroup({ name: "Finance", description: "" })).id,
  );
  await addDocuments(window, [invoice]);
  await read(
    window,
    async (core, folderId) => {
      const [document] = await core.listDocuments();
      if (document) await core.assignDocumentGroup(document.id, folderId);
      await core.createMind({ title: "Shortlist", folderId });
      await core.createTag({ name: "Vendor" });
    },
    finance,
  );
  const before = await stateOf(window);
  expect(before.minds).toEqual([`Shortlist:${finance}`]);
  expect(before.documents).toEqual([`Invoice 4471:${finance}`]);

  // A Mind, from its menu, in its open Folder.
  await sidebarFolder(window, "Finance").getByTestId("folder-row").click();
  const mind = window.getByTestId("mind-row").filter({ hasText: "Shortlist" });
  await mind.hover();
  await mind.getByTestId("mind-menu").click();
  await mind.getByTestId("mind-actions").getByTestId("mind.delete").click();
  await expect(undoLine(window).getByTestId("undo-message")).toHaveText(
    "Moved “Shortlist” to Recently deleted",
  );
  await screenshot(undoLine(window), "undo-line-en");
  await noQuestion(window);

  // A Document, from its menu.
  const document = window.getByTestId("document-list-item").filter({ hasText: "Invoice 4471" });
  await openDocumentMenu(document);
  await document.getByTestId("document-file-actions").getByTestId("document.delete").click();
  await expect(undoLine(window).getByTestId("undo-message")).toHaveText(
    "Moved “Invoice 4471” to Recently deleted",
  );
  await noQuestion(window);

  // A Tag, in Manage tags.
  await pointAndClick(window, window.getByTestId("open-library"));
  await pointAndClick(window, window.getByRole("button", { name: "Manage tags" }));
  const slides = window.locator('[data-testid="tag-row"]', { hasText: "Vendor" });
  await pointAndClick(window, slides.getByTestId("delete-tag"));
  await expect(slides).toHaveCount(0);
  await expect(window.locator("dialog[open]")).toHaveCount(1);
  await pointAndClick(window, window.getByRole("button", { name: "Done" }));
  await expect(undoLine(window).getByTestId("undo-message")).toHaveText(
    "Moved “Vendor” to Recently deleted",
  );

  // A Folder, from its menu.
  const folder = sidebarFolder(window, "Finance");
  await folder.getByTestId("folder-row").hover();
  await folder.getByTestId("folder-menu").click();
  await folder.getByTestId("folder.delete").click();
  await expect(undoLine(window).getByTestId("undo-message")).toHaveText(
    "Moved “Finance” to Recently deleted",
  );
  await noQuestion(window);
  await expect
    .poll(() => deletedNames(window))
    .toEqual(["Finance", "Vendor", "Invoice 4471", "Shortlist"]);

  // The view lists them newest first, each with where it was and its 30 days.
  const view = await openDeleted(window);
  await expect(rows(window).getByTestId("deleted-name")).toHaveText([
    "Finance",
    "Vendor",
    "Invoice 4471",
    "Shortlist",
  ]);
  await expect(row(window, "Shortlist").getByTestId("deleted-was-in")).toHaveText(
    "Not in a Folder",
  );
  await expect(row(window, "Invoice 4471").getByTestId("deleted-was-in")).toHaveText(
    "Not in a Folder",
  );
  await expect(row(window, "Vendor").getByTestId("deleted-was-in")).toHaveText("Tags");
  await expect(row(window, "Finance").getByTestId("deleted-was-in")).toHaveText("Library");
  await expect(rows(window).getByTestId("deleted-left")).toHaveText(Array(4).fill("30 days"));
  await expect(rows(window).getByTestId("deleted-when").first()).toContainText("Today");
  await screenshot(window, "recently-deleted-rows-en");

  // Selecting them all restores them together, to where they were.
  await row(window, "Finance").getByTestId("deleted-select").check();
  await row(window, "Shortlist")
    .getByTestId("deleted-select")
    .click({ modifiers: ["Shift"] });
  await expect(window.getByTestId("deleted-selection")).toContainText("4 selected");
  await window.getByTestId("deleted-restore-selected").click();
  await expect(undoLine(window).getByTestId("undo-message")).toHaveText("Restored 4 items");
  await expect(window.getByTestId("deleted-nothing")).toBeVisible();
  expect(await stateOf(window)).toEqual(before);

  // Undo deletes them again; one more Undo... is the stack's: Restore is one change.
  await undoLine(window).getByTestId("undo-action").click();
  await expect.poll(() => deletedNames(window)).toHaveLength(4);
  await expect(rows(window)).toHaveCount(4);

  // One row's own button restores it alone, in its Folder.
  await row(window, "Finance").getByTestId("deleted-restore").click();
  await expect(undoLine(window).getByTestId("undo-message")).toHaveText("Restored “Finance”");
  await row(window, "Shortlist").getByTestId("deleted-restore").click();
  await expect(undoLine(window).getByTestId("undo-message")).toHaveText(
    "Restored “Shortlist” to Finance",
  );
  await expect(view).toBeVisible();
});

test("Delete Permanently asks once and says what is lost; the Document's file stays, and it can't be restored", async () => {
  test.setTimeout(120_000);
  const file = join(sources, "Tides.txt");
  await writeFile(file, "Neap tides are the smallest of the month.\n");
  const { window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [file]);
  const documentId = await read(window, async (core) => (await core.listDocuments())[0]?.id ?? "");
  const item = window.getByTestId("document-list-item").filter({ hasText: "Tides" });
  await openDocumentMenu(item);
  await item.getByTestId("document-file-actions").getByTestId("document.delete").click();
  await expect(item).toHaveCount(0);
  await access(file);

  await openDeleted(window);
  await row(window, "Tides").getByTestId("deleted-delete").click();
  const dialog = window.getByTestId("delete-permanently-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("heading")).toHaveText("Delete “Tides” permanently?");
  await expect(dialog).toContainText(
    "IncarnaMind forgets the text it read from “Tides” and its Tags.",
  );
  await expect(dialog).toContainText("Your files stay where they are.");
  await screenshot(window, "delete-permanently-en");

  // Cancel keeps it; Esc does too.
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toBeHidden();
  await expect(rows(window)).toHaveCount(1);
  await row(window, "Tides").getByTestId("deleted-delete").click();
  await window.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(rows(window)).toHaveCount(1);

  await row(window, "Tides").getByTestId("deleted-delete").click();
  await dialog.getByTestId("confirm-delete-permanently").click();
  await expect(window.getByTestId("deleted-nothing")).toBeVisible();
  await access(file);
  const refused = await read(
    window,
    async (core, id) =>
      core.restore([{ kind: "document", id }]).then(
        () => "restored",
        (error: Error) => error.message,
      ),
    documentId,
  );
  expect(refused).toContain("deleted permanently");
});

test("emptying Recently deleted with many items asks once, counting what is lost", async () => {
  test.setTimeout(120_000);
  const { window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await read(window, async (core) => {
    for (let index = 1; index <= 12; index++) {
      const mind = await core.createMind({ title: `Draft ${index}` });
      await core.deleteMind(mind.id);
    }
  });
  await openDeleted(window);
  await expect(rows(window)).toHaveCount(12);
  await window.getByTestId("deleted-empty").click();
  const dialog = window.getByTestId("delete-permanently-dialog");
  await expect(dialog.getByRole("heading")).toHaveText("Empty Recently deleted?");
  await expect(dialog).toContainText("The notes in 12 Minds are gone.");
  await dialog.getByTestId("confirm-delete-permanently").click();
  await expect(window.getByTestId("deleted-nothing")).toHaveText(
    "Nothing was deleted in the last 30 days.",
  );
  expect(await deletedNames(window)).toEqual([]);
  await screenshot(window, "recently-deleted-empty-en");
});

test("the view holds at a narrow window and in Chinese, with a very long name", async () => {
  test.setTimeout(120_000);
  const longName = "The annual report of the harbour authority on tides and ferries ".repeat(3);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await read(
    window,
    async (core, name) => {
      for (const title of [name, "季度报告"]) {
        const mind = await core.createMind({ title });
        await core.deleteMind(mind.id);
      }
      await core.updateSettings({ user: { language: "zh-CN" } });
    },
    longName.trim(),
  );
  await setWindowSize(app, window, 900, 660);
  await openDeleted(window);
  await expect(window.getByTestId("deleted-heading")).toHaveText("回收站");
  await expect(rows(window)).toHaveCount(2);
  const list = window.getByTestId("recently-deleted");
  const overflow = await list.evaluate((element) => {
    const scroller = element.querySelector<HTMLElement>(".overflow-y-auto");
    return (scroller?.scrollWidth ?? 0) - (scroller?.clientWidth ?? 0);
  });
  expect(overflow).toBeLessThanOrEqual(0);
  // The name is cut by the row, not wrapped by letter; the actions stay inside it.
  const long = row(window, "harbour authority");
  const name = await long.getByTestId("deleted-name").boundingBox();
  const box = await long.boundingBox();
  if (!name || !box) throw new Error("The row isn't showing.");
  expect(name.x + name.width).toBeLessThanOrEqual(box.x + box.width);
  await expect(long.getByTestId("deleted-restore")).toHaveText("恢复");
  await expect(long.getByTestId("deleted-delete")).toHaveText("彻底删除…");
  await expect(long.getByTestId("deleted-left")).toHaveText("30 天");
  await screenshot(window, "recently-deleted-zh-narrow");

  await long.getByTestId("deleted-delete").click();
  const dialog = window.getByTestId("delete-permanently-dialog");
  await expect(dialog).toContainText("此操作无法撤销。你的文件仍在原处。");
  await screenshot(window, "delete-permanently-zh-narrow");
  await window.keyboard.press("Escape");
  await setWindowSize(app, window, 1600, 660);
  await screenshot(window, "recently-deleted-zh-wide");
});
