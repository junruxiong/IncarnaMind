import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  addDocuments,
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  filterByTag,
  launchApp,
  openDocumentTags,
  openViewer,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

let dataDir: string;
/** Where the files a test adds are written: outside the data folder. */
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

test("a new Document is tagged automatically, and a Tag the User removes stays removed after a re-tag", async () => {
  const summary = join(sources, "Q3 summary.txt");
  await writeFile(summary, "Quarterly report for the sales team. Revenue grew in every region.\n");

  const first = await launchApp(dataDir, { fakeChat: true });
  const { window } = first;
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [summary]);

  // The scripted model applies every Tag whose name is a word in the Document: "Report".
  // Its row stays one line; its Tags menu shows its Tags.
  const item = window.getByTestId("document-list-item");
  await expect(item).toHaveAttribute("data-tagging", "tagged");
  let tags = await openDocumentTags(item);
  await expect(tags).toHaveText(["Report"]);
  await expect(tags).toHaveAttribute("data-source", "automatic");

  // The User takes it off, and adds another, in the same menu.
  await item.getByRole("menuitemcheckbox", { name: "Report", exact: true }).click();
  await expect(tags).toHaveCount(0);
  const invoice = item.getByRole("menuitemcheckbox", { name: "Invoice", exact: true });
  await invoice.click();
  await expect(invoice).toHaveAttribute("aria-checked", "true");
  await expect(tags).toHaveText(["Invoice"]);
  await expect(tags).toHaveAttribute("data-source", "user");
  await window.keyboard.press("Escape");
  await expect(item.getByTestId("document-tags-popover")).toBeHidden();

  // A new Tag, made in the Tags dialog (from the Documents label's Tags menu), which
  // re-tagging can apply.
  await window.getByTestId("tag-filter-menu").click();
  await window.getByTestId("manage-tags").click();
  const dialog = window.getByTestId("tags-dialog");
  await expect(dialog.getByTestId("tag-row")).toHaveCount(7);
  await dialog.getByTestId("tag-name-input").fill("Quarterly");
  await dialog.getByTestId("tag-description-input").fill("Covers one quarter of a year.");
  await dialog.getByTestId("save-tag").click();
  await expect(dialog.getByTestId("tag-row")).toHaveCount(8);
  await dialog.getByRole("button", { name: "Done" }).click();
  await expect(dialog).toBeHidden();

  // Re-tagging the Document applies the new Tag; the removed one stays off, the added one on.
  await item.hover();
  await item.getByTestId("document-tags-menu").click();
  await window.getByTestId("retag-document").click();
  await expect(item).toHaveAttribute("data-tagging", "tagged");
  tags = await openDocumentTags(item);
  await expect(tags).toHaveText(["Invoice", "Quarterly"]);
  await window.keyboard.press("Escape");

  // The sidebar filters by Tag, and says which; choosing it again shows them all.
  await filterByTag(window, "Report");
  await expect(item).toHaveCount(0);
  await expect(window.getByTestId("tag-filter-active")).toContainText("Report");
  await filterByTag(window, "Quarterly");
  await expect(item).toHaveCount(1);
  await expect(window.getByTestId("tag-filter-active")).toContainText("Quarterly");
  await filterByTag(window, "Quarterly");
  await expect(window.getByTestId("tag-filter-active")).toHaveCount(0);
  await window.getByTestId("tag-filter-menu").click();
  await expect(
    window
      .getByTestId("tag-filters")
      .getByRole("menuitemradio", { name: "Quarterly", exact: true }),
  ).toHaveAttribute("aria-checked", "false");
  await window.keyboard.press("Escape");
  await expect(item).toHaveCount(1);
  // The filter's row clears it too.
  await filterByTag(window, "Report");
  await expect(item).toHaveCount(0);
  await window.getByTestId("tag-filter-clear").click();
  await expect(item).toHaveCount(1);
  await first.app.close();

  // Everything is still there after a restart.
  const second = await launchApp(dataDir, { fakeChat: true });
  const restored = second.window.getByTestId("document-list-item");
  await expect(await openDocumentTags(restored)).toHaveText(["Invoice", "Quarterly"]);
  await second.app.close();
});

test("without a model, Documents are ready to search, and one notice, not one per Document, says tagging waits", async () => {
  const files = {
    "Ideas.txt": "A report on ideas.\n",
    "Plans.txt": "Plans for the next quarter.\n",
    "Minutes.md": "# Minutes\n\nNotes from the weekly meeting.\n",
  };
  const paths: string[] = [];
  for (const [name, text] of Object.entries(files)) {
    paths.push(join(sources, name));
    await writeFile(join(sources, name), text);
  }
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await addDocuments(window, paths);

  const items = window.getByTestId("document-list-item");
  await expect(items).toHaveCount(3);
  for (let index = 0; index < 3; index++) {
    await expect(items.nth(index).getByTestId("document-status")).toHaveText("Ready");
    await expect(items.nth(index)).toHaveAttribute("data-tagging", "waiting-for-provider");
  }
  // One notice, in the sidebar's footer; no tagging line under any Document, and no Tags.
  const notice = window.getByTestId("tagging-waiting");
  await expect(notice).toHaveCount(1);
  await expect(window.getByTestId("sidebar-footer").getByTestId("tagging-waiting")).toBeVisible();
  await expect(notice).toContainText("Automatic tagging is waiting for a model.");
  await expect(window.getByTestId("document-tagging")).toHaveCount(0);
  for (let index = 0; index < 3; index++) {
    await expect(await openDocumentTags(items.nth(index))).toHaveCount(0);
    await window.keyboard.press("Escape");
  }

  // Its button opens Settings on its Chat model page, where a chat model or a Jev key can be set up.
  await notice.getByTestId("tagging-waiting-setup").click();
  const settings = window.getByTestId("settings");
  await expect(settings).toBeVisible();
  await expect(settings).toHaveAttribute("data-page", "chat-model");
  await expect(settings.getByTestId("chat-model-settings")).toBeVisible();
  await expect(settings.getByTestId("jev-settings")).toBeVisible();
  await closeSettings(window);

  // Once a model is set up, the Documents are tagged and the notice goes.
  await useLocalChatModel(window);
  for (let index = 0; index < 3; index++) {
    await expect(items.nth(index)).toHaveAttribute("data-tagging", "tagged");
  }
  await expect(notice).toHaveCount(0);
  await app.close();
});

test("a Document's Tags menu near the window's bottom opens upward with its first item focused, and Esc closes only it", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const files = await Promise.all(
    ["One", "Two", "Three", "Four", "Five", "Six"].map(async (name) => {
      const path = join(sources, `${name}.txt`);
      await writeFile(path, `${name} notes about tides.`);
      return path;
    }),
  );
  await addDocuments(window, files);
  await openViewer(window);
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.setSize(1000, 470));
  await expect.poll(() => window.evaluate(() => globalThis.innerHeight)).toBeLessThan(600);

  const last = window.getByTestId("document-list-item").last();
  await last.hover();
  await last.getByTestId("document-tags-menu").click();
  const menu = last.getByTestId("document-tags-popover");
  await expect(menu).toBeVisible();
  const box = await menu.boundingBox();
  const height = await window.evaluate(() => globalThis.innerHeight);
  if (!box) throw new Error("The menu isn't visible.");
  expect(box.y).toBeGreaterThanOrEqual(0);
  expect(box.y + box.height).toBeLessThanOrEqual(height);
  await expect(menu.getByTestId("tag-menu-item").first()).toBeFocused();

  // Esc closes the menu, and leaves the Document viewer open.
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await expect(window.getByTestId("viewer")).toBeVisible();
  await app.close();
});
