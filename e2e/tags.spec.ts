import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
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
  const item = window.getByTestId("document-list-item");
  await expect(item).toHaveAttribute("data-tagging", "tagged");
  const chips = item.getByTestId("document-tag");
  await expect(chips).toHaveText(["Report"]);
  await expect(chips).toHaveAttribute("data-source", "automatic");

  // The User takes it off, and adds another from the Document's Tags menu.
  await chips.getByTestId("remove-document-tag").click();
  await expect(chips).toHaveCount(0);
  await item.getByTestId("document-tags-menu").click();
  const invoice = window.getByRole("menuitemcheckbox", { name: "Invoice" });
  await invoice.click();
  await expect(invoice).toHaveAttribute("aria-checked", "true");
  await window.keyboard.press("Escape");
  await expect(chips).toHaveText(["Invoice"]);
  await expect(chips).toHaveAttribute("data-source", "user");

  // A new Tag, made in the Tags dialog, which re-tagging can apply.
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
  await item.getByTestId("document-tags-menu").click();
  await window.getByTestId("retag-document").click();
  await expect(chips).toHaveText(["Invoice", "Quarterly"]);
  await expect(item).toHaveAttribute("data-tagging", "tagged");

  // The sidebar filters by Tag.
  const filters = window.getByTestId("tag-filters").getByTestId("tag-filter");
  await filters.filter({ hasText: "Report" }).click();
  await expect(item).toHaveCount(0);
  await filters.filter({ hasText: "Quarterly" }).click();
  await expect(item).toHaveCount(1);
  await filters.filter({ hasText: "Quarterly" }).click();
  await expect(filters.filter({ hasText: "Quarterly" })).toHaveAttribute("aria-pressed", "false");
  await expect(item).toHaveCount(1);
  await first.app.close();

  // Everything is still there after a restart.
  const second = await launchApp(dataDir, { fakeChat: true });
  await expect(second.window.getByTestId("document-tag")).toHaveText(["Invoice", "Quarterly"]);
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
  // One notice at the top of the Documents section; no tagging line under any Document.
  const notice = window.getByTestId("tagging-waiting");
  await expect(notice).toHaveCount(1);
  await expect(notice).toContainText("Automatic tagging is waiting for a model.");
  await expect(window.getByTestId("document-tagging")).toHaveCount(0);
  await expect(items.getByTestId("document-tag")).toHaveCount(0);

  // Its button opens Settings, where a chat model or a Jev key can be set up.
  await notice.getByTestId("tagging-waiting-setup").click();
  const settings = window.getByTestId("settings");
  await expect(settings).toBeVisible();
  await expect(settings.getByTestId("chat-model-settings")).toBeVisible();
  await expect(settings.getByTestId("jev-settings")).toBeVisible();
  await settings.getByRole("button", { name: "Done" }).click();
  await expect(settings).toBeHidden();

  // Once a model is set up, the Documents are tagged and the notice goes.
  await useLocalChatModel(window);
  for (let index = 0; index < 3; index++) {
    await expect(items.nth(index)).toHaveAttribute("data-tagging", "tagged");
  }
  await expect(notice).toHaveCount(0);
  await app.close();
});
