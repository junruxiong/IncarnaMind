import { realpath, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  interceptOpenPath,
  interceptShowItemInFolder,
  launchApp,
  openDocumentMenu,
  pathsOpened,
  pathsShown,
  removeDataFolder,
} from "./app";

const REPORT = buildPdf([{ lines: ["Quarterly report"] }, { lines: ["Results"] }]);

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

test("Open in default app and Show in folder use the file where the User keeps it", async () => {
  await writeFile(join(sources, "Report.pdf"), REPORT);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [join(sources, "Report.pdf")]);
  // Never the real shell: the test hooks record what it would have been asked.
  await interceptOpenPath(app);
  await interceptShowItemInFolder(app);
  const original = await realpath(join(sources, "Report.pdf"));
  const item = window.getByTestId("document-list-item");

  // Renaming (from its menu) changes the Document's name, never its file.
  await openDocumentMenu(item);
  await window.getByRole("menuitem", { name: "Rename" }).click();
  await window.getByRole("textbox", { name: "New name for Report" }).fill("Q3 report");
  await window.keyboard.press("Enter");
  await expect(item).toContainText("Q3 report");

  await openDocumentMenu(item);
  await window.getByRole("menuitem", { name: "Open in default app" }).click();
  await expect.poll(() => pathsOpened(app)).toEqual([original]);

  await openDocumentMenu(item);
  await window.getByRole("menuitem", { name: "Show in folder" }).click();
  await expect.poll(() => pathsShown(app)).toEqual([original]);
  await app.close();
});
