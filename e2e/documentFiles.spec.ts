import { readFile, realpath, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  interceptOpenPath,
  interceptSaveDialog,
  launchApp,
  pathsOpened,
  removeDataFolder,
  saveDialogsAsked,
} from "./app";

const REPORT = buildPdf([{ lines: ["Quarterly report"] }, { lines: ["Results"] }]);
const NOTES = "# Field notes\n\nThe key finding was that attention spans shrink after lunch.\n";

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

test("Save a copy… writes a Document's original where the User chose, under its name", async () => {
  await writeFile(join(sources, "Report.pdf"), REPORT);
  await writeFile(join(sources, "Field notes.markdown"), NOTES);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [join(sources, "Report.pdf"), join(sources, "Field notes.markdown")]);
  const item = window.getByTestId("document-list-item").filter({ hasText: "Report" });

  // Renamed, the Document's name is what the copy is called.
  await item.hover();
  await item.getByRole("button", { name: "Rename Report" }).click();
  await window.getByRole("textbox", { name: "New name for Report" }).fill("Q3 report");
  await window.keyboard.press("Enter");
  await expect(item).toContainText("Q3 report");

  // The system save dialog is answered by the test hook.
  const target = join(sources, "Saved copy.pdf");
  await interceptSaveDialog(app, target);
  await item.hover();
  await item.getByTestId("document-file-menu").click();
  await window.getByRole("menuitem", { name: "Save a copy…" }).click();

  await expect.poll(() => readFile(target).catch(() => null)).not.toBeNull();
  expect(new Uint8Array(await readFile(target))).toEqual(REPORT);
  const [asked] = await saveDialogsAsked(app);
  expect(basename(asked?.defaultPath ?? "")).toBe("Q3 report.pdf");
  expect(asked?.filters).toEqual([{ name: "PDF document", extensions: ["pdf"] }]);

  // A Markdown Document is saved as Markdown.
  const notesTarget = join(sources, "Notes copy.md");
  await interceptSaveDialog(app, notesTarget);
  const notes = window.getByTestId("document-list-item").filter({ hasText: "Field notes" });
  await notes.hover();
  await notes.getByTestId("document-file-menu").click();
  await window.getByRole("menuitem", { name: "Save a copy…" }).click();
  await expect.poll(() => readFile(notesTarget, "utf8").catch(() => null)).toBe(NOTES);
  const [notesAsked] = await saveDialogsAsked(app);
  expect(basename(notesAsked?.defaultPath ?? "")).toBe("Field notes.md");
  expect(notesAsked?.filters).toEqual([{ name: "Markdown", extensions: ["md"] }]);
  await app.close();
});

test("Open in default app opens a copy named after the Document, not its stored file", async () => {
  await writeFile(join(sources, "Report.pdf"), REPORT);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [join(sources, "Report.pdf")]);
  await interceptOpenPath(app);
  const item = window.getByTestId("document-list-item");

  await item.hover();
  await item.getByTestId("document-file-menu").click();
  await window.getByRole("menuitem", { name: "Open in default app" }).click();

  await expect.poll(() => pathsOpened(app)).toHaveLength(1);
  const [opened] = await pathsOpened(app);
  expect(basename(opened ?? "")).toBe("Report.pdf");
  // A copy outside the data folder: the stored file, named by its hash, stays where it is.
  expect(await realpath(opened ?? "")).not.toContain(await realpath(dataDir));
  expect(new Uint8Array(await readFile(opened ?? ""))).toEqual(REPORT);
  await app.close();
});
