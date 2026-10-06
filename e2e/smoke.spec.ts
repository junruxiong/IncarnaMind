import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import { createDataFolder, dragBy, launchApp, openViewer, removeDataFolder, widthOf } from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("a Mind created before quitting is still there after reopening the app", async () => {
  // First run: create a Mind. It appears in the sidebar and opens in the centre.
  const first = await launchApp(dataDir);
  await first.window.getByTestId("new-mind").click();

  const created = first.window.getByTestId("mind-list-item");
  await expect(created).toHaveCount(1);
  const mindId = await created.getAttribute("data-mind-id");
  expect(mindId).toBeTruthy();
  await expect(first.window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${mindId}`);
  await expect(first.window.getByTestId("mind-title")).toBeVisible();
  await first.app.close();

  // Second run on the same data folder: the Mind is listed and opens.
  const second = await launchApp(dataDir);
  const restored = second.window.getByTestId("mind-list-item");
  await expect(restored).toHaveCount(1);
  await expect(restored).toHaveAttribute("data-mind-id", `${mindId}`);
  await restored.click();
  await expect(second.window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${mindId}`);
  await expect(second.window.getByTestId("mind-title")).toBeVisible();
  await second.app.close();
});

test("what was typed in a Mind, and its title, are still there after reopening the app", async () => {
  const paragraphs = ["Typed before quitting.", "A second paragraph."];

  const first = await launchApp(dataDir);
  await first.window.getByTestId("new-mind").click();
  await first.window.getByTestId("mind-title").fill("Field notes");
  const editor = first.window.getByTestId("mind-editor");
  await editor.click();
  await first.window.keyboard.type(paragraphs[0] ?? "");
  await first.window.keyboard.press("Enter");
  await first.window.keyboard.type(paragraphs[1] ?? "");
  await expect(editor.locator("p")).toHaveText(paragraphs);
  await expect(first.window.getByTestId("mind-list-item")).toHaveText("Field notes");
  await first.app.close();

  const second = await launchApp(dataDir);
  const item = second.window.getByTestId("mind-list-item");
  await expect(item).toHaveText("Field notes");
  await item.click();
  await expect(second.window.getByTestId("mind-title")).toHaveValue("Field notes");
  await expect(second.window.getByTestId("mind-editor").locator("p")).toHaveText(paragraphs);
  await second.app.close();
});

test("the sidebar lists the most recently edited Mind first, and a deleted Mind leaves it", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  const items = window.getByTestId("mind-list-item");
  for (const title of ["Older", "Newer"]) {
    await window.getByTestId("new-mind").click();
    await window.getByTestId("mind-title").fill(title);
    await expect(items.first()).toHaveText(title);
  }
  await expect(items).toHaveText(["Newer", "Older"]);

  // Editing the older Mind moves it to the top.
  await items.filter({ hasText: "Older" }).click();
  await window.getByTestId("mind-editor").click();
  await window.keyboard.type("An edit");
  await expect(items).toHaveText(["Older", "Newer"]);

  // Deleting it asks first, then removes it from the sidebar and closes it.
  const older = window.getByRole("listitem").filter({ hasText: "Older" });
  await older.hover();
  await older.getByTestId("delete-mind").click();
  await window.getByTestId("confirm-delete-mind").click();
  await expect(items).toHaveText(["Newer"]);
  await expect(window.getByTestId("mind-pane")).toHaveCount(0);
  await first.app.close();

  // It stays deleted after a restart.
  const second = await launchApp(dataDir);
  await expect(second.window.getByTestId("mind-list-item")).toHaveText(["Newer"]);
  await second.app.close();
});

test("the Document viewer is hidden until opened, resizes from its left edge and keeps its width", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  const viewer = window.getByTestId("viewer");
  const mindArea = window.getByTestId("mind-area");

  // Closed on launch: no panel at all, and the Mind area takes the rest of the width.
  await expect(viewer).toHaveCount(0);
  const fullWidth = await widthOf(mindArea);

  // Opening it narrows the Mind area.
  await openViewer(window);
  await expect(viewer).toBeVisible();
  expect(await widthOf(mindArea)).toBeLessThan(fullWidth);

  // Dragging the left edge 100px to the left widens the panel by 100px.
  const openedWidth = await widthOf(viewer);
  await dragBy(window, window.getByTestId("viewer-resize"), -100);
  await expect.poll(() => widthOf(viewer)).toBe(openedWidth + 100);

  // Esc closes it and the Mind area gets its full width back.
  await window.keyboard.press("Escape");
  await expect(viewer).toHaveCount(0);
  expect(await widthOf(mindArea)).toBe(fullWidth);
  await first.app.close();

  // The width is a per-device setting, so it survives a restart. The close button closes it.
  const second = await launchApp(dataDir);
  await expect(second.window.getByTestId("viewer")).toHaveCount(0);
  await openViewer(second.window);
  await expect.poll(() => widthOf(second.window.getByTestId("viewer"))).toBe(openedWidth + 100);
  await second.window.getByTestId("viewer-close").click();
  await expect(second.window.getByTestId("viewer")).toHaveCount(0);
  await second.app.close();
});

test("added files are processed, show as ready in the sidebar, and can be deleted", async () => {
  // The User's own files live outside the data folder.
  const sources = await createDataFolder();
  try {
    const notes = join(sources, "Reading notes.txt");
    const report = join(sources, "Report.pdf");
    await writeFile(notes, "Attention lets a model focus on the most relevant words.\n");
    await writeFile(
      report,
      buildPdf([{ lines: ["Quarterly report"] }, { lines: ["Revenue grew by ten percent."] }]),
    );

    const first = await launchApp(dataDir);
    const { window } = first;
    await window.getByTestId("add-documents-input").setInputFiles([notes, report]);

    const items = window.getByTestId("document-list-item");
    await expect(items).toHaveCount(2);
    const notesItem = items.filter({ hasText: "Reading notes" });
    const reportItem = items.filter({ hasText: "Report" });
    await expect(notesItem).toHaveAttribute("data-status", "ready");
    await expect(reportItem).toHaveAttribute("data-status", "ready");
    await expect(notesItem.getByTestId("document-status")).toHaveText("Ready");
    await first.app.close();

    // After a restart both are still there. Deleting one asks first, then removes it.
    const second = await launchApp(dataDir);
    const restored = second.window.getByTestId("document-list-item");
    await expect(restored).toHaveCount(2);
    await restored.filter({ hasText: "Reading notes" }).getByTestId("delete-document").click();
    await second.window.getByTestId("confirm-delete-document").click();
    await expect(restored).toHaveCount(1);
    await expect(restored).toHaveAttribute("data-status", "ready");
    await second.app.close();
  } finally {
    await removeDataFolder(sources);
  }
});
