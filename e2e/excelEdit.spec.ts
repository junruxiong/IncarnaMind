/**
 * Spike (excel-edit): a workbook in a Linked folder edited in the sheet
 * viewer, its totals recalculated, saved as a copy beside the other exports
 * while the original stays byte for byte, and the copy cited on the cell that
 * changed. With INCARNAMIND_SPIKE_BIG naming a workbook of 10,000 rows or
 * more, it also times how long that one takes to open.
 */
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, realpath } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  documentIdOf,
  launchApp,
  linkFolderFromSidebar,
  openDocumentAt,
  removeDataFolder,
  setWindowSize,
  useOutputLocation,
} from "./app";

const FIXTURES = join(__dirname, "..", "tests", "fixtures", "formats");
const sha = async (path: string) =>
  createHash("sha256")
    .update(await readFile(path))
    .digest("hex");

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

test("a workbook edited in the viewer recalculates, saves a copy and leaves the original alone, and the copy is cited on its cell", async () => {
  const library = join(await realpath(sources), "Library");
  const output = join(await realpath(sources), "Exports");
  await mkdir(library, { recursive: true });
  await mkdir(output, { recursive: true });
  const original = join(library, "Regional Revenue.xlsx");
  await copyFile(join(FIXTURES, "Regional Revenue.xlsx"), original);
  const before = await sha(original);

  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1800, 1000);
  await useOutputLocation(window, output);
  await linkFolderFromSidebar(app, window, library);
  await expect(window.getByTestId("document-list-item")).toHaveAttribute("data-status", "ready");

  await openDocumentAt(window, { documentId: await documentIdOf(window, "Regional Revenue") });
  const viewer = window.getByTestId("viewer");
  const grid = viewer.getByTestId("viewer-grid");
  await expect(grid.locator('td[data-ref="F7"]')).toHaveText("£1,627,700");

  await viewer.getByTestId("sheet-edit").click();
  await expect(viewer.getByTestId("sheet-edit-bar")).toBeVisible();
  const west = grid.locator('td[data-ref="B7"]');
  await west.hover();
  await west.click();
  await expect(viewer.getByTestId("sheet-edit-ref")).toHaveText("B7");
  await expect(viewer.getByTestId("sheet-formula-input")).toHaveValue("350200");
  await window.keyboard.type("400000");
  await expect(viewer.getByTestId("sheet-cell-input")).toHaveValue("400000");
  await window.keyboard.press("Enter");

  // The cell keeps its format, and what depends on it is worked out again.
  await expect(west).toHaveText("£400,000");
  await expect(grid.locator('td[data-ref="F7"]')).toHaveText("£1,677,500");
  await expect(grid.locator('td[data-ref="B8"]')).toHaveText("£1,630,400");
  await expect(grid.locator('td[data-ref="F8"]')).toHaveText("£6,861,900");
  await expect(viewer.getByTestId("sheet-edit-count")).toHaveText("1 cell changed");
  await viewer.getByTestId("sheet-selected").waitFor();
  await expect(viewer.getByTestId("sheet-edit-ref")).toHaveText("B8");
  await expect(viewer.getByTestId("sheet-formula-input")).toHaveValue("=SUM(B4:B7)");

  await viewer.getByTestId("sheet-save-copy").click();
  await expect(viewer.getByTestId("sheet-edit-saved")).toContainText(
    "Saved “Regional Revenue (edited).xlsx” in Exports",
  );
  expect(await sha(original)).toBe(before);

  // The copy is a Document of its own, its totals the new ones; a Citation of the new total opens on its cell.
  await expect(window.getByTestId("document-list-item")).toHaveCount(2);
  const copy = await documentIdOf(window, "Regional Revenue (edited)");
  await openDocumentAt(window, { documentId: copy, pageFrom: 1, pageTo: 1, quote: "1,677,500" });
  const copyGrid = viewer.getByTestId("viewer-grid");
  const cited = copyGrid.locator("td[data-quote-highlight]");
  await expect(cited).toHaveCount(1);
  await expect(cited).toHaveAttribute("data-ref", "F7");
  await expect(cited).toHaveText("£1,677,500");
  await expect(cited).toBeInViewport();
  await app.close();
});

test("a workbook of 10,000 rows or more opens in the viewer, and its calculator starts", async () => {
  const big = process.env.INCARNAMIND_SPIKE_BIG;
  test.skip(!big, "Set INCARNAMIND_SPIKE_BIG to a workbook of 10,000 rows or more.");
  const name = (big as string).split("/").pop() as string;
  const file = join(await realpath(sources), name);
  await copyFile(big as string, file);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1600, 900);
  await addDocuments(window, [file]);
  const id = await documentIdOf(window, name.replace(/\.xlsx$/, ""));
  await window.evaluate(() => {
    (globalThis as { openedAt?: number }).openedAt = performance.now();
  });
  await openDocumentAt(window, { documentId: id });
  const viewer = window.getByTestId("viewer");
  await expect(viewer.getByTestId("viewer-grid").locator('td[data-ref="A2"]')).toBeVisible({
    timeout: 30_000,
  });
  const openMs = await window.evaluate(
    () => performance.now() - ((globalThis as { openedAt?: number }).openedAt ?? 0),
  );
  const started = await window.evaluate(() => performance.now());
  await viewer.getByTestId("sheet-edit").click();
  await expect(viewer.getByTestId("sheet-edit-bar")).toBeVisible({ timeout: 30_000 });
  const editMs = (await window.evaluate(() => performance.now())) - started;
  const cell = viewer.getByTestId("viewer-grid").locator('td[data-ref="D2"]');
  await cell.hover();
  await cell.click();
  const typed = await window.evaluate(() => performance.now());
  await window.keyboard.type("7");
  await window.keyboard.press("Enter");
  await expect(viewer.getByTestId("sheet-edit-count")).toHaveText("1 cell changed");
  const recalcMs = (await window.evaluate(() => performance.now())) - typed;
  console.log(
    `SPIKE open ${Math.round(openMs)} ms; edit mode ${Math.round(editMs)} ms; edit and recalculate ${Math.round(recalcMs)} ms`,
  );
  await app.close();
});
