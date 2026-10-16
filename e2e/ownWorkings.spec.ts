import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  openDocumentMenu,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/*
 * A Document the User keeps as their own workings (#897): "Don't cite this
 * Document" in the shared Document menu, in the sidebar and the Library, a
 * "Not cited" label beside its name, Undo by the undo line and ⌘Z, and, in a
 * check of a sentence only that file states, the card's own line. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (target: Page | Locator, name: string) => {
  if (SCREENSHOTS) await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const REPORT = buildPdf([
  { lines: ["Annual report", "Revenue was $1,482.6 million in 2025, up from $1,231.0 million."] },
]);
const FIGURE = "Penguins nest on the harbour cliffs.";

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

/** The Annual report and the User's workings, added: a spreadsheet of their own figures and a line of notes. */
async function addReportAndWorkings(window: Page) {
  await writeFile(join(sources, "Annual report.pdf"), REPORT);
  await writeFile(
    join(sources, "Workings.csv"),
    "Item,2024,2025\nRevenue (USD million),1231.0,1482.6\nOperating margin,0.18,0.21\n",
  );
  await writeFile(join(sources, "Field notes.txt"), `${FIGURE}\n`);
  await dismissChatSetup(window);
  await addDocuments(window, [
    join(sources, "Annual report.pdf"),
    join(sources, "Workings.csv"),
    join(sources, "Field notes.txt"),
  ]);
}

const sidebarRow = (window: Page, name: string) =>
  window.getByTestId("document-list-item").filter({ hasText: name });

const libraryRow = (window: Page, name: string) =>
  window
    .getByTestId("library")
    .getByTestId("library-document")
    .filter({ has: window.getByTestId("library-document-name").getByText(name, { exact: true }) });

async function openLibraryMenu(row: Locator): Promise<void> {
  await row.hover();
  await row.getByTestId("document-file-menu").click();
}

test("Don't cite this Document in the sidebar: the label shows beside its name, Undo and ⌘Z take it back, and the menu says Cite this Document again", async () => {
  const { window } = await launchApp(dataDir);
  await addReportAndWorkings(window);
  const workings = sidebarRow(window, "Workings");
  const report = sidebarRow(window, "Annual report");
  const label = workings.getByTestId("document-not-cited");
  await expect(label).toHaveCount(0);

  // The item has the hint, and the report has the item too.
  await openDocumentMenu(workings);
  const item = workings.getByTestId("document.dontCite");
  await expect(item).toHaveText("Don't cite this Document");
  await expect(item).toHaveAttribute(
    "title",
    "IncarnaMind can still read it to understand your question, but won't cite it or use it to support a check.",
  );
  await expect(workings.getByTestId("document.citeAgain")).toHaveCount(0);
  await pointAndClick(window, item);

  await expect(label).toHaveText("Not cited");
  await expect(report.getByTestId("document-not-cited")).toHaveCount(0);
  const undo = window.getByTestId("undo-line");
  await expect(undo.getByTestId("undo-message")).toHaveText("“Workings” won't be cited");
  // The label sits right of the name, inside the row, and the name stays on one line.
  const [name, labelBox, row] = await Promise.all([
    workings.getByTestId("row-text").boundingBox(),
    label.boundingBox(),
    workings.boundingBox(),
  ]);
  if (!name || !labelBox || !row) throw new Error("The row isn't laid out.");
  expect(labelBox.x).toBeGreaterThanOrEqual(name.x + name.width);
  expect(labelBox.x + labelBox.width).toBeLessThanOrEqual(row.x + row.width);
  expect(labelBox.height).toBeLessThanOrEqual(row.height);
  await screenshot(window.getByTestId("sidebar"), "not-cited-sidebar-en");

  // ⌘Z takes it back; the menu offers it again, and Undo takes that back too.
  await window.keyboard.press("ControlOrMeta+z");
  await expect(label).toHaveCount(0);
  await openDocumentMenu(workings);
  await pointAndClick(window, workings.getByTestId("document.dontCite"));
  await expect(label).toBeVisible();
  await openDocumentMenu(workings);
  const again = workings.getByTestId("document.citeAgain");
  await expect(again).toHaveText("Cite this Document again");
  await expect(workings.getByTestId("document.dontCite")).toHaveCount(0);
  await pointAndClick(window, again);
  await expect(label).toHaveCount(0);
  await expect(undo.getByTestId("undo-message")).toHaveText("“Workings” can be cited again");
  await undo.getByTestId("undo-action").click();
  await expect(label).toBeVisible();
});

test("In the Library and in Chinese: the same menu item and label, one Undo for several Documents, and the label holds at the narrowest window", async () => {
  const { app, window } = await launchApp(dataDir);
  await addReportAndWorkings(window);
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 900, 720);
  await window.getByTestId("open-library").click();
  const workings = libraryRow(window, "Workings");
  const notes = libraryRow(window, "Field notes");
  await expect(workings).toBeVisible();

  await openLibraryMenu(workings);
  const item = workings.getByTestId("document.dontCite");
  await expect(item).toHaveText("不作为引用来源");
  await expect(item).toHaveAttribute(
    "title",
    "IncarnaMind 仍会读它来理解你的问题，但不会引用它，也不会用它来支持核对。",
  );
  await pointAndClick(window, item);
  const label = workings.getByTestId("document-not-cited");
  await expect(label).toHaveText("不引用");
  await expect(window.getByTestId("undo-line").getByTestId("undo-message")).toHaveText(
    "“Workings”不再作为引用来源",
  );
  // In the row: after the name, before the Tags column, never wrapping or leaving the row.
  const [name, labelBox, rowBox] = await Promise.all([
    workings.getByTestId("library-document-name").boundingBox(),
    label.boundingBox(),
    workings.boundingBox(),
  ]);
  if (!name || !labelBox || !rowBox) throw new Error("The row isn't laid out.");
  expect(labelBox.x).toBeGreaterThanOrEqual(name.x + name.width);
  expect(labelBox.x + labelBox.width).toBeLessThanOrEqual(rowBox.x + rowBox.width);
  expect(labelBox.height).toBeLessThan(24);
  await screenshot(window.getByTestId("library"), "not-cited-library-zh-narrow");

  // The sidebar shows it too.
  await expect(sidebarRow(window, "Workings").getByTestId("document-not-cited")).toHaveText(
    "不引用",
  );

  // ⌘Z takes it back. Two selected: one item names both, one undo line, and one ⌘Z.
  await window.keyboard.press("ControlOrMeta+z");
  await expect(label).toHaveCount(0);
  await pointAndClick(window, workings.getByTestId("library-document-name"), {
    modifiers: ["ControlOrMeta"],
  });
  await pointAndClick(window, notes.getByTestId("library-document-name"), {
    modifiers: ["ControlOrMeta"],
  });
  await openLibraryMenu(notes);
  await pointAndClick(window, notes.getByTestId("document.dontCite"));
  await expect(notes.getByTestId("document-not-cited")).toHaveText("不引用");
  await expect(label).toHaveText("不引用");
  await expect(window.getByTestId("undo-line").getByTestId("undo-message")).toHaveText(
    "2 个文档不再作为引用来源",
  );
  await window.keyboard.press("ControlOrMeta+z");
  await expect(label).toHaveCount(0);
  await expect(notes.getByTestId("document-not-cited")).toHaveCount(0);
});

test("A sentence only the marked file states is Unsupported, and its card says so, in English and Chinese", async () => {
  const { window } = await launchApp(dataDir, { fakeChat: true });
  await useLocalChatModel(window);
  await addReportAndWorkings(window);
  await openDocumentMenu(sidebarRow(window, "Field notes"));
  await pointAndClick(window, sidebarRow(window, "Field notes").getByTestId("document.dontCite"));
  await expect(sidebarRow(window, "Field notes").getByTestId("document-not-cited")).toBeVisible();

  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type(FIGURE);
  const line = editor.locator(":scope > p").first();
  await expect(line).toHaveText(FIGURE);
  await slideSelect(window, line, "Penguins", { line, words: "cliffs." });
  const menu = window.getByTestId("selection-menu");
  await pointAndClick(window, menu.getByTestId("selection-action-find-support"));

  const mark = window.getByTestId("claim-mark");
  await expect(mark).toHaveAttribute("data-check", "unsupported");
  await pointAndClick(window, mark);
  const card = window.getByTestId("claim-card");
  await expect(card.getByTestId("claim-badge")).toHaveText("Unsupported");
  await expect(card.getByTestId("claim-own-workings")).toHaveText(
    "Only your own working file says this.",
  );
  await expect(card.getByTestId("claim-nothing")).toHaveCount(0);
  await screenshot(window, "not-cited-card-en");

  await setLanguage(window, "zh-CN");
  await expect(card.getByTestId("claim-own-workings")).toHaveText("只有你自己的工作文件这样写。");
  await screenshot(window, "not-cited-card-zh");
});
