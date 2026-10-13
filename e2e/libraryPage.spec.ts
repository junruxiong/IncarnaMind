import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  placeAssistant,
  pointAndClick,
  pointAt,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  sidebarFolder,
} from "./app";
import { addLibrary, FIXTURE_COUNT, FOLDERS, NOT_IN_A_FOLDER, SCAN_NAME } from "./libraryFixture";

/*
 * The Library as one page grouped by Folder (#240): its head, one section per
 * Folder that folds and remembers it, Not in a Folder last and Recently deleted
 * folded at the foot, rows of 40px with no native controls, search, filters
 * and the problem line narrowing every section, Move to… for Documents, a
 * Folder from the sidebar brought into view, and columns that give way at a
 * narrow window. On a Library of about 100 Documents in every format, with
 * English and Chinese names, by the mouse (slid onto each target) and the keyboard.
 */

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

const DELETED = ["Holiday rota", "Office floor plan"];
const LOOSE = NOT_IN_A_FOLDER.length - DELETED.length;

/** Opens a Library of about 100 Documents, two of them deleted since. */
async function openFullLibrary(window: Page): Promise<Locator> {
  await dismissChatSetup(window);
  await addLibrary(window, sources);
  await window.evaluate(async (names) => {
    const api = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const documents = await api.listDocuments();
    for (const name of names) {
      const doc = documents.find((each) => each.name === name);
      if (doc) await api.deleteDocument(doc.id);
    }
  }, DELETED);
  await window.getByTestId("open-library").click();
  const library = window.getByTestId("library");
  await expect(library.getByTestId("library-count")).toHaveText(
    String(FIXTURE_COUNT - DELETED.length),
  );
  return library;
}

const sections = (library: Locator) => library.getByTestId("library-section");

const section = (library: Locator, name: string) =>
  sections(library).filter({
    has: library.page().locator('[data-testid="library-section-name"]', {
      hasText: new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`),
    }),
  });

const rowsOf = (scope: Locator) => scope.getByTestId("library-document");

const row = (scope: Locator, name: string) =>
  rowsOf(scope).filter({
    has: scope.page().getByTestId("library-document-name").getByText(name, { exact: true }),
  });

const left = (locator: Locator) =>
  locator.evaluate((element) => element.getBoundingClientRect().left);

test("the page is grouped by Folder in the sidebar's order, with Not in a Folder last and Recently deleted folded at the foot, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1600, 900);
    const library = await openFullLibrary(window);
    await expect(library.getByTestId("library-heading")).toContainText("Library");
    await expect(window.getByTestId("tab-title").filter({ hasText: "Library" })).toHaveCount(1);
    await expect(window.getByTestId("open-library")).toContainText("Library");

    // One section per Folder, in the sidebar's order, then Not in a Folder.
    const sidebarNames = await window
      .getByTestId("library-folders")
      .getByTestId("library-folder")
      .locator(':scope > div [data-testid="row-text"]')
      .allTextContents();
    expect(sidebarNames).toHaveLength(FOLDERS.length);
    await expect(sections(library).getByTestId("library-section-name")).toHaveText([
      ...sidebarNames,
      "Not in a Folder",
    ]);
    for (const folder of FOLDERS) {
      await expect(section(library, folder.name)).toHaveAttribute(
        "data-count",
        String(folder.documents.length),
      );
    }
    await expect(section(library, "Not in a Folder")).toHaveAttribute("data-count", String(LOOSE));
    const finance = section(library, "Finance");
    await expect(finance.getByTestId("library-section-head")).toContainText(
      "Invoices, budgets and statements",
    );

    // A long section lists its first 8, then "Show all 17".
    await expect(rowsOf(finance)).toHaveCount(8);
    await pointAndClick(window, finance.getByTestId("library-show-all"));
    await expect(rowsOf(finance)).toHaveCount(17);
    await expect(finance.getByTestId("library-show-all")).toHaveCount(0);

    // Rows: name, Tags, format and when it was added; no native select or disclosure anywhere.
    const invoice = row(finance, "Invoice 4471");
    await expect(invoice.getByTestId("tag-chip")).toHaveText("Invoice");
    await expect(invoice.getByTestId("library-document-format")).toHaveText("PDF");
    await expect(invoice.getByTestId("library-document-added")).not.toHaveText("");
    await expect(library.locator("select, details, summary")).toHaveCount(0);
    const tallest = () =>
      rowsOf(library).evaluateAll((rows) =>
        Math.max(...rows.map((each) => each.getBoundingClientRect().height)),
      );
    expect(await tallest()).toBeLessThanOrEqual(40);

    // The words of the title, the search, + Filter and the first section and row start on one edge.
    const edge = await left(library.getByTestId("library-heading"));
    const plus = library.getByTestId("library-add-filter").locator("svg");
    for (const each of [
      library.getByTestId("library-search").locator("xpath=.."),
      plus,
      sections(library).first().getByTestId("library-section-head"),
      rowsOf(library).first(),
    ]) {
      expect(Math.abs((await left(each)) - edge)).toBeLessThanOrEqual(1);
    }
    // Every menu button opens with the app's chevron, never a typed caret.
    await expect(library.getByTestId("library-add-documents").locator("svg")).toHaveCount(2);
    await expect(library.getByTestId("library-add-documents")).not.toContainText("▾");

    // The problem, in one line with one action.
    await expect(library.getByTestId("library-problem")).toHaveText(
      "1 Document has no text: a scan · Show",
    );

    // Recently deleted waits folded at the foot; opened, each row offers Restore.
    const deleted = window.getByTestId("deleted-section-head");
    await expect(deleted).toHaveAttribute("data-open", "false");
    await expect(deleted.getByTestId("library-section-count")).toHaveText("2");
    await expect(window.getByTestId("deleted-item")).toHaveCount(0);
    await pointAndClick(window, deleted.getByTestId("deleted-heading"));
    await expect(window.getByTestId("deleted-item")).toHaveCount(2);
    await expect(window.getByTestId("deleted-name")).toHaveText(DELETED.toReversed());

    // In Chinese, the same page in its words, and rows still 40px.
    await setLanguage(window, "zh-CN");
    await expect(library.getByTestId("library-heading")).toContainText("资料库");
    await expect(sections(library).getByTestId("library-section-name").last()).toHaveText(
      "不在文件夹中",
    );
    await expect(library.getByTestId("library-problem")).toHaveText(
      "1 个文档没有文字：扫描件 · 显示",
    );
    await expect(window.getByTestId("deleted-heading")).toHaveText("回收站");
    expect(await tallest()).toBeLessThanOrEqual(40);
    await expect(window.getByTestId("tab-title").filter({ hasText: "资料库" })).toHaveCount(1);
  } finally {
    await app.close();
  }
});

test("sections fold and remember it after a restart, by the mouse and by ← and →; Collapse all and Expand all", async () => {
  const first = await launchApp(dataDir);
  try {
    const { window } = first;
    const library = await openFullLibrary(window);
    const board = section(library, "Board");
    await pointAndClick(window, board.getByTestId("library-section-toggle"));
    await expect(board.getByTestId("library-section-head")).toHaveAttribute("data-open", "false");
    await expect(rowsOf(board)).toHaveCount(0);

    // By keyboard: ← folds the focused section, → opens it, ↓ walks on into its rows.
    const finance = section(library, "Finance");
    await finance.getByTestId("library-section-name").focus();
    await window.keyboard.press("ArrowLeft");
    await expect(rowsOf(finance)).toHaveCount(0);
    await window.keyboard.press("ArrowRight");
    await expect(rowsOf(finance)).toHaveCount(8);
    await window.keyboard.press("ArrowDown");
    await expect(rowsOf(finance).first().locator("[data-row-main]")).toBeFocused();
    await window.keyboard.press("ArrowUp");
    await window.keyboard.press("ArrowLeft");
    await expect(rowsOf(finance)).toHaveCount(0);
  } finally {
    await first.app.close();
  }

  const again = await launchApp(dataDir);
  try {
    const { window } = again;
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await expect(rowsOf(section(library, "Contracts"))).toHaveCount(5);
    await expect(rowsOf(section(library, "Board"))).toHaveCount(0);
    await expect(rowsOf(section(library, "Finance"))).toHaveCount(0);

    const foldAll = library.getByTestId("library-fold-all");
    await expect(foldAll).toHaveText("Collapse all");
    await pointAndClick(window, foldAll);
    await expect(rowsOf(library)).toHaveCount(0);
    await expect(foldAll).toHaveText("Expand all");
    await pointAndClick(window, foldAll);
    await expect(rowsOf(section(library, "Board"))).toHaveCount(8);
    await expect(foldAll).toHaveText("Collapse all");
  } finally {
    await again.app.close();
  }
});

test("search narrows every section and hides the empty ones; a filter chip and the problem line's Show narrow it too", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    const library = await openFullLibrary(window);
    const search = library.getByTestId("library-search");

    // / focuses the search from the list.
    await rowsOf(library).first().locator("[data-row-main]").focus();
    await window.keyboard.press("/");
    await expect(search).toBeFocused();
    await window.keyboard.type("sensor");
    await expect(sections(library).getByTestId("library-section-name")).toHaveText([
      "Thesis: urban heat",
    ]);
    await expect(rowsOf(library)).toHaveCount(5);
    await expect(window.getByTestId("recently-deleted")).toHaveCount(0);

    // Every word must match, in names and Tags; Chinese names by their characters.
    await search.fill("contract nda");
    await expect(rowsOf(library)).toHaveCount(2);
    expect(
      (await rowsOf(library).getByTestId("library-document-name").allTextContents()).sort(),
    ).toEqual(["Mutual NDA, Alder Systems", "NDA: Polar Freight"]);
    await search.fill("纪要");
    await expect(rowsOf(library)).toHaveCount(3);

    // Nothing found says so, with one way back.
    await search.fill("zzz nothing");
    await expect(library.getByTestId("library-no-match")).toHaveText(
      "No Documents match · Clear search and filters",
    );
    await pointAndClick(window, library.getByTestId("library-clear-narrowing"));
    await expect(search).toHaveValue("");
    await expect(sections(library)).toHaveCount(FOLDERS.length + 1);

    // + Filter, then Format, then PDF: a chip that reads as a sentence; × drops it.
    await pointAndClick(window, library.getByTestId("library-add-filter"));
    await pointAndClick(
      window,
      window.locator('[data-testid="library-filter-kind"][data-facet="format"]'),
    );
    await pointAndClick(
      window,
      window.locator('[data-testid="library-filter-option"][data-value="pdf"]'),
    );
    await window.keyboard.press("Escape");
    await expect(library.getByTestId("library-filter")).toHaveText("Format: PDF");
    const formats = await rowsOf(library).getByTestId("library-document-format").allTextContents();
    expect(new Set(formats)).toEqual(new Set(["PDF"]));
    await pointAndClick(window, library.getByTestId("library-filter-remove"));
    await expect(library.getByTestId("library-filter")).toHaveCount(0);

    // The problem line's Show filters to the scan, in Not in a Folder.
    await pointAndClick(window, library.getByTestId("library-problem-show"));
    await expect(rowsOf(library).getByTestId("library-document-name")).toHaveText([SCAN_NAME]);
    await expect(rowsOf(library).getByTestId("document-status")).toContainText("No text");
    await expect(library.getByTestId("library-filter")).toContainText("Status: No text");
  } finally {
    await app.close();
  }
});

test("five Documents go into a Folder with Move to…, from the selection bar and by ⇧⌘M, with Undo; no row has a dropdown", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1600, 900);
    const library = await openFullLibrary(window);
    const loose = section(library, "Not in a Folder");
    const contracts = section(library, "Contracts");
    const five = [
      "Call notes: procurement",
      "Team offsite agenda",
      "Launch sync 2026-04-02",
      "Stand-up notes, 29 Sep",
      "Stand-up notes, 22 Sep",
    ];
    await pointAndClick(window, loose.getByTestId("library-show-all"));
    for (const name of five) {
      await pointAndClick(window, row(loose, name).getByTestId("library-document-name"), {
        modifiers: ["ControlOrMeta"],
      });
    }
    const bar = library.getByTestId("library-selection");
    await expect(bar.getByRole("status")).toHaveText("5 selected");
    await expect(bar.getByTestId("library-selection-action")).toHaveText([
      /^Move to…(⇧⌘M|Ctrl\+Shift\+M)$/,
      "Ask about these 5 Documents",
      "Delete 5 Documents",
    ]);
    // The quiet bar gives way to the selection's.
    await expect(library.getByTestId("library-bar")).toHaveCount(0);

    await pointAndClick(window, bar.locator('[data-command="document.moveTo"]'));
    const picker = window.getByTestId("move-to-picker");
    await expect(picker.getByTestId("move-to-search")).toBeFocused();
    await window.keyboard.type("Contracts");
    await window.keyboard.press("Enter");
    await expect(window.getByTestId("undo-line")).toContainText("Moved 5 Documents to Contracts");
    await expect(contracts).toHaveAttribute("data-count", "10");
    await expect(loose).toHaveAttribute("data-count", String(LOOSE - 5));

    await window.keyboard.press("ControlOrMeta+z");
    await expect(contracts).toHaveAttribute("data-count", "5");
    await expect(loose).toHaveAttribute("data-count", String(LOOSE));

    // Esc, with the focus back on the bar, clears the selection; ⇧⌘M on a focused row moves that one.
    await window.keyboard.press("Escape");
    await expect(bar).toHaveCount(0);
    const call = row(loose, "Call notes: procurement");
    await call.locator("[data-row-main]").focus();
    await window.keyboard.press("ControlOrMeta+Shift+M");
    await expect(picker.getByTestId("move-to-search")).toBeFocused();
    await window.keyboard.type("Board");
    await window.keyboard.press("Enter");
    await expect(row(section(library, "Board"), "Call notes: procurement")).toHaveCount(1);

    // And its ⋯ has Move to… in the shared order, after the row's own actions.
    const offsite = row(loose, "Team offsite agenda");
    await pointAt(window, offsite);
    await pointAndClick(window, offsite.getByTestId("document-file-menu"));
    const menu = window.getByTestId("document-file-actions").filter({ visible: true });
    await expect(menu.getByTestId("document.moveTo")).toHaveText(/Move to…/);
    await pointAndClick(window, menu.getByTestId("document.moveTo"));
    await expect(picker.getByTestId("move-to-option").first()).toHaveText("Not in a Folder");
    await window.keyboard.press("Escape");
    await expect(library.locator("select")).toHaveCount(0);
  } finally {
    await app.close();
  }
});

test("Tags are edited on the row: a click in the cell opens the picker over it, T and ⌥⌘T from the keyboard, × takes one off, and each change has its own ⌘Z", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1100, 760);
    const library = await openFullLibrary(window);
    await placeAssistant(window, { folded: true });
    const finance = section(library, "Finance");
    await pointAndClick(window, finance.getByTestId("library-show-all"));
    const invoice = row(finance, "Invoice 4471");
    const cell = invoice.getByTestId("document-tags");
    const chips = cell.getByTestId("tag-chip");
    await expect(chips).toHaveText(["Invoice"]);

    // A click on a chip opens the picker over the cell, its field where the chips were.
    await pointAndClick(window, chips.first());
    const picker = invoice.getByTestId("document-tags-popover");
    const field = picker.getByTestId("tag-picker-input");
    await expect(field).toBeFocused();
    await expect(window.getByTestId("viewer")).toHaveCount(0);
    const [cellBox, pickerBox] = await Promise.all([cell.boundingBox(), picker.boundingBox()]);
    if (!cellBox || !pickerBox) throw new Error("The picker isn't showing.");
    expect(Math.abs(pickerBox.x - cellBox.x)).toBeLessThanOrEqual(8);
    expect(pickerBox.y).toBeLessThanOrEqual(cellBox.y);
    expect(pickerBox.y + pickerBox.height).toBeGreaterThan(cellBox.y + cellBox.height);
    await expect(picker.getByTestId("tag-picker-keys")).toHaveText(
      /^↵ adds or removes · ⌫ removes the last · (⌘Z|Ctrl\+Z) undoes · Esc closes$/,
    );

    // Two Tags added; ⌘Z in the empty field takes back one change at a time, and ⇧⌘Z does it again.
    const tokens = picker.getByTestId("tag-token");
    await window.keyboard.type("Contract");
    await window.keyboard.press("Enter");
    await window.keyboard.type("Report");
    await window.keyboard.press("Enter");
    await expect(tokens).toHaveText(["Contract", "Invoice", "Report"]);
    await window.keyboard.press("ControlOrMeta+z");
    await expect(tokens).toHaveText(["Contract", "Invoice"]);
    await window.keyboard.press("ControlOrMeta+z");
    await expect(tokens).toHaveText(["Invoice"]);
    await window.keyboard.press("ControlOrMeta+Shift+z");
    await expect(tokens).toHaveText(["Contract", "Invoice"]);
    await expect(field).toBeFocused();
    await window.keyboard.press("Escape");
    await expect(picker).toBeHidden();
    await expect(cell.getByTestId("document-tags-menu")).toBeFocused();
    await expect(chips).toHaveText(["Contract", "Invoice"]);

    // Pointed at, a chip offers × to take it off in one click; ⌘Z puts it back.
    const contract = chips.filter({ hasText: "Contract" });
    await pointAt(window, library.getByTestId("library-count"));
    await expect(contract.getByTestId("remove-document-tag")).toBeHidden();
    await pointAt(window, contract);
    await pointAndClick(window, contract.getByTestId("remove-document-tag"));
    await expect(chips).toHaveText(["Invoice"]);
    await expect(window.getByTestId("undo-line")).toContainText("Contract");
    await window.keyboard.press("ControlOrMeta+z");
    await expect(chips).toHaveText(["Contract", "Invoice"]);

    // T on a focused row opens its Tags, and Esc gives the focus back to the row; ⌥⌘T does the same.
    const statement = row(finance, "Bank statement, August");
    const statementPicker = statement.getByTestId("document-tags-popover");
    await statement.locator("[data-row-main]").focus();
    await window.keyboard.press("t");
    await expect(statementPicker.getByTestId("tag-picker-input")).toBeFocused();
    await window.keyboard.press("Escape");
    await expect(statementPicker).toBeHidden();
    await expect(statement.locator("[data-row-main]")).toBeFocused();
    await window.keyboard.press("ControlOrMeta+Alt+t");
    await expect(statementPicker.getByTestId("tag-picker-input")).toBeFocused();
    await window.keyboard.type("Invoice");
    await window.keyboard.press("Enter");
    await window.keyboard.press("Escape");
    await expect(statement.getByTestId("tag-chip")).toHaveText(["Invoice"]);

    // Two selected: T on one of them opens Tags… in the selection's bar, for both.
    const others = [row(finance, "Invoice 4502"), row(finance, "Invoice 4519")];
    for (const each of others) await pointAndClick(window, each.getByTestId("library-select"));
    const bar = library.getByTestId("library-selection");
    await expect(bar.getByTestId("library-selection-tags")).toHaveText(/^Tags…(⌥⌘T|Ctrl\+Alt\+T)$/);
    await others[1]?.locator("[data-row-main]").focus();
    await window.keyboard.press("t");
    const bulk = bar.getByTestId("selection-tags-popover");
    await expect(bulk.getByTestId("tag-picker-input")).toBeFocused();
    await window.keyboard.type("Report");
    await window.keyboard.press("Enter");
    await window.keyboard.press("Escape");
    for (const each of others) {
      await expect(each.getByTestId("tag-chip")).toHaveText(["Invoice", "Report"]);
    }
    await window.keyboard.press("Escape");
    await expect(bar).toHaveCount(0);

    // In Chinese, at the narrowest window, the picker speaks Chinese and stays in the window.
    await setLanguage(window, "zh-CN");
    await setWindowSize(app, window, 900, 660);
    await invoice.scrollIntoViewIfNeeded();
    await pointAndClick(window, chips.first());
    await expect(picker.getByTestId("tag-picker-keys")).toHaveText(
      /^↵ 添加或移除 · ⌫ 移除最后一个 · (⌘Z|Ctrl\+Z) 撤销 · Esc 关闭$/,
    );
    const narrow = await picker.boundingBox();
    const width = await window.evaluate(() => innerWidth);
    if (!narrow) throw new Error("The picker isn't showing.");
    expect(narrow.x).toBeGreaterThanOrEqual(0);
    expect(narrow.x + narrow.width).toBeLessThanOrEqual(width);
    await window.keyboard.press("Escape");
  } finally {
    await app.close();
  }
});

test("the selection's bar holds at 900px and at 1,100px beside the viewer, in English and Chinese: no sideways scroll, Esc in sight, the rest under ⋯", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    const library = await openFullLibrary(window);
    const bar = library.getByTestId("library-selection");
    const scroller = library.locator(":scope > div");
    const holds = async () => {
      await expect(bar).toBeVisible();
      await expect(bar.getByTestId("library-selection-clear")).toBeInViewport();
      const [barBox, pageBox, esc] = await Promise.all([
        bar.boundingBox(),
        scroller.boundingBox(),
        bar.getByTestId("library-selection-clear").boundingBox(),
      ]);
      if (!barBox || !pageBox || !esc) throw new Error("The bar isn't showing.");
      expect(barBox.x + barBox.width).toBeLessThanOrEqual(pageBox.x + pageBox.width);
      expect(esc.x + esc.width).toBeLessThanOrEqual(barBox.x + barBox.width);
      expect(await scroller.evaluate((element) => element.scrollWidth - element.clientWidth)).toBe(
        0,
      );
      // What doesn't fit waits under ⋯, which holds every command for the selection.
      await pointAndClick(window, bar.getByTestId("document-file-menu"));
      const menu = window.getByTestId("document-file-actions").filter({ visible: true });
      await expect(menu.getByTestId("document.ask")).toBeVisible();
      await expect(menu.getByTestId("document.delete")).toBeVisible();
      // Move to… among them, its picker beside ⋯ when the bar's own button has folded away.
      await pointAndClick(window, menu.getByTestId("document.moveTo"));
      const picker = window.getByTestId("move-to-picker");
      await expect(picker.getByTestId("move-to-search")).toBeFocused();
      await expect(picker).toBeInViewport();
      await window.keyboard.press("Escape");
      await expect(picker).toBeHidden();
    };
    const finance = section(library, "Finance");
    const choose = async () => {
      await pointAndClick(window, rowsOf(finance).nth(0).getByTestId("library-document-name"), {
        modifiers: ["ControlOrMeta"],
      });
      await pointAndClick(window, rowsOf(finance).nth(1).getByTestId("library-document-name"), {
        modifiers: ["ControlOrMeta"],
      });
    };
    for (const language of ["en", "zh-CN"] as const) {
      await setLanguage(window, language);
      await setWindowSize(app, window, 900, 660);
      await choose();
      await holds();
      await window.keyboard.press("Escape");
      await expect(bar).toHaveCount(0);
    }

    // At 1,100px with a Document open beside the list.
    await setLanguage(window, "en");
    await placeAssistant(window, { folded: true });
    await setWindowSize(app, window, 1100, 660);
    await pointAndClick(window, rowsOf(finance).nth(2).getByTestId("library-document-name"));
    await expect(window.getByTestId("viewer")).toBeVisible();
    for (const language of ["en", "zh-CN"] as const) {
      await setLanguage(window, language);
      await choose();
      await holds();
      await window.keyboard.press("Escape");
    }
  } finally {
    await app.close();
  }
});

test("a Folder opened from the sidebar brings its section into view, open, with the focus on its name", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await setWindowSize(app, window, 1600, 660);
    const library = await openFullLibrary(window);
    const last = FOLDERS.find((folder) => folder.name === "财务报表");
    if (!last) throw new Error("The fixture has no Chinese Folder.");
    const target = section(library, last.name);
    await pointAndClick(window, target.getByTestId("library-section-toggle"));
    await expect(rowsOf(target)).toHaveCount(0);
    await library.evaluate((element) => element.querySelector(":scope > div")?.scrollTo(0, 0));
    // A Tag filter from the sidebar that none of its Documents has gives way too.
    await pointAndClick(window, window.getByTestId("tag-filter-menu"));
    await pointAndClick(
      window,
      window.getByTestId("tag-filters").getByTestId("tag-filter").filter({ hasText: "Slides" }),
    );
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("tag-filter-active")).toContainText("Slides");
    await expect(target).toHaveCount(0);

    const folder = sidebarFolder(window, last.name);
    await pointAt(window, folder.getByTestId("folder-row"));
    await pointAndClick(window, folder.getByTestId("folder-menu"));
    await pointAndClick(window, folder.getByTestId("folder.open"));
    await expect(target.getByTestId("library-section-name")).toBeFocused();
    await expect(target.getByTestId("library-section-name")).toBeInViewport();
    await expect(rowsOf(target)).toHaveCount(last.documents.length);
    // The page still shows every Folder.
    await expect(sections(library)).toHaveCount(FOLDERS.length + 1);
    await expect(window.getByTestId("tag-filter-active")).toHaveCount(0);
  } finally {
    await app.close();
  }
});

test("as the window narrows the columns give way, Added first, then Format, then Tags, and names never break by letter", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    const library = await openFullLibrary(window);
    await placeAssistant(window, { folded: true });
    const first = rowsOf(library).first();
    const shown = async () =>
      Promise.all(
        ["tag-chip", "library-document-format", "library-document-added"].map(
          async (id) =>
            (await first.getByTestId(id).first().count()) > 0 &&
            (await first.getByTestId(id).first().isVisible()),
        ),
      );
    const seen: boolean[][] = [];
    for (const width of [1600, 1300, 1150, 1050, 950, 900]) {
      await setWindowSize(app, window, width, 660);
      seen.push(await shown());
      // One line each: a long name is cut with "…", never wrapped.
      const heights = await rowsOf(library)
        .getByTestId("library-document-name")
        .evaluateAll((names) => names.map((name) => name.getBoundingClientRect().height));
      expect(Math.max(...heights)).toBeLessThanOrEqual(22);
    }
    expect(seen[0]).toEqual([true, true, true]);
    for (const [tags, format, added] of seen) {
      // A column shows only while every column before it in the order does.
      if (added) expect(format).toBe(true);
      if (format) expect(tags).toBe(true);
    }
    expect(seen.some(([, format, added]) => format && !added)).toBe(true);

    // At 1,100px with the Document open beside it, the rows keep their names.
    await setWindowSize(app, window, 1100, 660);
    await pointAndClick(window, first.getByTestId("library-document-name"));
    await expect(window.getByTestId("viewer")).toBeVisible();
    expect(await shown()).toEqual([false, false, false]);
    await expect(first.getByTestId("library-document-name")).toBeVisible();
    await expect
      .poll(() => library.evaluate((element) => element.scrollWidth <= element.clientWidth))
      .toBe(true);
  } finally {
    await app.close();
  }
});

test("an empty Library offers one Add Documents, its ways in, and a place to drop files", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await expect(library.getByTestId("library-count")).toHaveText("0");
    await expect(library.getByTestId("library-empty")).toContainText("Drop files or a folder here");
    await expect(library.getByTestId("library-bar")).toHaveCount(0);
    await expect(library.getByRole("button", { name: /Add Documents/ })).toHaveCount(1);
    await pointAndClick(window, library.getByTestId("library-add-documents"));
    await expect(
      window.getByTestId("library-add-documents-items").getByRole("menuitem"),
    ).toHaveText([
      "Files…",
      "A folder that stays in sync…",
      "A web page…",
      "Find sources…",
      "Connect an app…",
      "Import references…",
    ]);
    await window.keyboard.press("Escape");
    await setLanguage(window, "zh-CN");
    await expect(library.getByTestId("library-empty")).toContainText("将文件或文件夹拖到这里");
  } finally {
    await app.close();
  }
});
