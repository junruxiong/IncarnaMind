import { mkdir, utimes, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
  setWindowSize,
} from "./app";

/*
 * Sort (#243): by Name, Date added, Date modified or Kind, either way, from a
 * Folder's ⋯ and the sidebar's, remembered per Folder on this computer; in the
 * Library by the Name header and the "Sort:" button. Set INCARNAMIND_SCREENSHOTS
 * to a folder to also save screenshots there.
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

async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.waitForFunction(() =>
    document.getAnimations().every((animation) => animation.playState !== "running"),
  );
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const PDF = buildPdf([{ lines: ["Cherries", "Cherry trees bloom in spring."] }]);

/**
 * Three files whose own modified times run against the order they are added in:
 * banana first (and newest on disk), then Apple, then cherry (oldest on disk).
 */
async function writeFiles(): Promise<string[]> {
  const files: [string, string | Uint8Array, number][] = [
    ["banana.txt", "Bananas ripen off the tree.\n", Date.UTC(2026, 5, 1)],
    ["Apple.md", "# Apples\n\nApples keep through winter.\n", Date.UTC(2025, 5, 1)],
    ["cherry.pdf", PDF, Date.UTC(2024, 5, 1)],
  ];
  const paths: string[] = [];
  for (const [name, contents, time] of files) {
    const path = join(sources, name);
    await writeFile(path, contents);
    await utimes(path, new Date(time), new Date(time));
    paths.push(path);
  }
  return paths;
}

/** Adds the files one after another, so each is added after the one before, and waits until all are read. */
async function addInOrder(window: Page, paths: string[]): Promise<void> {
  const items = window.getByTestId("document-list-item");
  for (const [index, path] of paths.entries()) {
    await window.getByTestId("add-documents-input").setInputFiles(path);
    await expect(items).toHaveCount(index + 1);
    await expect
      .poll(() =>
        window.evaluate(
          async () =>
            (
              await (
                globalThis as unknown as { incarnamind: CoreBridge }
              ).incarnamind.listDocuments()
            ).filter((each) => each.status === "ready").length,
        ),
      )
      .toBe(index + 1);
  }
}

const bridgeOf = (window: Page) =>
  window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await bridge.listDocuments()).map(({ id, name }) => ({ id, name }));
  });

/** The names of the Documents listed under a Folder (or in the sidebar), top to bottom. */
const sidebarNames = (scope: Locator) =>
  scope.getByTestId("document-list-item").getByTestId("row-text").allTextContents();

async function slideTo(window: Page, target: Locator): Promise<void> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  await window.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 8 });
}

async function slideClick(window: Page, target: Locator): Promise<void> {
  await slideTo(window, target);
  await target.click();
}

/** Opens a Folder's ⋯, then Sort by…, and picks `choice` in the menu that opens. */
async function sortFolder(window: Page, folder: Locator, choices: string[]): Promise<void> {
  for (const choice of choices) {
    await slideTo(window, folder.getByTestId("folder-row"));
    await folder.getByTestId("folder-menu").click();
    await slideClick(window, folder.getByTestId("folder.sort"));
    const menu = window.getByTestId("sort-menu");
    await expect(menu).toBeVisible();
    await slideClick(window, menu.getByTestId(choice));
    await expect(menu).toBeHidden();
  }
}

test("a Folder sorts by Name, Date added, Date modified and Kind, either way, and remembers it after a restart", async () => {
  const paths = await writeFiles();
  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  await setWindowSize(first.app, first.window, 1400, 800);
  await addInOrder(first.window, paths);
  await first.window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const group = await core.createLibraryGroup({ name: "Reading", description: "" });
    for (const document of await core.listDocuments()) {
      await core.assignDocumentGroup(document.id, group.id);
    }
  });
  const folder = first.window.getByTestId("library-folder").filter({ hasText: "Reading" });
  await folder.getByTestId("folder-row").click();
  // Before any choice, the newest added first.
  await expect.poll(() => sidebarNames(folder)).toEqual(["cherry", "Apple", "banana"]);

  await sortFolder(first.window, folder, ["sort-name", "sort-asc"]);
  await expect.poll(() => sidebarNames(folder)).toEqual(["Apple", "banana", "cherry"]);
  await screenshot(first.window, "sort-name-ascending");
  await sortFolder(first.window, folder, ["sort-desc"]);
  await expect.poll(() => sidebarNames(folder)).toEqual(["cherry", "banana", "Apple"]);
  await sortFolder(first.window, folder, ["sort-added", "sort-asc"]);
  await expect.poll(() => sidebarNames(folder)).toEqual(["banana", "Apple", "cherry"]);
  // The file's own time, not when it was added: cherry's file is the oldest.
  await sortFolder(first.window, folder, ["sort-modified"]);
  await expect.poll(() => sidebarNames(folder)).toEqual(["cherry", "Apple", "banana"]);
  await sortFolder(first.window, folder, ["sort-desc"]);
  await expect.poll(() => sidebarNames(folder)).toEqual(["banana", "Apple", "cherry"]);
  // Markdown, PDF, Plain text.
  await sortFolder(first.window, folder, ["sort-kind", "sort-asc"]);
  await expect.poll(() => sidebarNames(folder)).toEqual(["Apple", "cherry", "banana"]);

  // The choice is the Folder's own, and the menu shows it.
  await folder.getByTestId("folder-row").hover();
  await folder.getByTestId("folder-menu").click();
  await first.window.keyboard.press("Escape");
  await first.app.close();

  const second = await launchApp(dataDir);
  await setWindowSize(second.app, second.window, 1400, 800);
  const reopened = second.window.getByTestId("library-folder").filter({ hasText: "Reading" });
  // Open, as it was left.
  await expect(reopened).toHaveAttribute("data-expanded", "true");
  await expect.poll(() => sidebarNames(reopened)).toEqual(["Apple", "cherry", "banana"]);
  await sortFolder(second.window, reopened, ["sort-name"]);
  await slideTo(second.window, reopened.getByTestId("folder-row"));
  await reopened.getByTestId("folder-menu").click();
  await slideClick(second.window, reopened.getByTestId("folder.sort"));
  const menu = second.window.getByTestId("sort-menu");
  await expect(menu.getByTestId("sort-name")).toHaveAttribute("aria-checked", "true");
  await expect(menu.getByTestId("sort-asc")).toHaveAttribute("aria-checked", "true");
  await expect(menu.getByTestId("sort-folders-first")).toHaveAttribute("aria-checked", "true");
  await screenshot(second.window, "sort-menu");
  await second.window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await second.app.close();
});

test("the sidebar's ⋯ sorts the Folders, and Use for all Folders sets one choice for every Folder", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1400, 800);
  const names = ["Mango", "apple", "Zebra"];
  await window.evaluate(async (folders) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    for (const name of folders) await core.createLibraryGroup({ name, description: "" });
  }, names);
  const folders = window.getByTestId("library-folder");
  const order = () => folders.getByTestId("row-text").allTextContents();
  await expect(folders).toHaveCount(3);

  await slideTo(window, window.getByTestId("open-library"));
  await window.getByTestId("sidebar-sort").click();
  const menu = window.getByTestId("sort-menu");
  await slideClick(window, menu.getByTestId("sort-name"));
  await expect(menu).toBeHidden();
  await slideTo(window, window.getByTestId("open-library"));
  await window.getByTestId("sidebar-sort").click();
  await slideClick(window, menu.getByTestId("sort-asc"));
  // Names ignore case.
  await expect.poll(order).toEqual(["apple", "Mango", "Zebra"]);
  await slideTo(window, window.getByTestId("open-library"));
  await window.getByTestId("sidebar-sort").click();
  await slideClick(window, menu.getByTestId("sort-desc"));
  await expect.poll(order).toEqual(["Zebra", "Mango", "apple"]);

  // One Folder chosen apart, then one choice for all: the Folder's own is forgotten.
  const mango = folders.filter({ hasText: "Mango" });
  await sortFolder(window, mango, ["sort-kind"]);
  await slideTo(window, mango.getByTestId("folder-row"));
  await mango.getByTestId("folder-menu").click();
  await slideClick(window, mango.getByTestId("folder.sort"));
  await slideClick(window, menu.getByTestId("sort-for-all"));
  await expect(menu).toBeHidden();
  const stored = await window.evaluate(() => localStorage.getItem("incarnamind.sort"));
  expect(JSON.parse(stored ?? "{}")).toMatchObject({
    all: { key: "kind", direction: "desc", foldersFirst: true },
  });
  await app.close();
});

/** Opens the Library's Sort, and picks `choice` in the menu that opens. */
async function sortLibrary(window: Page, library: Locator, choice: string): Promise<void> {
  await slideClick(window, library.getByTestId("library-sort"));
  const menu = window.getByTestId("sort-menu");
  await expect(menu).toBeVisible();
  await expect(menu.getByTestId("sort-folders-first")).toHaveCount(0);
  await slideClick(window, menu.getByTestId(choice));
  await expect(menu).toBeHidden();
}

test("the Library's Sort orders every section by Name, Date modified or Date added, either way, and remembers it after a restart", async () => {
  const paths = await writeFiles();
  let { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1400, 800);
  await addInOrder(window, paths);
  await window.getByTestId("open-library").click();
  let library = window.getByTestId("library");
  const names = () => library.getByTestId("library-document-name").allTextContents();
  const bar = () => library.getByTestId("library-sort");
  await expect(bar()).toHaveText("Sort: Date added ↓");
  await expect.poll(names).toEqual(["cherry", "Apple", "banana"]);

  await sortLibrary(window, library, "sort-name");
  await expect(bar()).toHaveText("Sort: Name ↓");
  await expect.poll(names).toEqual(["cherry", "banana", "Apple"]);
  await sortLibrary(window, library, "sort-asc");
  await expect(bar()).toHaveText("Sort: Name ↑");
  await expect.poll(names).toEqual(["Apple", "banana", "cherry"]);
  await screenshot(window, "sort-library-name");

  // banana's file is the newest, cherry's the oldest.
  await sortLibrary(window, library, "sort-modified");
  await sortLibrary(window, library, "sort-desc");
  await expect(bar()).toHaveText("Sort: Date modified ↓");
  await expect.poll(names).toEqual(["banana", "Apple", "cherry"]);

  // Each section keeps the one order: a Document filed in a Folder sorts among the Folder's.
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const group = await core.createLibraryGroup({ name: "Reading", description: "" });
    const documents = await core.listDocuments();
    await core.moveToFolder({
      documentIds: documents.filter((each) => each.name !== "Apple").map((each) => each.id),
      folderId: group.id,
    });
  });
  await expect.poll(names).toEqual(["banana", "cherry", "Apple"]);

  await app.close();
  ({ app, window } = await launchApp(dataDir));
  await window.getByTestId("open-library").click();
  library = window.getByTestId("library");
  await expect(bar()).toHaveText("Sort: Date modified ↓");
  await expect.poll(names).toEqual(["banana", "cherry", "Apple"]);
  await app.close();
});

test("Chinese names sort by pinyin in the Library and the sidebar", async () => {
  const chinese = ["张三", "李四", "安娜", "陈六"];
  for (const name of chinese) await writeFile(join(sources, `${name}.txt`), `${name}的笔记。\n`);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1400, 800);
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.updateSettings({ user: { language: "zh-CN" } });
  });
  const items = window.getByTestId("document-list-item");
  await window
    .getByTestId("add-documents-input")
    .setInputFiles(chinese.map((name) => join(sources, `${name}.txt`)));
  await expect(items).toHaveCount(4);
  await expect.poll(async () => (await bridgeOf(window)).length).toBe(4);
  await window.getByTestId("open-library").click();
  const library = window.getByTestId("library");
  await sortLibrary(window, library, "sort-name");
  await sortLibrary(window, library, "sort-asc");
  // an, chen, li, zhang.
  await expect
    .poll(() => library.getByTestId("library-document-name").allTextContents())
    .toEqual(["安娜", "陈六", "李四", "张三"]);
  await expect(library.getByTestId("library-sort")).toHaveText("排序：名称 ↑");
  await screenshot(window, "sort-library-zh");
  await app.close();
});
