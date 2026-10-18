import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import {
  addDocuments,
  coreOf,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * The undo line after Move, Delete and Rename of a Document with a 70-character name (#1122): the
 * name is cut in the middle and the line's own words and Undo stay whole, with the full name in the
 * tooltip and in Undo's accessible name. Set INCARNAMIND_SCREENSHOTS to a folder to also save
 * screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(target: Locator, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await motionSettled(target.page());
  await mkdir(SCREENSHOTS, { recursive: true });
  await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

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

const WORDS = {
  en: {
    name: (tag: string) =>
      `Quarterly review of harbour dues and the wider coastal levy schedule ${tag}`,
    moved: { start: "Moved “", end: "” to Finance" },
    renamed: { start: "Renamed to “", end: "”" },
    deleted: { start: "Moved “", end: "” to Recently deleted" },
  },
  "zh-CN": {
    name: (tag: string) => `${"信贷委员会合同审阅笔记".repeat(7).slice(0, 69)}${tag}`,
    moved: { start: "已将“", end: "”移动到“Finance”" },
    renamed: { start: "已重命名为“", end: "”" },
    deleted: { start: "已将“", end: "”移到回收站" },
  },
} as const;

/** The words of `line` that lie outside the sidebar. */
function outsideSidebar(line: Locator): Promise<string[]> {
  return line.evaluate((element) => {
    const edge = document.querySelector('[data-testid="sidebar"]')?.getBoundingClientRect();
    if (!edge) throw new Error("The sidebar isn't showing.");
    const outside: string[] = [];
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const range = document.createRange();
      range.selectNodeContents(node);
      for (const rect of range.getClientRects()) {
        const inside =
          rect.left >= edge.left - 0.5 &&
          rect.right <= edge.right + 0.5 &&
          rect.top >= edge.top - 0.5 &&
          rect.bottom <= edge.bottom + 0.5;
        if (!inside) outside.push(node.textContent ?? "");
      }
    }
    return outside;
  });
}

/** Where the first word of `target` starts, whichever line it is on. */
function textLeft(target: Locator): Promise<number> {
  return target.evaluate((element) => {
    const range = document.createRange();
    range.selectNodeContents(element);
    const first = range.getClientRects()[0];
    if (!first) throw new Error("The element has no text.");
    return first.left;
  });
}

async function boxOf(target: Locator) {
  const box = await target.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  return box;
}

/** The line says its verb and its end whole, the name cut in the middle, and Undo is whole beside or under it. */
async function expectWholeLine(
  window: Page,
  name: string,
  words: { start: string; end: string },
): Promise<void> {
  const line = window.getByTestId("undo-line");
  const message = line.getByTestId("undo-message");
  const undo = line.getByTestId("undo-action");
  const sidebar = window.getByTestId("sidebar");
  await expect(message).toHaveText(new RegExp(`^${words.start}`));
  const said = (await message.textContent()) ?? "";
  const characters = [...name];

  expect(said.startsWith(words.start)).toBe(true);
  expect(said.endsWith(words.end)).toBe(true);
  const shown = said.slice(words.start.length, said.length - words.end.length);
  expect(shown.includes("…")).toBe(true);
  expect(shown.startsWith(characters.slice(0, 10).join(""))).toBe(true);
  expect(shown.endsWith(characters.slice(-10).join(""))).toBe(true);
  expect(said).not.toContain(name);

  // The tooltip and Undo's accessible name have the whole name.
  expect(await message.getAttribute("title")).toContain(name);
  expect(await undo.getAttribute("aria-label")).toContain(name);

  // Nothing is cut off or outside the sidebar, and Undo sits after the message or under it.
  expect(await outsideSidebar(message)).toEqual([]);
  expect(await outsideSidebar(undo)).toEqual([]);
  const [messageBox, undoBox, sidebarBox] = await Promise.all([
    boxOf(message),
    boxOf(undo),
    boxOf(sidebar),
  ]);
  expect(await message.evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(
    true,
  );
  expect(undoBox.x).toBeGreaterThanOrEqual(sidebarBox.x);
  expect(undoBox.x + undoBox.width).toBeLessThanOrEqual(sidebarBox.x + sidebarBox.width);
  const after = messageBox.x + messageBox.width <= undoBox.x + 0.5;
  const under = messageBox.y + messageBox.height <= undoBox.y + 0.5;
  expect(after || under).toBe(true);
  // Alone on its row, Undo starts where the message does.
  if (under)
    expect(Math.abs((await textLeft(undo)) - (await textLeft(message)))).toBeLessThanOrEqual(1);
}

for (const language of ["en", "zh-CN"] as const) {
  test(`${language}: after Move, Rename and Delete of a Document with a 70-character name, the undo line keeps its words and Undo whole and cuts the name in the middle, at 900 and 1,440 px`, async () => {
    test.setTimeout(90_000);
    const { app, window } = await launchApp(dataDir);
    try {
      await dismissChatSetup(window);
      await setWindowSize(app, window, 1440, 760);
      await setLanguage(window, language);
      await (await coreOf(window)).evaluate((core) =>
        core.createLibraryGroup({ name: "Finance", description: "" }),
      );
      const words = WORDS[language];

      for (const [index, width] of [900, 1440].entries()) {
        await setWindowSize(app, window, width, 760);
        const first = words.name(index === 0 ? "A" : "C");
        const second = words.name(index === 0 ? "B" : "D");
        expect([...first]).toHaveLength(70);
        await writeFile(join(sources, `${first}.txt`), `${first} notes.\n`);
        await addDocuments(window, [join(sources, `${first}.txt`)]);
        await window.getByTestId("open-library").click();
        const row = window.getByTestId("library-document");
        const actions = window.getByTestId("document-file-actions").filter({ visible: true });
        const label = `${language}-${width}`;
        const choose = async (command: string) => {
          await row.hover();
          await pointAndClick(window, row.getByTestId("document-file-menu"));
          await pointAndClick(window, actions.getByTestId(command));
        };

        await choose("document.moveTo");
        const picker = window.getByTestId("move-to-picker");
        await expect(picker.getByTestId("move-to-search")).toBeFocused();
        await pointAndClick(
          window,
          picker.getByTestId("move-to-option").filter({ hasText: "Finance" }),
        );
        await expect(picker).toBeHidden();
        await expect(window.getByTestId("undo-message")).toContainText(words.moved.start);
        await screenshot(window.getByTestId("sidebar"), `${label}-move`);
        await expectWholeLine(window, first, words.moved);

        await choose("document.rename");
        const field = row.getByRole("textbox");
        await expect(field).toBeFocused();
        await field.fill(second);
        await window.keyboard.press("Enter");
        await expect(window.getByTestId("undo-message")).toContainText(words.renamed.start);
        await screenshot(window.getByTestId("sidebar"), `${label}-rename`);
        await expectWholeLine(window, second, words.renamed);

        await choose("document.delete");
        await expect(row).toHaveCount(0);
        await expect(window.getByTestId("undo-message")).toContainText(words.deleted.end);
        await screenshot(window.getByTestId("sidebar"), `${label}-delete`);
        await expectWholeLine(window, second, words.deleted);
      }
    } finally {
      await app.close();
    }
  });
}
