import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  enterContent,
  launchApp,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * Lists in a Mind (#1341): Tab and Shift-Tab change an item's level (bullets, numbers and
 * checklists), each one ⌘Z; and a list is indented as Word indents it (DESIGN.md): the text of
 * level 1 starts 48px from the text edge, each further level 48px more, wrapped lines aligned to it.
 * English at 1,440 px, Chinese at 900 px.
 */
const MAC = process.platform === "darwin";
const UNDO = MAC ? "Meta+z" : "Control+z";
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

const SETTINGS = [
  {
    language: "en",
    width: 1440,
    body: "Tides rise twice a day.",
    items: ["Spring tides", "Neap tides", "Tidal range"],
    long: "A long item whose words wrap onto a second line so that the lines can be seen to start at the same place under the first word of it, whatever the width of the window is today.",
  },
  {
    language: "zh-CN",
    width: 900,
    body: "潮汐每天涨落两次。",
    items: ["大潮", "小潮", "潮差"],
    long: "这是一个很长的条目，它的文字会折到第二行，这样就能看出各行是否都从同一处开始，而不论今天窗口有多宽，也不论字有多少。这是一个很长的条目，它的文字会折到第二行。",
  },
] as const;

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** The first Block, the body's text edge, and each list item's text, from the page's left. */
const edge = (editor: Locator) =>
  editor
    .locator(":scope > p")
    .first()
    .evaluate((p) => p.getBoundingClientRect().left);

const textLeft = (item: Locator) =>
  item.evaluate((li) => {
    const text = li.querySelector(":scope > p, :scope > div > p");
    return text ? text.getBoundingClientRect().left : Number.NaN;
  });

const near = (value: number, expected: number) =>
  expect(
    Math.abs(value - expected),
    `${value} is not within 1px of ${expected}`,
  ).toBeLessThanOrEqual(1);

/** The bullet or checkbox of an item, from the text edge: where it is drawn, not where the text is. */
const markerLeft = (item: Locator) =>
  item.evaluate((li) => {
    const box = li.querySelector(":scope > label");
    if (box) return box.getBoundingClientRect().left;
    const before = getComputedStyle(li, "::before");
    const text = li.querySelector(":scope > p");
    if (!text || before.content === "none") return Number.NaN;
    return text.getBoundingClientRect().left + Number.parseFloat(before.left);
  });

async function openMind(language: "en" | "zh-CN", width: number, body: string) {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setLanguage(window, language);
  await setWindowSize(app, window, width, 760);
  await newMind(window);
  await enterContent(window);
  const editor = window.getByTestId("mind-editor");
  await window.keyboard.type(body);
  await window.keyboard.press("Enter");
  return { app, window, editor };
}

async function screenshot(window: Page, name: string): Promise<void> {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

for (const { language, width, body, items, long } of SETTINGS) {
  test(`Tab and Shift-Tab change a bullet's level, with the caret in place and one ⌘Z each, and a list is indented as in Word (${language})`, async () => {
    const { app, window, editor } = await openMind(language, width, body);

    await window.keyboard.type("- ");
    await window.keyboard.type(items[0]);
    const edgeLeft = await edge(editor);
    // Tab at the first item has nothing to go under: nothing changes, and the focus stays in the page.
    await window.keyboard.press("Tab");
    await expect(editor).toBeFocused();
    await expect(editor.locator("ul")).toHaveCount(1);
    await expect(editor.locator("ul ul")).toHaveCount(0);
    // Level 1: the text 48px in from the edge, the bullet 24px in (DESIGN.md).
    const first = editor.locator("li").first();
    near((await textLeft(first)) - edgeLeft, 48);
    near((await markerLeft(first)) - edgeLeft, 24);
    await screenshot(window, `list-level-1-${language}-${width}`);

    // Tab makes the next item a level 2 item under the first: one step, the caret where it was.
    await window.keyboard.press("Enter");
    await window.keyboard.type(items[1]);
    await window.keyboard.press("Tab");
    const second = editor.locator("ul ul > li");
    await expect(second).toHaveText(items[1]);
    await expect(editor).toBeFocused();
    near((await textLeft(second)) - edgeLeft, 96);
    near((await markerLeft(second)) - edgeLeft, 72);
    await window.keyboard.type("!");
    await expect(second).toHaveText(`${items[1]}!`);
    await window.keyboard.press("Backspace");
    await screenshot(window, `list-level-2-${language}-${width}`);

    // A long item at level 3: its lines start at the same place.
    await window.keyboard.press("Enter");
    await window.keyboard.type(long);
    await window.keyboard.press("Tab");
    const third = editor.locator("ul ul ul > li");
    await expect(third).toHaveCount(1);
    near((await textLeft(third)) - edgeLeft, 144);
    near((await markerLeft(third)) - edgeLeft, 120);
    const linesOf = (item: Locator) =>
      item.locator("> p").evaluate((p) => {
        const range = document.createRange();
        range.selectNodeContents(p);
        const rects = [...range.getClientRects()].filter((rect) => rect.width > 0);
        const starts = new Map<number, number>();
        for (const rect of rects) {
          const top = Math.round(rect.top);
          starts.set(top, Math.min(starts.get(top) ?? Number.POSITIVE_INFINITY, rect.left));
        }
        return [...starts.values()];
      });
    const wrapped = await linesOf(third);
    expect(wrapped.length, "the long item wraps onto a second line").toBeGreaterThan(1);
    for (const start of wrapped) near(start, wrapped[0] ?? 0);

    // Shift-Tab moves it out a level at a time, and ⌘Z puts each step back.
    await window.keyboard.press("Shift+Tab");
    await expect(third).toHaveCount(0);
    await expect(editor.locator("ul ul > li")).toHaveCount(2);
    await window.keyboard.press(UNDO);
    await expect(third).toHaveCount(1);
    await window.keyboard.press(UNDO);
    await expect(third).toHaveCount(0);
    await expect(editor.locator("ul ul > li")).toHaveCount(2);
    await window.keyboard.press("Shift+Tab");
    const level1 = editor.locator(":scope > ul > li").nth(1);
    await expect(editor.locator("ul ul > li")).toHaveCount(1);
    near((await textLeft(level1)) - edgeLeft, 48);
    const again = await linesOf(level1);
    for (const start of again) near(start, again[0] ?? 0);

    // Backspace at the start of an empty item ends the list.
    await window.keyboard.press("Enter");
    await expect(editor.locator("li")).toHaveCount(4);
    await window.keyboard.press("Backspace");
    await expect(editor.locator("li")).toHaveCount(3);
    await app.close();
  });

  test(`Tab and Shift-Tab change the level of a numbered item, one ⌘Z each, indented the same (${language})`, async () => {
    const { app, window, editor } = await openMind(language, width, body);
    const edgeLeft = await edge(editor);

    await window.keyboard.type("1. ");
    await window.keyboard.type(items[0]);
    await window.keyboard.press("Enter");
    await window.keyboard.type(items[1]);
    await window.keyboard.press("Tab");
    await expect(editor.locator("ol ol > li")).toHaveText(items[1]);
    near((await textLeft(editor.locator("ol > li").first())) - edgeLeft, 48);
    near((await textLeft(editor.locator("ol ol > li"))) - edgeLeft, 96);
    await screenshot(window, `numbers-${language}-${width}`);
    await window.keyboard.press(UNDO);
    await expect(editor.locator("ol ol")).toHaveCount(0);
    await expect(editor.locator("ol > li")).toHaveCount(2);
    await app.close();
  });

  test(`Tab and Shift-Tab change the level of a checklist item, one ⌘Z, indented the same (${language})`, async () => {
    const { app, window, editor } = await openMind(language, width, body);
    const edgeLeft = await edge(editor);

    await window.keyboard.type("[ ] ");
    await window.keyboard.type(items[0]);
    await window.keyboard.press("Enter");
    await window.keyboard.type(items[1]);
    const tasks = editor.locator("li[data-checked]");
    await expect(tasks).toHaveCount(2);
    near((await textLeft(tasks.first())) - edgeLeft, 48);
    near((await markerLeft(tasks.first())) - edgeLeft, 24);
    await window.keyboard.press("Tab");
    const nested = editor.locator("ul ul li[data-checked]");
    await expect(nested).toHaveCount(1);
    await expect(nested.locator("> div")).toHaveText(items[1]);
    near((await textLeft(nested)) - edgeLeft, 96);
    near((await markerLeft(nested)) - edgeLeft, 72);
    await screenshot(window, `checklist-${language}-${width}`);
    await window.keyboard.press(UNDO);
    await expect(editor.locator("ul ul")).toHaveCount(0);
    await app.close();
  });
}
