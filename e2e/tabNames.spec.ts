import { expect, type Page, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * Tabs keep a readable name (#905): at 1,100 px with the assistant open, a tab shrinks to no less
 * than about ten characters of its name, or all of it when it is shorter, and the strip scrolls
 * to the rest. Hovering a tab shows its full name.
 */

const NAMES = {
  en: ["Weekly report to the board", "Résumé of the audit", "Q3", "Harbour tide tables", "Budget"],
  "zh-CN": ["茶叶产地与历史研究笔记", "第三季度销售回顾", "预算", "港口潮汐表整理稿", "Q3 review"],
} as const;

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** How many characters of a tab's name are inside its box: the rest is cut off by the ellipsis. */
const visibleCharacters = (window: Page, title: string) =>
  window.getByRole("tab", { name: title, exact: true }).evaluate((tab) => {
    const label = tab.querySelector('[data-testid="tab-title"]');
    const text = label?.firstChild;
    if (!label || !(text instanceof Text)) return 0;
    const box = label.getBoundingClientRect();
    const range = document.createRange();
    let shown = 0;
    for (const character of text.data) {
      range.setStart(text, shown);
      range.setEnd(text, shown + character.length);
      if (range.getBoundingClientRect().right > box.right + 0.5) break;
      shown += character.length;
    }
    return [...text.data.slice(0, shown)].length;
  });

for (const language of ["en", "zh-CN"] as const) {
  test(`at 1,100 px with five tabs, each shows ten characters of its name, or all of it, and the strip scrolls (${language})`, async () => {
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1100, 760);
    for (const name of NAMES[language]) {
      await newMind(window);
      await window.getByTestId("mind-title").fill(name);
      await window.keyboard.press("Enter");
    }
    const tabs = window.getByRole("tab");
    await expect(tabs).toHaveCount(NAMES[language].length);

    for (const name of NAMES[language]) {
      const tab = window.getByRole("tab", { name, exact: true });
      // Hovering shows the whole name.
      await expect(tab).toHaveAttribute("title", name);
      const wanted = Math.min(10, [...name].length);
      expect(await visibleCharacters(window, name)).toBeGreaterThanOrEqual(wanted);
    }
    // They don't fit side by side, so the strip scrolls to the rest, as it always could.
    const list = window.getByTestId("tab-list");
    expect(await list.evaluate((each) => each.scrollWidth > each.clientWidth)).toBe(true);
    await app.close();
  });
}
