import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  documentIdOf,
  launchApp,
  nameMind,
  openDocumentAt,
  placeAssistant,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * A group's tab strip rests on whole tabs (DESIGN.md, Components: Mind tabs, Width): no tab is
 * clipped at its edge, the shown tab is whole, and nothing covers the "+", at the narrow sizes
 * where the tabs don't fit. A strip being scrolled may show a part; these checks wait for it to rest.
 */

const LANGUAGES = ["en", "zh-CN"] as const;
type Language = (typeof LANGUAGES)[number];

const NAMES: Record<Language, string[]> = {
  en: [
    "Lit review: urban heat islands",
    "Interview notes, July",
    "Budget 2027",
    "Field work plan",
    "Reading list for the spring",
    "Meeting with the council",
    "Draft of the final report",
    "Survey results by district",
  ],
  "zh-CN": [
    "城市热岛文献综述",
    "七月访谈记录",
    "二〇二七年预算",
    "田野调查计划",
    "春季阅读清单",
    "与市议会的会议",
    "最终报告初稿",
    "各区调查结果",
  ],
};
const REPORT: Record<Language, string> = { en: "Urban heat", "zh-CN": "城市热岛报告" };

/** The nth Mind's name, different from every other. */
const nameOf = (language: Language, index: number) =>
  `${NAMES[language][index % NAMES[language].length]} ${index + 1}`;

const SIZES = [
  { width: 1100, height: 760 },
  { width: 900, height: 700 },
  { width: 1280, height: 760 },
  { width: 1440, height: 900 },
];

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

/**
 * What is wrong with the strips as they are drawn now: empty when each rests on whole tabs. A
 * strip the User scrolled away may rest with its shown tab out of sight.
 */
function problemsOf(window: Page, shownInView: boolean): Promise<string[]> {
  return window.evaluate((requireShown) => {
    const problems: string[] = [];
    const slack = 1;
    const strips = Array.from(document.querySelectorAll<HTMLElement>('[data-testid="tab-strip"]'));
    strips.forEach((strip, at) => {
      const list = strip.querySelector<HTMLElement>('[data-testid="tab-list"]');
      const plus = strip.querySelector<HTMLElement>('[data-testid="new-tab"]');
      if (!list || !plus) return;
      const where = `Strip ${at + 1}`;
      const view = list.getBoundingClientRect();
      const add = plus.getBoundingClientRect();
      let before = 0;
      let after = 0;
      let shown = 0;
      const tabs = Array.from(list.querySelectorAll<HTMLElement>('[role="tab"]'));
      for (const tab of tabs) {
        const box = tab.getBoundingClientRect();
        const name = tab.getAttribute("aria-label") ?? "a tab";
        const inside = box.left >= view.left - slack && box.right <= view.right + slack;
        const hiddenLeft = box.right <= view.left + slack;
        const hiddenRight = box.left >= view.right - slack;
        if (hiddenLeft) before++;
        if (hiddenRight) after++;
        if (!inside && !hiddenLeft && !hiddenRight) problems.push(`${where}: "${name}" is clipped`);
        if (tab.getAttribute("aria-selected") === "true") {
          shown++;
          if (requireShown && !inside)
            problems.push(`${where}: the shown tab "${name}" isn't whole`);
        }
        if (!inside) continue;
        if (box.right > add.left + slack) problems.push(`${where}: the + overlaps "${name}"`);
        const centre = document.elementFromPoint(
          box.left + box.width / 2,
          box.top + box.height / 2,
        );
        if (centre?.closest('[role="tab"]') !== tab)
          problems.push(`${where}: "${name}" is covered`);
        const title = tab.querySelector<HTMLElement>('[data-testid="tab-title"]');
        if (title) {
          const text = title.getBoundingClientRect();
          const whole = title.scrollWidth <= title.clientWidth;
          const ellipsis = getComputedStyle(title).textOverflow === "ellipsis";
          if (!whole && !ellipsis) problems.push(`${where}: "${name}" is cut mid-word`);
          if (text.left < box.left - slack || text.right > box.right + slack) {
            problems.push(`${where}: "${name}"'s name leaves its tab`);
          }
        }
      }
      if (tabs.length > 0 && tabs.length === before + after)
        problems.push(`${where}: shows no tab`);
      if (shown > 1) problems.push(`${where}: ${shown} tabs are shown`);
      if (before > 0 && !list.hasAttribute("data-hidden-before")) {
        problems.push(`${where}: tabs hide on the left with no fade`);
      }
      if (after > 0 && !list.hasAttribute("data-hidden-after")) {
        problems.push(`${where}: tabs hide on the right with no fade`);
      }
      const onPlus = document.elementFromPoint(add.left + add.width / 2, add.top + add.height / 2);
      if (!onPlus?.closest('[data-testid="new-tab"]')) problems.push(`${where}: the + is covered`);
    });
    if (document.documentElement.scrollWidth > globalThis.innerWidth) {
      problems.push("The page scrolls sideways");
    }
    return problems;
  }, shownInView);
}

const expectResting = (window: Page, message: string, { shownInView = true } = {}) =>
  expect.poll(() => problemsOf(window, shownInView), { message }).toEqual([]);

const groupAt = (window: Page, index: number) => window.getByTestId("tab-group").nth(index);
const listAt = (window: Page, index: number) => window.getByTestId("tab-list").nth(index);
const scrollOf = (list: Locator) => list.evaluate((element) => element.scrollLeft);
const titlesOf = (list: Locator) => list.getByRole("tab").getByTestId("tab-title");
const shownTab = (window: Page) => window.locator('[role="tab"][aria-selected="true"]');

/** A new Mind from a group's "+", named. */
async function addMind(window: Page, group: Locator, name: string): Promise<void> {
  const count = await window.getByTestId("mind-tab").count();
  await group.getByTestId("new-tab").click();
  await expect(window.getByTestId("mind-tab")).toHaveCount(count + 1);
  await nameMind(window, name);
}

/** A Report, added as a Document; its ID. */
async function addReport(window: Page, language: Language): Promise<string> {
  const path = join(sources, `${REPORT[language]}.pdf`);
  await writeFile(
    path,
    buildPdf([{ lines: ["Urban heat", "Eastgate peaked at 47.2 degrees on 19 July."] }]),
  );
  await addDocuments(window, [path]);
  return documentIdOf(window, REPORT[language]);
}

const OPENS_PDF = { quote: "Eastgate peaked at 47.2 degrees on 19 July." };

for (const language of LANGUAGES) {
  test(`the strip rests on whole tabs with 3 to 8 tabs and a PDF beside the Minds, at every window size (${language})`, async () => {
    test.setTimeout(240_000);
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await setWindowSize(app, window, 1600, 900);
    await setLanguage(window, language);
    const report = await addReport(window, language);
    await addMind(window, groupAt(window, 0), nameOf(language, 0));
    await addMind(window, groupAt(window, 0), nameOf(language, 1));
    await openDocumentAt(window, { documentId: report, ...OPENS_PDF });
    await expect(window.getByTestId("tab-group")).toHaveCount(2);

    let total = 3;
    for (const target of [3, 4, 5, 8]) {
      while (total < target) {
        await addMind(window, groupAt(window, 0), nameOf(language, total - 1));
        total++;
      }
      for (const size of SIZES) {
        await setWindowSize(app, window, size.width, size.height);
        await expectResting(window, `${target} tabs at ${size.width} × ${size.height}`);
      }
    }
    await app.close();
  });

  test(`two groups side by side each rest on whole tabs with 3 to 8 tabs (${language})`, async () => {
    test.setTimeout(300_000);
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await setWindowSize(app, window, 1100, 760);
    await setLanguage(window, language);
    await placeAssistant(window, { folded: true });
    const report = await addReport(window, language);
    await addMind(window, groupAt(window, 0), nameOf(language, 0));
    await openDocumentAt(window, { documentId: report, ...OPENS_PDF });
    await expect(window.getByTestId("tab-group")).toHaveCount(2);

    let minds = 1;
    let beside = 0;
    for (const target of [3, 5, 8]) {
      while (minds < target) {
        await addMind(window, groupAt(window, 0), nameOf(language, minds));
        minds++;
      }
      while (beside < target - 1) {
        await addMind(window, groupAt(window, 1), nameOf(language, 20 + beside));
        beside++;
      }
      await expectResting(window, `${target} tabs in each group`);
    }
    await app.close();
  });

  test(`a tab that becomes active comes whole into view, and stays whole as it changes size or the strip is scrolled (${language})`, async () => {
    test.setTimeout(240_000);
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await setWindowSize(app, window, 1100, 760);
    await setLanguage(window, language);
    const report = await addReport(window, language);
    for (let index = 0; index < 4; index++) {
      await addMind(window, groupAt(window, 0), nameOf(language, index));
    }
    await expectResting(window, "four Minds");
    const list = listAt(window, 0);
    const titles = titlesOf(list);
    const mindTitles = await titles.allTextContents();
    expect(mindTitles).toHaveLength(4);

    // The newest tab is shown. ⌃Tab and ⌃⇧Tab walk to tabs that are hidden, each coming whole into view.
    for (let step = 1; step <= 4; step++) {
      await window.keyboard.press("Control+Tab");
      await expect(shownTab(window).getByTestId("tab-title")).toHaveText(
        mindTitles[(3 + step) % 4] ?? "",
      );
      await expectResting(window, `⌃Tab, step ${step}`);
    }
    for (let step = 1; step <= 4; step++) {
      await window.keyboard.press("Control+Shift+Tab");
      await expect(shownTab(window).getByTestId("tab-title")).toHaveText(
        mindTitles[(7 - step) % 4] ?? "",
      );
      await expectResting(window, `⌃⇧Tab, step ${step}`);
    }

    // A click on a tab that is whole already leaves the strip where it is.
    await window.keyboard.press("Control+Tab");
    await expectResting(window, "the second tab");
    const placed = await scrollOf(list);
    const visible = await list.getByRole("tab").evaluateAll((tabs) => {
      const view = tabs[0]?.parentElement?.getBoundingClientRect();
      return tabs.findIndex((tab) => {
        const box = tab.getBoundingClientRect();
        return view !== undefined && box.left >= view.left && box.right <= view.right;
      });
    });
    expect(visible).toBeGreaterThanOrEqual(0);
    await list.getByRole("tab").nth(visible).click();
    await expectResting(window, "a click on a whole tab");
    expect(await scrollOf(list)).toBe(placed);

    // A Mind opened from the sidebar, its tab hidden, comes into view.
    const hidden = await list.getByRole("tab").evaluateAll((tabs) => {
      const view = tabs[0]?.parentElement?.getBoundingClientRect();
      return tabs.findIndex((tab) => {
        const box = tab.getBoundingClientRect();
        return view !== undefined && (box.right <= view.left + 1 || box.left >= view.right - 1);
      });
    });
    expect(hidden).toBeGreaterThanOrEqual(0);
    const hiddenName = mindTitles[hidden] ?? "";
    await window.getByTestId("mind-list-item").filter({ hasText: hiddenName }).first().click();
    await expect(shownTab(window).getByTestId("tab-title")).toHaveText(hiddenName);
    await expectResting(window, "a Mind opened from the sidebar");

    // A new tab, then its name grown, which widens the tab after it was brought into view.
    await addMind(window, groupAt(window, 0), "Q3");
    await expectResting(window, "a short name");
    await nameMind(window, `${nameOf(language, 5)}, with a longer name than before`);
    await expectResting(window, "its name grown");

    // A mouse wheel scrolls the strip, and it rests on whole tabs when it stops.
    const box = await list.boundingBox();
    if (!box) throw new Error("The strip isn't visible.");
    await window.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await window.mouse.wheel(0, -2000);
    await expect(list).not.toHaveAttribute("data-hidden-before");
    await expectResting(window, "scrolled to the start", { shownInView: false });
    await window.mouse.wheel(0, 70);
    await expect.poll(() => scrollOf(list)).toBeGreaterThan(0);
    await expectResting(window, "scrolled a little", { shownInView: false });

    // A Document opened from the sidebar beside the Minds, its tab hidden or not, is in view too.
    await window.getByTestId("document-list-item").filter({ hasText: REPORT[language] }).click();
    await expect(window.getByTestId("document-tab")).toHaveCount(1);
    await expectResting(window, "a Document opened");
    await openDocumentAt(window, { documentId: report, ...OPENS_PDF });
    await expectResting(window, "a Citation opened the Document");
    await app.close();
  });
}
