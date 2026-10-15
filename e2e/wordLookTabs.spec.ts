import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Page, test } from "@playwright/test";
import { paragraph, run, wordFile } from "../tests/helpers/word";
import { contractFile, RESUME_TEXT_WIDTH_TWIPS, resumeFile } from "../tests/helpers/wordLookFiles";
import {
  createDataFolder,
  dismissChatSetup,
  importWordFromMenu,
  interceptOpenDialog,
  launchApp,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
/** The résumé's body size, which the Mind's writing size stands for. */
const RESUME_BODY = 10.5;
const RESUME_WIDTH = RESUME_TEXT_WIDTH_TWIPS / 20;

let dataDir: string;
let home: string;
let files: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  home = await createDataFolder();
  files = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
  await removeDataFolder(files);
});

async function importFile(app: ElectronApplication, window: Page, name: string, bytes: Buffer) {
  const path = join(files, name);
  await writeFile(path, bytes);
  await interceptOpenDialog(app, path);
  const dialog = await importWordFromMenu(window);
  await dialog.getByTestId("word-import-confirm").click();
  await expect(dialog).toBeHidden();
}

/** Where `text` starts and ends in the paragraph holding it, from the paragraph's left edge; `pt` is a point in pixels for a file whose body size is `body`. */
function placeOf(window: Page, text: string, body = RESUME_BODY) {
  return window.evaluate(
    ([wanted, size]) => {
      const editor = document.querySelector('[data-testid="mind-editor"]');
      const walker = document.createTreeWalker(editor ?? document.body, NodeFilter.SHOW_TEXT);
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const at = node.textContent?.indexOf(wanted) ?? -1;
        if (at < 0) continue;
        const range = document.createRange();
        range.setStart(node, at);
        range.setEnd(node, at + wanted.length);
        const box = range.getBoundingClientRect();
        const paragraph = node.parentElement?.closest("p");
        const line = paragraph?.getBoundingClientRect();
        if (!paragraph || !line) return null;
        return {
          left: box.left - line.left + paragraph.scrollLeft,
          right: box.right - line.left + paragraph.scrollLeft,
          width: paragraph.scrollWidth,
          shown: paragraph.clientWidth,
          overflow: getComputedStyle(paragraph).overflowX,
          pt: Number.parseFloat(getComputedStyle(editor ?? paragraph).fontSize) / size,
        };
      }
      return null;
    },
    [text, body] as const,
  );
}

async function chooseLook(window: Page, look: "word" | "plain") {
  await pointAndClick(window, window.getByTestId("group-menu"));
  await pointAndClick(window, window.getByTestId(`look-${look}`));
}

const pageScrollsSideways = (window: Page) =>
  window.evaluate(() => {
    const page = document.scrollingElement ?? document.documentElement;
    return page.scrollWidth > page.clientWidth;
  });

test("a résumé's tab stops line its columns up at the file's stops, its dates ending at the right margin's stop, and stay so as you type", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await importFile(app, window, "林晓_简历.docx", resumeFile());

  const date = "2023.09 – 2026.06";
  await expect.poll(async () => (await placeOf(window, date))?.right).toBeDefined();
  // The right stop is at the text width; the subject and degree start at their left stops.
  const offStop = async (text: string, edge: "left" | "right", at: number) => {
    const place = await placeOf(window, text);
    return place ? Math.abs(place[edge] - at * place.pt) : Number.POSITIVE_INFINITY;
  };
  await expect.poll(() => offStop(date, "right", RESUME_WIDTH)).toBeLessThan(3);
  expect(await offStop("金融学", "left", 120)).toBeLessThan(3);
  expect(await offStop("硕士", "left", 210)).toBeLessThan(3);
  const subject = await placeOf(window, "金融学");
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "word-tabs-resume.png") });

  // Typing at the line's start moves the text before the first stop, not the columns after it.
  const before = subject?.left ?? 0;
  await window
    .getByText("复旦大学", { exact: false })
    .first()
    .click({ position: { x: 2, y: 8 } });
  await window.keyboard.press("Home");
  await window.keyboard.type("A");
  await expect.poll(async () => (await placeOf(window, "金融学"))?.left).toBeCloseTo(before, 0);
  await expect.poll(() => offStop(date, "right", RESUME_WIDTH)).toBeLessThan(3);

  // Typed right after a tab, a letter is text in the column, not part of the tab.
  for (let key = 0; key < "复旦大学\t".length; key++) await window.keyboard.press("ArrowRight");
  await window.keyboard.type("B");
  await expect.poll(() => offStop("B金融学", "left", 120)).toBeLessThan(3);
  const tabs = window
    .getByTestId("mind-editor")
    .locator("p", { hasText: "B金融学" })
    .locator("[data-word-tab]");
  await expect(tabs).toHaveCount(3);
  for (const text of await tabs.allTextContents()) expect(text).toBe("\t");
  await expect.poll(() => offStop(date, "right", RESUME_WIDTH)).toBeLessThan(3);

  // Plain Mind draws a tab as any Mind does; the file's look lines the columns up again.
  await chooseLook(window, "plain");
  await expect.poll(() => offStop(date, "right", RESUME_WIDTH)).toBeGreaterThan(20);
  await chooseLook(window, "word");
  await expect.poll(() => offStop(date, "right", RESUME_WIDTH)).toBeLessThan(3);
  await app.close();
});

test("a clause's text after its number starts at the clause's hanging indent", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await importFile(app, window, "Supply agreement.docx", contractFile());
  const clause = window.getByTestId("mind-editor").locator("p", { hasText: "Prices are fixed" });
  await expect(clause.locator("[data-word-tab]")).toHaveCount(1);
  // The second line starts at the indent; so does the first line's text, after "2.1".
  const lines = () =>
    clause.evaluate((element) => {
      const lefts = (words: string) => {
        const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
          const at = node.textContent?.indexOf(words) ?? -1;
          if (at < 0) continue;
          const range = document.createRange();
          range.setStart(node, at);
          range.setEnd(node, at + 1);
          return range.getBoundingClientRect().left;
        }
        return Number.NaN;
      };
      return { number: lefts("2.1"), text: lefts("Prices"), second: lefts("that,") };
    });
  await expect
    .poll(async () => {
      const { text, second } = await lines();
      return Math.abs(text - second);
    })
    .toBeLessThan(2);
  const { number, text } = await lines();
  expect(text - number).toBeGreaterThan(30);
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "word-tabs-contract.png") });
  await app.close();
});

test("a tab that wraps onto another line when the column narrows reaches that line's next default stop", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  // One line in a whole window's column; two in half of it, where the tab follows "Leeds" on the second.
  const sentence = "The supplier delivers the goods to the buyer's warehouse in Leeds";
  await importFile(
    app,
    window,
    "Delivery.docx",
    wordFile({
      styles:
        '<w:docDefaults><w:rPrDefault><w:rPr><w:sz w:val="24"/></w:rPr></w:rPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>',
      body: paragraph(`${run(sentence)}<w:r><w:tab/></w:r>${run("Clause 4")}`),
    }),
  );
  // From the column's edge to "Clause 4", in points: a multiple of the 36 pt default stops.
  const offDefaultStop = async () => {
    const place = await placeOf(window, "Clause 4", 12);
    if (!place) return { at: Number.NaN, off: Number.POSITIVE_INFINITY };
    const points = place.left / place.pt;
    return { at: place.left, off: Math.abs(points - Math.round(points / 36) * 36) * place.pt };
  };
  await expect.poll(async () => (await offDefaultStop()).off).toBeLessThan(2);
  const wide = (await offDefaultStop()).at;
  // The Mind open beside itself: its column narrows where it is.
  await pointAndClick(window, window.getByTestId("split-group"));
  await expect(window.locator('[data-testid="mind-pane"]:visible')).toHaveCount(2);
  await expect.poll(async () => (await offDefaultStop()).at).not.toBe(wide);
  await expect.poll(async () => (await offDefaultStop()).off).toBeLessThan(2);
  await app.close();
});

test("a dotted leader fills a tab up to a right stop", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  const contents = paragraph(`${run("Introduction")}<w:r><w:tab/></w:r>${run("3")}`).replace(
    "<w:p>",
    '<w:p><w:pPr><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs></w:pPr>',
  );
  await importFile(app, window, "Contents.docx", wordFile({ body: contents }));
  const tab = window.getByTestId("mind-editor").locator("[data-word-tab]");
  await expect(tab).toHaveAttribute("data-leader", "dot");
  const page = await placeOf(window, "3");
  const title = await placeOf(window, "Introduction");
  expect(page && title && page.left - title.right).toBeGreaterThan(100);
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, "word-tabs-leader.png") });
  await app.close();
});

test("at 1,100 px a line of tab stops scrolls in its own frame, in English and Chinese, and the page never does", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1100, 720);
  await dismissChatSetup(window);
  await importFile(app, window, "林晓_简历.docx", resumeFile());
  const date = "2023.09 – 2026.06";
  for (const language of ["en", "zh-CN"] as const) {
    await setLanguage(window, language);
    await expect.poll(async () => (await placeOf(window, date))?.overflow).toBe("auto");
    const place = await placeOf(window, date);
    expect(place && place.width > place.shown).toBe(true);
    expect(await pageScrollsSideways(window)).toBe(false);
    if (SCREENSHOTS) {
      await window.screenshot({ path: join(SCREENSHOTS, `word-tabs-1100-${language}.png`) });
    }
  }
  await app.close();
});
