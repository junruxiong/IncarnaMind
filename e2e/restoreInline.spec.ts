import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  frontMind,
  launchApp,
  motionSettled,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * Review changes puts each change's own Restore original after its new words,
 * never over the text, and the applied line sits with the paragraph it
 * changed (#1018). The scripted model's "shorten" drops "very" and "really"
 * (非常 and 真的): two changes in one paragraph. Set INCARNAMIND_SCREENSHOTS to
 * a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const SIZES = [
  { width: 1440, height: 900 },
  { width: 1100, height: 760 },
] as const;

interface Case {
  language: "en" | "zh-CN";
  lines: readonly [string, string];
  ask: string;
  shortened: string;
  veryBack: string;
  reallyBack: string;
  restore: string;
}

const CASES: readonly Case[] = [
  {
    language: "en",
    lines: [
      "Tides rise twice a day along this coast.",
      "The tides here are very strong and really regular.",
    ],
    ask: "shorten it",
    shortened: "The tides here are strong and regular.",
    veryBack: "The tides here are very strong and regular.",
    reallyBack: "The tides here are strong and really regular.",
    restore: "Restore original",
  },
  {
    language: "zh-CN",
    lines: ["潮水每天涨落两次。", "潮水非常强，而且真的很有规律。"],
    ask: "缩短",
    shortened: "潮水强，而且很有规律。",
    veryBack: "潮水非常强，而且很有规律。",
    reallyBack: "潮水强，而且真的很有规律。",
    restore: "恢复原文",
  },
];

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** What the paragraph says: its own words, without the struck ones or the buttons drawn among them. */
const wordsOf = (paragraph: Locator) =>
  paragraph.evaluate((node) => {
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    let words = "";
    for (let each = walker.nextNode(); each; each = walker.nextNode()) {
      if (!each.parentElement?.closest(".applied-old, [contenteditable='false']")) {
        words += each.textContent ?? "";
      }
    }
    return words;
  });

/** How many of the Mind's controls overlap a word of its paragraph, the struck ones included. */
const controlsOverText = (editor: Locator) =>
  editor.evaluate((node) => {
    const rects = (range: Range) => [...range.getClientRects()];
    const words: DOMRect[] = [];
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    for (let each = walker.nextNode(); each; each = walker.nextNode()) {
      if (each.parentElement?.closest("button, [contenteditable='false']:not(.applied-old)")) {
        continue;
      }
      const range = document.createRange();
      range.selectNodeContents(each);
      words.push(...rects(range));
    }
    const frame = node.closest(".mind-editor-frame") ?? node;
    const controls = [...frame.querySelectorAll("button, .change-bar")].map((each) =>
      each.getBoundingClientRect(),
    );
    const meets = (a: DOMRect, b: DOMRect) =>
      a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1;
    return controls.filter((control) => words.some((word) => meets(control, word))).length;
  });

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Whether `a` comes before `b` as a page is read: on an earlier line, or the same line and further left. */
const readsBefore = (a: Box, b: Box) => {
  const apart = b.y + b.height / 2 - (a.y + a.height / 2);
  return Math.abs(apart) < Math.min(a.height, b.height) / 2 ? a.x < b.x : apart > 0;
};

const gapsAround = async (line: Locator, above: Locator, below: Locator) => {
  const [lineBox, aboveBox, belowBox] = await Promise.all([
    line.boundingBox(),
    above.boundingBox(),
    below.boundingBox(),
  ]);
  if (!lineBox || !aboveBox || !belowBox) throw new Error("The line or a paragraph isn't drawn.");
  return {
    above: lineBox.y - (aboveBox.y + aboveBox.height),
    below: belowBox.y - (lineBox.y + lineBox.height),
  };
};

for (const { language, lines, ask, shortened, veryBack, reallyBack, restore } of CASES) {
  test(`Review changes ends each change with its own Restore original, after its words and in reading order; the applied line sits with the paragraph it changed (${language})`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);

    for (const { width, height } of SIZES) {
      await setWindowSize(app, window, width, height);
      await newMind(window);
      const editor = frontMind(window).getByTestId("mind-editor");
      await editor.click();
      await window.keyboard.type(lines[0]);
      await window.keyboard.press("Enter");
      await window.keyboard.type(lines[1]);
      const paragraphs = editor.locator(":scope > p");
      await expect(paragraphs).toHaveText([...lines]);
      const [first, note] = [paragraphs.first(), paragraphs.nth(1)];
      const line = editor.getByTestId("applied-line");

      await window.keyboard.press("ControlOrMeta+a");
      await window.keyboard.press("ControlOrMeta+j");
      await expect(window.getByTestId("composer-input")).toBeFocused();
      await window.keyboard.type(ask);
      await window.keyboard.press("Enter");
      await expect(window.getByTestId("edit-pending")).toHaveCount(0);
      await expect.poll(() => wordsOf(note)).toBe(shortened);

      // The line is nearer the paragraph it changed than the one above it, with and without Review.
      const applied = await gapsAround(line, first, note);
      expect(applied.below).toBeGreaterThanOrEqual(0);
      expect(applied.below).toBeLessThan(applied.above);
      await expect(line.getByTestId("applied-count")).toHaveCSS("font-weight", "600");
      await expect(line.getByTestId("applied-review")).toHaveCSS("font-weight", "600");
      await expect(line.getByTestId("applied-review")).toHaveCSS("color", "rgb(23, 25, 28)");

      await pointAndClick(window, line.getByTestId("applied-review"));
      const restores = note.getByTestId("change-restore");
      const removed = note.locator("del.applied-old");
      await expect(restores).toHaveText([restore, restore]);
      await expect(removed).toHaveCount(2);
      const reviewing = await gapsAround(line, first, note);
      expect(reviewing.below).toBeGreaterThanOrEqual(0);
      expect(reviewing.below).toBeLessThan(reviewing.above);
      await expect(line.getByTestId("applied-count")).toHaveCSS("font-weight", "600");
      await expect(line.getByTestId("applied-done")).toHaveCSS("font-weight", "600");
      await expect(line.getByTestId("applied-done")).toHaveCSS("color", "rgb(23, 25, 28)");
      await expect(restores.first()).toHaveCSS("font-size", "12px");
      await expect(restores.first()).toHaveCSS("color", "rgb(101, 107, 116)");

      // Nothing floats over the words, and each Restore follows its own struck words.
      await expect(window.getByTestId("applied-change-bar")).toHaveCount(0);
      expect(await controlsOverText(editor)).toBe(0);
      const [struckFirst, struckLast, restoreFirst, restoreLast] = await Promise.all([
        removed.first().boundingBox(),
        removed.last().boundingBox(),
        restores.first().boundingBox(),
        restores.last().boundingBox(),
      ]);
      const editorBox = await editor.boundingBox();
      if (!struckFirst || !struckLast || !restoreFirst || !restoreLast || !editorBox) {
        throw new Error("A change isn't drawn.");
      }
      expect(readsBefore(struckFirst, restoreFirst)).toBe(true);
      expect(readsBefore(restoreFirst, struckLast)).toBe(true);
      expect(readsBefore(struckLast, restoreLast)).toBe(true);
      for (const box of [restoreFirst, restoreLast]) {
        expect(box.x + box.width).toBeLessThanOrEqual(editorBox.x + editorBox.width + 1);
      }
      await motionSettled(window);
      await screenshot(window, `restore-inline-${language}-${width}`);

      // By Tab, in reading order.
      await editor.focus();
      const reached: Box[] = [];
      for (let presses = 0; presses < 8 && reached.length < 2; presses++) {
        await window.keyboard.press("Tab");
        const box = await window.evaluate(() => {
          const active = document.activeElement;
          if (!(active instanceof HTMLElement) || active.dataset.testid !== "change-restore") {
            return null;
          }
          const { x, y, width, height } = active.getBoundingClientRect();
          return { x, y, width, height };
        });
        if (box) reached.push(box);
      }
      expect(reached).toHaveLength(2);
      const [early, late] = reached;
      expect(early && late && readsBefore(early, late)).toBe(true);

      // Enter on the second restores it alone, and Undo brings it back.
      await window.keyboard.press("Enter");
      await expect.poll(() => wordsOf(note)).toBe(reallyBack);
      await expect(removed).toHaveCount(1);
      await expect(restores).toHaveCount(1);
      await expect(window.getByTestId("undo-message")).toBeVisible();
      await window.keyboard.press("ControlOrMeta+z");
      await expect.poll(() => wordsOf(note)).toBe(shortened);

      // The mouse on the first restores it alone, and Undo brings it back.
      await expect(restores).toHaveCount(1);
      await pointAndClick(window, restores.first());
      await expect.poll(() => wordsOf(note)).toBe(veryBack);
      await window.keyboard.press("ControlOrMeta+z");
      await expect.poll(() => wordsOf(note)).toBe(shortened);
    }
    await app.close();
  });
}

const REPLACEMENTS = [
  {
    language: "en",
    line: "The tides here are strong and regular.",
    ask: "replace regular with punctual",
    old: "regular",
    added: "punctual",
  },
  {
    language: "zh-CN",
    line: "潮水每天非常规律地涨落两次。",
    ask: "把规律换成准时",
    old: "规律",
    added: "准时",
  },
] as const;

for (const { language, line: written, ask, old, added } of REPLACEMENTS) {
  test(`a replaced word has a space between its struck and new words, and its Restore original after them, which puts the old word back and leaves the line clear (${language})`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1440, 900);
    await newMind(window);
    const editor = frontMind(window).getByTestId("mind-editor");
    await editor.click();
    await window.keyboard.type(written);
    const note = editor.locator(":scope > p").first();
    await expect(note).toHaveText(written);

    await window.keyboard.press("ControlOrMeta+a");
    await window.keyboard.press("ControlOrMeta+j");
    await expect(window.getByTestId("composer-input")).toBeFocused();
    await window.keyboard.type(ask);
    await window.keyboard.press("Enter");
    await expect(window.getByTestId("edit-pending")).toHaveCount(0);
    const line = editor.getByTestId("applied-line");
    await pointAndClick(window, line.getByTestId("applied-review"));

    const struck = note.locator("del.applied-old");
    const underlined = note.locator(".applied-new");
    const restore = note.getByTestId("change-restore");
    await expect(struck).toHaveText(old);
    await expect(underlined).toHaveText(added);
    const [struckBox, addedBox, restoreBox] = await Promise.all([
      struck.boundingBox(),
      underlined.boundingBox(),
      restore.boundingBox(),
    ]);
    if (!struckBox || !addedBox || !restoreBox) throw new Error("The change isn't drawn.");
    expect(addedBox.x - (struckBox.x + struckBox.width)).toBeGreaterThanOrEqual(2);
    expect(restoreBox.x - (addedBox.x + addedBox.width)).toBeGreaterThanOrEqual(2);
    expect(await controlsOverText(editor)).toBe(0);

    await pointAndClick(window, restore);
    await expect.poll(() => wordsOf(note)).toBe(written);
    await expect(line).toHaveCount(0);
    await app.close();
  });
}
