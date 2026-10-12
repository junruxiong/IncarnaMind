import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
  WORD_KEY,
} from "./app";

/*
 * The word count under the Mind's title: words for English, characters for
 * Chinese, the selection's own count, a target per Mind, and a count that
 * waits for a pause in typing. Set INCARNAMIND_SCREENSHOTS to a folder to also
 * save screenshots of it at a wide and a narrow window, in English and Chinese.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const setLanguage = (window: Page, language: "en" | "zh-CN") =>
  window.evaluate(
    (value) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: value },
      }),
    language,
  );

test("the count follows the writing, in words and in characters, and the selection has its own", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window);
  const button = window.getByTestId("mind-count-button");
  await expect(button).toHaveText("0 words");

  await window.getByTestId("mind-editor").click();
  await window.keyboard.type("The tide turns at noon");
  await expect(button).toHaveText("5 words");
  await expect(button).toHaveAttribute("title", /Citation markers/);

  // Select the last two words with the keyboard.
  await window.keyboard.press(`Shift+${WORD_KEY}+ArrowLeft`);
  await window.keyboard.press(`Shift+${WORD_KEY}+ArrowLeft`);
  await expect(button).toHaveText("2 of 5 words");
  await window.keyboard.press("ArrowRight");
  await expect(button).toHaveText("5 words");

  // Chinese is counted by character.
  await window.keyboard.press("Enter");
  await window.keyboard.type("潮水每天涨落两次");
  await expect(button).toHaveText("13 characters");
  await window.keyboard.press("Enter");
  await window.keyboard.type("潮水每天涨落两次潮水每天涨落两次潮水每天涨落两次");
  await expect(button).toHaveText("37 characters");
  await app.close();
});

test("typing is not slowed: the count is worked out after a pause, not for each key", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window);
  const button = window.getByTestId("mind-count-button");
  await window.getByTestId("mind-editor").click();

  // Count how many times the count changes while a burst of keys goes in.
  await window.evaluate(() => {
    const target = document.querySelector('[data-testid="mind-count-button"]');
    const seen = { changes: 0 };
    (globalThis as unknown as { countChanges: typeof seen }).countChanges = seen;
    if (target)
      new MutationObserver(() => {
        seen.changes += 1;
      }).observe(target, { attributes: true, attributeFilter: ["data-total"] });
  });
  const burst = "word ".repeat(80);
  await window.keyboard.type(burst, { delay: 0 });
  await expect(button).toHaveAttribute("data-total", "80");
  const changes = await window.evaluate(
    () => (globalThis as unknown as { countChanges: { changes: number } }).countChanges.changes,
  );
  expect(changes).toBeLessThan(burst.length / 4);
  await app.close();
});

test("a target shows progress, is kept for the Mind, and can be removed", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  await dismissChatSetup(window);
  await newMind(window);
  const button = window.getByTestId("mind-count-button");
  await window.getByTestId("mind-editor").click();
  await window.keyboard.type("one two three four five");
  await expect(window.getByTestId("mind-editor")).toHaveText("one two three four five");
  await expect(button).toHaveText("5 words");
  await expect(window.getByTestId("mind-count-progress")).toHaveCount(0);

  await button.click();
  const menu = window.getByTestId("mind-count-menu");
  await expect(menu).toBeVisible();
  // The menu puts the focus on its checked item as it opens; wait for that before typing in the field.
  await expect(
    window.locator('[data-testid="mind-count-basis"][aria-checked="true"]'),
  ).toBeFocused();
  const field = window.getByTestId("mind-count-target");
  await field.fill("words");
  await window.getByTestId("mind-count-set").click();
  await expect(menu.getByRole("alert")).toBeVisible();
  await field.fill("20");
  await field.press("Enter");
  await expect(menu).toBeHidden();
  await expect(button).toHaveText("5 / 20 words");
  await expect(window.getByTestId("mind-count-progress")).toHaveAttribute("aria-valuenow", "25");

  // What the count shows is chosen in the menu: characters without spaces here.
  await button.click();
  await window.locator('[data-testid="mind-count-basis"][data-basis="charactersNoSpaces"]').click();
  await expect(button).toHaveText("19 / 20 characters");
  await first.app.close();

  // After a restart the Mind still has both.
  const second = await launchApp(dataDir);
  await second.window.getByTestId("mind-list-item").click();
  await expect(second.window.getByTestId("mind-count-button")).toHaveText("19 / 20 characters");

  await second.window.getByTestId("mind-count-button").click();
  await second.window.getByTestId("mind-count-remove").click();
  await expect(second.window.getByTestId("mind-count-button")).toHaveText("19 characters");
  await expect(second.window.getByTestId("mind-count-progress")).toHaveCount(0);
  await second.app.close();
});

for (const language of ["en", "zh-CN"] as const) {
  for (const width of [1440, 900]) {
    test(`the count fits under the title at ${width}px in ${language}`, async () => {
      const { app, window } = await launchApp(dataDir);
      await dismissChatSetup(window);
      await setWindowSize(app, window, width, 700);
      await setLanguage(window, language);
      await newMind(window);
      await window.getByTestId("mind-title").fill("Tides of the Aldermoor coast");
      await window.getByTestId("mind-editor").click();
      await window.keyboard.type(
        language === "en"
          ? "The tide turns at noon, and again near midnight."
          : "潮水在正午转向，午夜前后再次转向。",
      );
      const button = window.getByTestId("mind-count-button");
      await button.click();
      await expect(
        window.locator('[data-testid="mind-count-basis"][aria-checked="true"]'),
      ).toBeFocused();
      const target = window.getByTestId("mind-count-target");
      await target.click();
      await target.fill(language === "en" ? "8,000" : "3万");
      await target.press("Enter");
      await expect(window.getByTestId("mind-count-progress")).toBeVisible();
      // The count waits for a pause in typing.
      await expect(button).not.toHaveAttribute("data-total", "0");

      // Under the title, inside the card, and nothing scrolls sideways.
      const title = await window.getByTestId("mind-title").boundingBox();
      const count = await window.getByTestId("mind-count").boundingBox();
      const card = await window.getByTestId("mind-pane").boundingBox();
      if (!title || !count || !card) throw new Error("Not laid out.");
      expect(count.y).toBeGreaterThanOrEqual(title.y + title.height - 20);
      expect(count.x + count.width).toBeLessThanOrEqual(card.x + card.width);
      expect(await window.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        width,
      );
      if (SCREENSHOTS)
        await window.screenshot({ path: join(SCREENSHOTS, `count-${language}-${width}.png`) });
      await app.close();
    });
  }
}
