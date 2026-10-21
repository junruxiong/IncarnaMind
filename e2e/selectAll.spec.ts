import { join } from "node:path";
import { expect, type Locator, test } from "@playwright/test";
import { translate } from "../src/shared/i18n";
import {
  clickEditItem,
  coreOf,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  openManageTags,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  showSettingsPage,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

for (const language of ["en", "zh-CN"] as const) {
  for (const route of ["menu", "keyboard"] as const) {
    test(`${route} Select All replaces text in the palette, Tags and monthly cap (${language})`, async () => {
      const { app, window } = await launchApp(dataDir);
      try {
        await setWindowSize(app, window, language === "en" ? 1280 : 900, 660);
        await dismissChatSetup(window);
        await setLanguage(window, language);
        const core = await coreOf(window);
        // Playwright's key dispatch can bypass the macOS menu that takes ⌘A (#1351).
        const selectAll = async (field: Locator) => {
          const value = await field.inputValue();
          if (route === "menu") await clickEditItem(app, "edit.selectAll");
          else await window.keyboard.press("ControlOrMeta+a");
          await expect
            .poll(() =>
              field.evaluate((element) => {
                if (!(element instanceof HTMLInputElement)) throw new Error("Not an input.");
                return element.selectionStart === null
                  ? document.getSelection()?.toString()
                  : element.value.slice(element.selectionStart, element.selectionEnd ?? 0);
              }),
            )
            .toBe(value);
        };
        const shot = async (name: string) => {
          const folder = process.env.INCARNAMIND_SCREENSHOTS;
          if (route === "menu" && folder) {
            await motionSettled(window);
            await window.screenshot({ path: join(folder, `${name}-${language}.png`) });
          }
        };

        await window.keyboard.press("ControlOrMeta+k");
        const palette = window.getByTestId("palette-search");
        await expect(palette).toBeFocused();
        await window.keyboard.type("abc def");
        await selectAll(palette);
        await window.keyboard.type("Z");
        await expect(palette).toHaveValue("Z");
        await selectAll(palette);
        await window.keyboard.type("财务 review 🧾");
        await selectAll(palette);
        await window.keyboard.type("财务");
        await expect(palette).toHaveValue("财务");
        await shot("palette-after");

        const minds = await core.evaluate(async (bridge) => (await bridge.listMinds()).length);
        await window.keyboard.press("ControlOrMeta+t");
        await expect(palette).toBeFocused();
        expect(await core.evaluate(async (bridge) => (await bridge.listMinds()).length)).toBe(
          minds,
        );
        await window.keyboard.press("Escape");

        await openManageTags(window, language);
        const tags = window.getByTestId("tags-dialog");
        const search = tags.getByTestId("tag-search");
        await pointAndClick(window, search);
        await window.keyboard.type("caiwu");
        await selectAll(search);
        await window.keyboard.type("cw");
        await expect(search).toHaveValue("cw");
        await selectAll(search);
        await window.keyboard.press("Backspace");
        await expect(search).toHaveValue("");
        await pointAndClick(
          window,
          tags.getByRole("button", { name: translate(language, "tags.dialog.done"), exact: true }),
        );
        await expect(tags).toBeHidden();

        await window.keyboard.press("ControlOrMeta+,");
        const settings = await showSettingsPage(window, "general");
        const toggle = settings.getByTestId("spending-cap-switch");
        await pointAndClick(window, toggle);
        const amount = settings.getByTestId("spending-cap-amount");
        await amount.fill("150");
        await window.keyboard.press("Enter");
        await expect
          .poll(() => core.evaluate(async (bridge) => (await bridge.getSettings()).user.monthlyCap))
          .toMatchObject({ amount: 150 });
        await pointAndClick(window, amount);
        await window.keyboard.press("End");
        await selectAll(amount);
        await window.keyboard.type("7");
        await window.keyboard.press("Tab");
        await expect(amount).toHaveValue("7");
        await expect
          .poll(() => core.evaluate(async (bridge) => (await bridge.getSettings()).user.monthlyCap))
          .toMatchObject({ amount: 7 });
        await shot("cap-after");
      } finally {
        await app.close();
      }
    });
  }
}
