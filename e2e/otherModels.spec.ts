import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  acceptPlainTextKeys,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  openSettings,
  pointAndClick,
  removeDataFolder,
  setWindowSize,
  showSettingsPage,
} from "./app";

/**
 * The model menu holds a provider's current chat models, and ids it lists
 * beyond them wait under Settings → Models → Other models until the User turns
 * "In the menu" on (#513). What a provider's list holds is tested in the core,
 * with stubbed lists; here a model is added, as from a list, and the menu follows.
 * Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (SCREENSHOTS) await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

let dataDir: string;
let home: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  home = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
});

test("a model added from Other models is in the model chip's menu, and leaves it when turned off", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  try {
    await dismissChatSetup(window);
    await acceptPlainTextKeys(window);
    await setWindowSize(app, window, 1440, 800);
    const providerId = await window.evaluate(async () => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const provider = await bridge.saveChatProvider({
        kind: "openai",
        modelId: "gpt-6.1-sol",
        apiKey: "key-not-real",
      });
      await bridge.setChatModelInMenu({
        providerId: provider.id,
        modelId: "o3-deep-research",
        inMenu: true,
      });
      return provider.id;
    });
    await newMind(window);
    const options = window.getByTestId("composer-model-menu").getByTestId("composer-model-option");
    const menuModels = async () => {
      await pointAndClick(window, window.getByTestId("composer-model"));
      await expect(options.first()).toBeVisible();
      const models = await options.evaluateAll((all) =>
        all.map((each) => each.getAttribute("data-model")),
      );
      await window.keyboard.press("Escape");
      return models;
    };

    await expect.poll(menuModels).toEqual(["gpt-6.1-sol", "o3-deep-research"]);

    const settings = await openSettings(window, "models");
    const other = settings.getByTestId("other-models-settings");
    await expect(other.getByRole("heading", { name: "Other models" })).toBeVisible();
    const row = other.getByTestId("other-model").filter({ hasText: "o3-deep-research" });
    await expect(row.getByRole("checkbox", { name: "In the menu" })).toBeChecked();
    await screenshot(settings, "other-models-en");

    await row.getByRole("checkbox", { name: "In the menu" }).click();
    await expect(other).toBeHidden();
    await settings.getByRole("button", { name: "Close Settings" }).click();
    await expect.poll(menuModels).toEqual(["gpt-6.1-sol"]);

    await window.evaluate(
      (id) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.setChatModelInMenu({
          providerId: id,
          modelId: "o3-deep-research",
          inMenu: true,
        }),
      providerId,
    );
    await window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "zh-CN" },
      }),
    );
    await window.getByRole("button", { name: "设置" }).click();
    const settingsZh = await showSettingsPage(window, "models");
    const otherZh = settingsZh.getByTestId("other-models-settings");
    await expect(otherZh.getByRole("heading", { name: "其他模型" })).toBeVisible();
    await expect(otherZh.getByRole("checkbox", { name: "在菜单中" })).toBeChecked();
    await screenshot(settingsZh, "other-models-zh");
  } finally {
    await app.close();
  }
});
