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
 * The model menu holds a provider's current chat models; what it lists beyond
 * them waits under Settings → Models → Other models, grouped by why, until the
 * User ticks it (#513). What each provider's list holds is tested in the core;
 * here OpenAI's list is stubbed in the main process. Set INCARNAMIND_SCREENSHOTS
 * to a folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (SCREENSHOTS) await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const LISTED = [
  "gpt-6.1-sol",
  "gpt-6-luna",
  "gpt-4-turbo",
  "gpt-6.2-sol-preview",
  "babbage-002",
  "gpt-audio",
  "o3-deep-research",
];

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

test("Other models: grouped by why, ticked into the model menu, with a wrong tick taken back from the Problem line", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  try {
    await dismissChatSetup(window);
    await acceptPlainTextKeys(window);
    await setWindowSize(app, window, 1100, 900);
    await app.evaluate((_electron, ids) => {
      const original = globalThis.fetch;
      globalThis.fetch = (input, init) => {
        const url =
          typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        return url === "https://api.openai.com/v1/models"
          ? Promise.resolve(Response.json({ data: ids.map((id) => ({ id })) }))
          : original(input, init);
      };
    }, LISTED);
    const providerId = await window.evaluate(async () => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const provider = await bridge.saveChatProvider({
        kind: "openai",
        modelId: "gpt-6.1-sol",
        apiKey: "key-not-real",
      });
      if (provider.service) await bridge.allowDataFlow("chat", provider.service.id);
      await bridge.setChatModelInMenu({
        providerId: provider.id,
        modelId: "gpt-9-not-found",
        inMenu: true,
      });
      return provider.id;
    });
    await newMind(window);
    const chip = window.getByTestId("composer-model");
    const options = window.getByTestId("composer-model-menu").getByTestId("composer-model-option");
    const menuModels = async () => {
      await pointAndClick(window, chip);
      await expect(options.first()).toBeVisible();
      const models = await options.evaluateAll((all) =>
        all.map((each) => each.getAttribute("data-model")),
      );
      await window.keyboard.press("Escape");
      return models;
    };
    await expect.poll(menuModels).toEqual(["gpt-6.1-sol", "gpt-6-luna", "gpt-9-not-found"]);

    // Closed at first: one row with the count.
    const settings = await openSettings(window, "models");
    const other = settings.getByTestId("other-models-settings");
    const toggle = other.getByTestId("other-models-toggle");
    await expect(toggle).toContainText("Other models · 6");
    await expect(toggle).toContainText("Models OpenAI lists that aren’t in the menu");
    await expect(other.getByTestId("other-model")).toHaveCount(0);
    await screenshot(window, "other-models-collapsed-en");

    // Open, by keyboard: three groups, and the one that can't answer has no box.
    await toggle.focus();
    await window.keyboard.press("Enter");
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
    await expect(other).toContainText("Tick a model to add it to the model menu.");
    const rows = other.getByTestId("other-model");
    const ids = () => rows.evaluateAll((all) => all.map((each) => each.getAttribute("data-model")));
    const order = [
      "gpt-6.2-sol-preview",
      "gpt-9-not-found",
      "gpt-4-turbo",
      "babbage-002",
      "gpt-audio",
      "o3-deep-research",
    ];
    await expect.poll(ids).toEqual(order);
    const group = (name: string) => other.getByRole("group", { name });
    await expect(group("Not known to IncarnaMind")).toContainText("May not work");
    await expect(group("Older models")).toContainText("gpt-4-turbo");
    const cannot = group("Can’t answer Questions here");
    await expect(cannot.getByRole("checkbox")).toHaveCount(0);
    await expect(cannot).toContainText("Completions only");
    await expect(cannot).toContainText("Audio model");
    await expect(cannot).toContainText("Needs the deep research API");
    await expect(
      other.getByTestId("other-model").filter({ hasText: "gpt-9-not-found" }).getByRole("checkbox"),
    ).toBeChecked();
    await screenshot(window, "other-models-open-en");

    // Tick by clicking the row: it goes into the menu and doesn't move.
    const older = rows.filter({ hasText: "gpt-4-turbo" });
    await pointAndClick(window, older);
    await expect(older.getByRole("checkbox")).toBeChecked();
    await expect.poll(ids).toEqual(order);
    await settings.getByRole("button", { name: "Close Settings" }).click();
    await expect
      .poll(menuModels)
      .toEqual(["gpt-6.1-sol", "gpt-6-luna", "gpt-9-not-found", "gpt-4-turbo"]);

    // Untick: it leaves the menu again.
    await openSettings(window, "models");
    await toggle.click();
    await pointAndClick(window, rows.filter({ hasText: "gpt-4-turbo" }));
    await expect(rows.filter({ hasText: "gpt-4-turbo" }).getByRole("checkbox")).not.toBeChecked();
    await settings.getByRole("button", { name: "Close Settings" }).click();
    await expect.poll(menuModels).toEqual(["gpt-6.1-sol", "gpt-6-luna", "gpt-9-not-found"]);

    // A wrong tick: its first Question fails, and the Problem line takes it back.
    await pointAndClick(window, chip);
    await pointAndClick(window, options.filter({ hasText: "gpt-9-not-found" }));
    await expect(window.getByTestId("composer-input")).toBeFocused();
    await window.keyboard.type("Which model answers?");
    await window.keyboard.press("Enter");
    const problem = window.getByTestId("answer-error");
    await expect(problem).toContainText("gpt-9-not-found");
    await screenshot(problem, "other-models-problem-en");
    await pointAndClick(window, problem.getByTestId("answer-error-action"));
    await expect(problem.getByTestId("answer-error-action")).toHaveText("Choose another model");
    // The Mind still has it chosen; once another is, the menu no longer lists it.
    await pointAndClick(window, chip);
    await pointAndClick(window, options.filter({ hasText: "gpt-6-luna" }));
    await expect.poll(menuModels).toEqual(["gpt-6.1-sol", "gpt-6-luna"]);

    // In Chinese.
    await window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "zh-CN" },
      }),
    );
    await window.getByRole("button", { name: "设置" }).click();
    const settingsZh = await showSettingsPage(window, "models");
    const otherZh = settingsZh.getByTestId("other-models-settings");
    await expect(otherZh.getByTestId("other-models-toggle")).toContainText("其他模型 · 5");
    await otherZh.getByTestId("other-models-toggle").click();
    await expect(otherZh).toContainText("勾选模型即可加入模型菜单。");
    await expect(otherZh.getByRole("group", { name: "无法在这里回答问题" })).toContainText(
      "仅支持文本补全",
    );
    await screenshot(window, "other-models-open-zh");
    await otherZh.getByTestId("other-models-toggle").click();
    await screenshot(window, "other-models-collapsed-zh");
    expect(providerId).not.toBe("");
  } finally {
    await app.close();
  }
});
