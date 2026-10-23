import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import { translate } from "../src/shared/i18n";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  openSettings,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  showSettingsPage,
  tabTo,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function fits(window: Page, tooltip: Locator) {
  await motionSettled(window);
  const box = await tooltip.boundingBox();
  expect(box).not.toBeNull();
  const viewport = await window.evaluate(() => ({ width: innerWidth, height: innerHeight }));
  expect(box?.x ?? -1).toBeGreaterThanOrEqual(0);
  expect(box?.y ?? -1).toBeGreaterThanOrEqual(0);
  expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual(viewport.width);
  expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual(viewport.height);
}

async function shot(window: Page, name: string) {
  const directory = process.env.INCARNAMIND_SCREENSHOTS;
  if (!directory) return;
  await mkdir(directory, { recursive: true });
  await window.screenshot({ path: join(directory, `${name}.png`) });
}

for (const language of ["en", "zh-CN"] as const) {
  test(`Settings help opens by keyboard and pointer without changing its setting, and Escape closes only the help (${language})`, async () => {
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    const settings = await openSettings(window, "general");
    await setLanguage(window, language);
    const name = language === "en" ? "More information" : "查看说明";
    const tooltip = window.getByRole("tooltip");
    for (const width of [1440, 900]) {
      await setWindowSize(app, window, width, width === 900 ? 560 : 900);
      await showSettingsPage(window, "general");
      await window.mouse.move(0, 0);
      const cap = settings.getByTestId("spending-cap-settings");
      const help = cap.getByRole("button", { name, exact: true });
      const toggle = cap.getByTestId("spending-cap-switch");
      const checked = await toggle.isChecked();
      const explanation = translate(language, "cost.cap.tip");
      await expect(help).toBeVisible();
      await expect(toggle).toHaveAccessibleDescription(explanation);
      await tabTo(window, help);
      await expect(help).toBeFocused();
      expect(await help.evaluate((each) => getComputedStyle(each).outlineStyle)).not.toBe("none");
      await expect(help).toHaveAccessibleDescription(explanation);
      await expect(tooltip).toHaveText(explanation);
      await fits(window, tooltip);
      await shot(window, `help-general-${language}-${width}`);
      await window.keyboard.press("Escape");
      await expect(tooltip).toHaveCount(0);
      await expect(settings).toBeVisible();
      await expect(help).toBeFocused();
      await expect(toggle).toHaveAccessibleDescription(explanation);
      await window.keyboard.press("Enter");
      await expect(tooltip).toBeVisible();
      await window.keyboard.press("Escape");
      await window.keyboard.press("Space");
      await expect(tooltip).toBeVisible();
      await window.keyboard.press("Tab");
      await expect(toggle).toBeFocused();
      await expect(tooltip).toHaveCount(0);
      await expect(toggle).toBeChecked({ checked });

      await showSettingsPage(window, "privacy");
      const updates = settings.locator('[data-traffic-id="update-check"]');
      const info = updates.getByRole("button", { name, exact: true });
      const updateSwitch = updates.getByTestId("automatic-update-checks");
      const updatesChecked = await updateSwitch.isChecked();
      await tabTo(window, info);
      await expect(info).toBeFocused();
      await expect(tooltip).toHaveText(translate(language, "privacy.traffic.update-check.tip"));
      await fits(window, tooltip);
      await shot(window, `help-privacy-${language}-${width}`);
      await window.keyboard.press("Escape");
      await expect(tooltip).toHaveCount(0);
      await expect(settings).toBeVisible();
      await expect(info).toBeFocused();
      await window.keyboard.press("Tab");
      await info.hover();
      await expect(tooltip).toBeVisible();
      await tooltip.hover();
      await expect(tooltip).toBeVisible();
      await window.mouse.move(0, 0);
      await expect(tooltip).toHaveCount(0);
      await info.hover();
      await tooltip.hover();
      await window.keyboard.press("Escape");
      await expect(tooltip).toHaveCount(0);
      await window.mouse.move(0, 0);
      await tabTo(window, info);
      await expect(tooltip).toBeVisible();
      await window.keyboard.press("Tab");
      await expect(tooltip).toHaveCount(0);
      await pointAndClick(window, info);
      await expect(tooltip).toBeVisible();
      await window.keyboard.press("Escape");
      await expect(tooltip).toHaveCount(0);
      await expect(settings).toBeVisible();
      await expect(updateSwitch).toBeChecked({ checked: updatesChecked });
    }
  });
}
