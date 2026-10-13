import { expect, test } from "@playwright/test";
import { createDataFolder, launchApp, removeDataFolder, setLanguage, setWindowSize } from "./app";

/**
 * The example Mind follows a change of interface language while nobody has
 * touched it (#709): its title, its Question, its Answer and its Documents become
 * the other language's, with every Citation found, and back again.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("switching the interface to Chinese and back swaps the untouched example, its Answer and its Documents", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true });
  await setWindowSize(app, window, 1440, 900);
  const title = window.getByTestId("mind-title");
  await expect(title).toHaveValue("Where tea comes from");
  const checks = window.getByTestId("mind-pane").getByTestId("margin-check");
  await expect(checks.nth(1)).toHaveAttribute("data-check", "found", { timeout: 30_000 });

  await setLanguage(window, "zh-CN");
  await expect(title).toHaveValue("茶从哪里来", { timeout: 30_000 });
  await expect(window.getByTestId("mind-editor")).toContainText("茶");
  await expect(checks).toHaveCount(2);
  await expect(checks.nth(0)).toHaveAttribute("data-check", "found", { timeout: 30_000 });
  await expect(checks.nth(1)).toHaveAttribute("data-check", "found", { timeout: 30_000 });

  await setLanguage(window, "en");
  await expect(title).toHaveValue("Where tea comes from", { timeout: 30_000 });
  await expect(checks.nth(1)).toHaveAttribute("data-check", "found", { timeout: 30_000 });
  await app.close();
});
