import { expect, type Locator, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openSettings,
  removeDataFolder,
  type SettingsPage,
  setLanguage,
  showSettingsPage,
} from "./app";

/*
 * Small marks (#615): green is for Citation checks only, so "Stays on this
 * computer" is ink-secondary with the computer icon before it, in Settings
 * and Privacy (the model menu's is checked in composer.spec.ts), and no text
 * in Settings is the check green.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** ink-secondary and success (#4A4F57, #166534), as computed. */
const INK_SECONDARY = "rgb(74, 79, 87)";
const CHECK_GREEN = "rgb(22, 101, 52)";

const PAGES: SettingsPage[] = [
  "general",
  "models",
  "search",
  "meetings",
  "connectors",
  "skills",
  "agents",
  "privacy",
];

/** The text in a part of the page that is drawn in the check green. */
function greenTexts(scope: Locator): Promise<string[]> {
  return scope.evaluate((root, green) => {
    const found: string[] = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.textContent?.trim();
      const parent = node.parentElement;
      if (text && parent && getComputedStyle(parent).color === green) found.push(text);
    }
    return found;
  }, CHECK_GREEN);
}

async function expectOnThisComputer(marks: Locator): Promise<void> {
  const mark = marks.first();
  await expect(mark).toBeVisible();
  await expect(mark).toHaveCSS("color", INK_SECONDARY);
  // The icon comes before the words.
  expect(await mark.evaluate((element) => element.firstElementChild?.tagName)).toBe("svg");
}

test("“Stays on this computer” is ink-secondary with a computer icon in Settings and Privacy, and no text in Settings is the check green, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);

  // Settings stays open as the language changes under it.
  const settings = await openSettings(window, "models");
  for (const language of ["en", "zh-CN"] as const) {
    await setLanguage(window, language);

    // Settings › Models, with no provider yet: the choices, the local one marked.
    await showSettingsPage(window, "models");
    await expectOnThisComputer(
      settings.getByTestId("chat-provider-choices").getByTestId("on-this-computer"),
    );

    // Privacy: a data flow that goes nowhere.
    await showSettingsPage(window, "privacy");
    await expectOnThisComputer(
      window.getByTestId("data-flow-not-in-use").getByTestId("on-this-computer"),
    );

    // Nothing on any page of Settings is green: it has no Citation checks.
    for (const page of PAGES) {
      await showSettingsPage(window, page);
      expect(await greenTexts(settings), `${language}: ${page}`).toEqual([]);
    }
  }
  await app.close();
});
