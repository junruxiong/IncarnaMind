import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import { longContract } from "../tests/helpers/longContract";
import { contractFile, resumeFile } from "../tests/helpers/wordLookFiles";
import {
  createDataFolder,
  dismissChatSetup,
  importWordFromMenu,
  interceptOpenDialog,
  launchApp,
  motionSettled,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

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

const fontOf = (target: Locator) =>
  target.evaluate((element) => getComputedStyle(element).fontFamily);

async function chooseLook(window: Page, look: "word" | "plain") {
  await window.getByTestId("group-menu").click();
  await window.getByTestId(`look-${look}`).click();
}

test("a contract imported from Word shows the file's look, and Look › Plain Mind draws it as any Mind, and back", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await importFile(app, window, "Supply agreement v5 (from Northwind).docx", contractFile());

  const editor = window.getByTestId("mind-editor");
  const clause = editor.locator("p", { hasText: "Prices are fixed" });
  const title = editor.locator("[data-word-look]", { hasText: "Supply agreement" }).first();
  await expect(clause).toHaveCSS("text-align", "justify");
  await expect(title).toHaveCSS("text-align", "center");
  await expect(title).toHaveCSS("font-variant-caps", "small-caps");
  expect(await fontOf(title)).toMatch(/^"Book Antiqua"/);
  // The app's own parts stay the app's.
  expect(await fontOf(window.getByTestId("mind-title"))).not.toMatch(/Book Antiqua/);
  if (SCREENSHOTS) {
    await window.screenshot({ path: join(SCREENSHOTS, "word-look-contract.png") });
  }

  await window.getByTestId("group-menu").click();
  await expect(window.getByTestId("look-word")).toHaveAttribute("aria-checked", "true");
  await expect(window.getByTestId("look-plain")).toHaveAttribute("aria-checked", "false");
  if (SCREENSHOTS) {
    await motionSettled(window);
    await window.screenshot({ path: join(SCREENSHOTS, "word-look-menu.png") });
  }
  await window.getByTestId("look-plain").click();
  await expect(clause).toHaveCSS("text-align", "start");
  expect(await fontOf(title)).not.toMatch(/Book Antiqua/);
  await expect(clause).toContainText("Prices are fixed for the first twelve months");
  if (SCREENSHOTS) {
    await motionSettled(window);
    await window.screenshot({ path: join(SCREENSHOTS, "word-look-contract-plain.png") });
  }

  await chooseLook(window, "word");
  await expect(clause).toHaveCSS("text-align", "justify");
  expect(await fontOf(title)).toMatch(/^"Book Antiqua"/);
  if (SCREENSHOTS) {
    await setWindowSize(app, window, 1100, 720);
    await motionSettled(window);
    await window.screenshot({ path: join(SCREENSHOTS, "word-look-contract-1100.png") });
  }

  await setLanguage(window, "zh-CN");
  await window.getByTestId("group-menu").click();
  const menu = window.getByTestId("group-menu-items");
  await expect(menu).toContainText("外观");
  await expect(window.getByTestId("look-word")).toHaveText("Word 文件原样");
  await expect(window.getByTestId("look-plain")).toHaveText("普通 Mind");
  if (SCREENSHOTS) {
    await motionSettled(window);
    await window.screenshot({ path: join(SCREENSHOTS, "word-look-menu-zh.png") });
  }
  await app.close();
});

test("the 304-page contract opens in its look, with one rule a style", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await dismissChatSetup(window);
  await importFile(app, window, "Framework Services Agreement.docx", longContract());

  const editor = window.getByTestId("mind-editor");
  // The title and every clause's heading in: the import has written every Block.
  await expect(editor.locator("h1")).toHaveCount(241, { timeout: 120_000 });
  await expect(editor.locator("p", { hasText: /^240\.11 / })).toHaveCSS("text-align", "justify");
  const rules = await window.evaluate(() =>
    [...document.querySelectorAll(".mind-editor-frame > style")].map(
      (style) => (style as HTMLStyleElement).sheet?.cssRules.length ?? 0,
    ),
  );
  expect(rules).toHaveLength(1);
  expect(rules[0]).toBeLessThan(20);
  await app.close();
});

test("two Word Minds side by side each keep their own look", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await importFile(app, window, "林晓_简历.docx", resumeFile());
  await importFile(app, window, "Supply agreement.docx", contractFile());
  await window.getByTestId("split-group").click();
  await expect(window.getByRole("region", { name: "Group 2" })).toBeVisible();
  // Split opens the shown Mind beside; the first group goes back to the résumé.
  await window
    .getByRole("region", { name: "Group 1" })
    .getByRole("tab", { name: "林晓_简历" })
    .click();
  const panes = window.locator('[data-testid="mind-pane"]:visible');
  await expect(panes).toHaveCount(2);

  const resume = panes.filter({ hasText: "林　晓" });
  const contract = panes.filter({ hasText: "Supply agreement" });
  const name = resume.locator("[data-word-look]", { hasText: "林　晓" }).first();
  await expect(name).toHaveCSS("text-align", "center");
  expect(await fontOf(name)).toMatch(/华文中宋/);
  expect(await fontOf(name)).not.toMatch(/Book Antiqua/);
  const clause = contract.locator("p", { hasText: "Prices are fixed" });
  await expect(clause).toHaveCSS("text-align", "justify");
  expect(await fontOf(clause)).toMatch(/^"Book Antiqua"/);
  if (SCREENSHOTS) {
    await window.screenshot({ path: join(SCREENSHOTS, "word-look-side-by-side.png") });
  }
  await app.close();
});
