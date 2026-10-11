import { expect, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  enterContent,
  launchApp,
  removeDataFolder,
  setWindowSize,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("a Mind with no title is named by its first line, in the sidebar and the tab, until a title is typed", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);

  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-mind").click();
  const title = window.getByTestId("mind-title");
  await expect(title).toBeFocused();
  const row = window.getByTestId("mind-list-item");
  const tab = window.getByTestId("mind-tab");
  await expect(row).toContainText("Untitled");
  await expect(title).toHaveAttribute("placeholder", "Untitled");

  await enterContent(window);
  await window.keyboard.type("Tides in Bergen");
  await expect(row).toContainText("Tides in Bergen");
  await expect(tab).toContainText("Tides in Bergen");
  await expect(title).toHaveAttribute("placeholder", "Tides in Bergen");
  await expect(title).toHaveValue("");

  await window.keyboard.type(" and the moon");
  await expect(row).toContainText("Tides in Bergen and the moon");

  await title.click();
  await window.keyboard.type("My own title");
  await expect(row).toContainText("My own title");
  await expect(row).not.toContainText("Tides");
  await expect(tab).toContainText("My own title");

  await title.press("ControlOrMeta+a");
  await title.press("Backspace");
  await expect(row).toContainText("Tides in Bergen and the moon");
});

test("a long first line is cut at a word, and a Chinese one at a character", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);

  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-mind").click();
  await enterContent(window);
  await window.keyboard.type(
    "How much cooler are large irrigated parks than the surrounding built-up streets during clear windless summer nights",
  );
  const row = window.getByTestId("mind-list-item");
  await expect(row).toContainText("parks than the");
  await expect(row).not.toContainText("surrounding");

  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-mind").click();
  await enterContent(window);
  await window.keyboard.type("夜间降温".repeat(20));
  await expect(window.getByTestId("mind-list-item").first()).toContainText("夜间降温".repeat(15));
  await expect(window.getByTestId("mind-list-item").first()).not.toContainText(
    "夜间降温".repeat(16),
  );
});
