import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  assistantAnswers,
  createDataFolder,
  launchApp,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

test("a first open shows only the example page: no tab strip, no assistant, a slim sidebar (#123)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await setWindowSize(app, window, 1440, 900);
  const sidebar = window.getByTestId("sidebar");
  await expect(window.getByTestId("mind-pane")).toBeVisible();

  await expect(window.getByTestId("tab-list")).toHaveCount(0);
  await expect(window.getByTestId("assistant")).toBeHidden();
  await expect(window.getByTestId("ask-pill")).toBeVisible();
  await expect(window.getByTestId("ask-pill")).toContainText("Ask anything");

  await expect(sidebar.getByTestId("open-library")).toHaveCount(0);
  await expect(sidebar.getByTestId("browse-views")).toHaveCount(0);
  await expect(sidebar.getByTestId("document-list-item")).toHaveCount(0);
  await expect(sidebar.getByTestId("not-in-a-folder-toggle")).toHaveText("Examples");
  await expect(sidebar.getByTestId("mind-list-item")).toHaveCount(1);
  await expect(sidebar.getByTestId("sidebar-status")).toHaveCount(0);
  await expect(window.getByTestId("mind-count-button")).toHaveCount(0);

  await useLocalChatModel(window);
  await window.getByTestId("ask-pill").click();
  await expect(window.getByTestId("assistant")).toBeVisible();
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await expect(window.getByTestId("ask-pill")).toHaveCount(0);
  await app.close();
});

test("the tab strip comes with the second tab, and the Library with the first own Document (#123)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await expect(window.getByTestId("tab-list")).toHaveCount(0);
  await newMind(window);
  await expect(window.getByTestId("tab-list")).toBeVisible();
  await expect(window.getByTestId("tab-title")).toHaveCount(2);

  const sidebar = window.getByTestId("sidebar");
  await expect(sidebar.getByTestId("open-library")).toHaveCount(0);
  await writeFile(join(sources, "Harbour notes.md"), "# Harbour notes\n\nTide tables.\n");
  await window
    .getByTestId("add-documents-input")
    .setInputFiles([join(sources, "Harbour notes.md")]);
  await expect(sidebar.getByTestId("open-library")).toBeVisible();
  await app.close();
});

test("the assistant comes with the first Question, and stays after a restart (#123)", async () => {
  const first = await launchApp(dataDir, { examples: true, fakeChat: true });
  await useLocalChatModel(first.window);
  await first.window.keyboard.press("ControlOrMeta+j");
  await expect(first.window.getByTestId("assistant")).toBeVisible();
  await first.window.keyboard.type("Where does tea come from?");
  await first.window.keyboard.press("Enter");
  await expect(assistantAnswers(first.window).last()).toHaveAttribute("data-status", "done", {
    timeout: 15_000,
  });
  await first.app.close();

  const again = await launchApp(dataDir, { examples: true, fakeChat: true });
  await expect(again.window.getByTestId("assistant")).toBeVisible();
  await expect(again.window.getByTestId("ask-pill")).toHaveCount(0);
  await again.app.close();
});

test("the quiet first open holds at 1,100 px in Chinese (#123)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true });
  await setWindowSize(app, window, 1100, 720);
  await setLanguage(window, "zh-CN");
  const pill = window.getByTestId("ask-pill");
  await expect(pill).toContainText("提问");
  const box = await pill.boundingBox();
  const card = await window.getByTestId("card").boundingBox();
  expect(box && card && box.x >= card.x && box.x + box.width <= card.x + card.width).toBe(true);
  await expect(window.getByTestId("sidebar").getByTestId("not-in-a-folder-toggle")).toHaveText(
    "示例",
  );
  await app.close();
});
