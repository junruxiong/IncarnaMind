import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * An Answer whose model writes a Citation marker and never calls `cite` shows
 * no marker: its words stay, and nothing opens a Citation that isn't there;
 * in English and Chinese.
 */

const TIDES = buildPdf([
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

let dataDir: string;
let sources: string;
let home: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
  home = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
  await removeDataFolder(home);
});

test("a marker written without a cite call is not shown, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await newMind(window);
  await window.getByTestId("mind-editor").click();

  const english = await askInAssistant(window, "When do spring tides happen, without citing?");
  const englishText = english.getByTestId("assistant-answer-text");
  await expect(englishText).toContainText("Your Documents answer this");
  await expect(englishText).not.toContainText("[^");
  await expect(english.getByTestId("citation-chip")).toHaveCount(0);

  await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  const chinese = await askInAssistant(window, "大潮什么时候出现，不引用？");
  const chineseText = chinese.getByTestId("assistant-answer-text");
  await expect(chineseText).toContainText("文档回答了这个问题");
  await expect(chineseText).not.toContainText("[^");
  await expect(chinese.getByTestId("citation-chip")).toHaveCount(0);
});
