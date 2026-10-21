import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  frontMindName,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * New Mind under an Answer (#1559): the Answer as it is, its Citations checked, at once in the
 * empty Mind in front, or else in a new Mind beside it, named by its first line, a name another
 * Mind has taking a number. The scripted model answers from the Document with a Citation. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

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

test("New Mind fills the empty Mind in front, then makes new Minds named by the Answer's first line, numbered apart, its Citation checked and nothing left to accept", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1440, 900);
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  const tabs = window.getByTestId("mind-tab");
  await expect(tabs).toHaveCount(1);

  const answer = await askInAssistant(window, "When do spring tides happen?");
  const button = answer.getByTestId("answer-new-mind");
  await expect(button).toHaveText("New Mind");
  await pointAndClick(window, button);
  await expect(editor.getByTestId("citation").first()).toHaveAttribute("data-check", "found");
  await expect(tabs).toHaveCount(1);
  await expect(window.getByTestId("proposals-bar")).toHaveCount(0);
  // Named by its first line a moment after the Answer is in.
  await expect(frontMindName(window)).not.toHaveText(/^(Untitled|未命名)$/);
  const first = (await frontMindName(window).textContent())?.trim() ?? "";
  await screenshot(window, "new-mind-filled");

  // With text in front, a new Mind beside it, named by the same first line, so numbered.
  await pointAndClick(window, button);
  await expect(tabs).toHaveCount(2);
  await expect(frontMindName(window)).toHaveText(`${first} 2`);
  await expect(editor.getByTestId("citation").first()).toHaveAttribute("data-check", "found");
  await expect(window.getByTestId("proposals-bar")).toHaveCount(0);
  await pointAndClick(window, button);
  await expect(tabs).toHaveCount(3);
  await expect(frontMindName(window)).toHaveText(`${first} 3`);
  await screenshot(window, "new-mind-numbered");

  await setLanguage(window, "zh-CN");
  await expect(button).toHaveText("新建 Mind");
});
