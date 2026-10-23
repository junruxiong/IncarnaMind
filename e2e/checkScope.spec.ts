import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/**
 * A Mind's chosen scope narrows its check (#1623): the Search scope set with "@" in the
 * composer is where the check looks, and every card says where that was; Search all Documents
 * looks everywhere, and the card says so. The scripted model supports a sentence that shares
 * three words with a line it was handed.
 */
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

const OWN = "Harbours publish the times of high water every week.";
const ELSEWHERE = "Penguins nest on the harbour cliffs every spring.";

test("a check looks in the Mind's chosen scope, and its card says where it looked, in English and Chinese", async () => {
  test.setTimeout(120_000);
  await writeFile(join(sources, "Harbour times.txt"), `${OWN}\n`);
  await writeFile(join(sources, "Seabirds.txt"), `${ELSEWHERE}\n`);
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home: sources });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Harbour times.txt"), join(sources, "Seabirds.txt")]);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type(`${OWN} ${ELSEWHERE}`);

  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("@Harbour");
  await expect(window.getByTestId("scope-picker")).toBeVisible();
  await window.keyboard.press("Enter");
  const composer = window.getByTestId("composer");
  await expect(composer.getByTestId("scope-chip")).toHaveText("Harbour times");

  const line = editor.locator(":scope > p").first();
  await slideSelect(window, line, "Harbours publish", { line, words: "spring." });
  await pointAndClick(
    window,
    window.getByTestId("selection-menu").getByTestId("selection-action-find-support"),
  );
  const marks = window.getByTestId("claim-mark");
  await expect(marks).toHaveCount(2);
  await expect(marks.first()).toHaveAttribute("data-check", "supported");
  await expect(marks.nth(1)).toHaveAttribute("data-check", "unsupported");
  await motionSettled(window);
  const card = window.getByTestId("claim-card");

  await pointAndClick(window, marks.first());
  await expect(card.getByTestId("claim-checked-in")).toHaveText("Checked against “Harbour times”");
  await window.keyboard.press("Escape");

  // Not found where it looked: the line that says so names the scope, and only it does.
  await pointAndClick(window, marks.nth(1));
  await expect(card.getByTestId("claim-nothing")).toHaveText(
    "I didn’t find this in “Harbour times”.",
  );
  await expect(card.getByTestId("claim-checked-in")).toHaveCount(0);
  await pointAndClick(window, card.getByTestId("claim-search-all"));
  await expect(marks.nth(1)).toHaveAttribute("data-check", "supported");
  await pointAndClick(window, marks.nth(1));
  await expect(card.getByTestId("claim-checked-in")).toHaveText("Checked against “All Documents”");
  await window.keyboard.press("Escape");

  await setLanguage(window, "zh-CN");
  await pointAndClick(window, marks.first());
  await expect(card.getByTestId("claim-checked-in")).toHaveText("已对照“Harbour times”核对");
  await window.keyboard.press("Escape");
  await pointAndClick(window, marks.nth(1));
  await expect(card.getByTestId("claim-checked-in")).toHaveText("已对照“全部文档”核对");
  await app.close();
});
