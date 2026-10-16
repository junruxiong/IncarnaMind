import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/**
 * A check's result stays as it was (#819): never marked "needs re-check" by a
 * change of interface language or a relaunch, only by a change to the
 * sentence, a Document it relied on or the Mind's Folder (#96). An example's
 * Documents, which a language change would swap, stay while a check relies on them.
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

const TIDES = buildPdf([
  { lines: ["Harbour times", "Harbours publish the times of high water every 12 months."] },
]);
const WRONG = "Harbours publish the times of high water every 6 months.";
const QUOTE = "Harbours publish the times of high water every 12 months.";

test("a Contradicted result keeps its mark and card after a change of language and a relaunch; editing the sentence makes it need a re-check", async () => {
  test.setTimeout(150_000);
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const first = await launchApp(dataDir, { fakeChat: true, home: sources });
  await setWindowSize(first.app, first.window, 1440, 900);
  await dismissChatSetup(first.window);
  await useLocalChatModel(first.window);
  await addDocuments(first.window, [join(sources, "Tides.pdf")]);
  await newMind(first.window);
  await first.window.getByTestId("mind-title").fill("Memo");
  const editor = first.window.getByTestId("mind-editor");
  await editor.click();
  await first.window.keyboard.type(WRONG);
  const line = editor.locator(":scope > p").first();
  await slideSelect(first.window, line, "Harbours", { line, words: "months." });
  await pointAndClick(
    first.window,
    first.window
      .getByTestId("selection-menu")
      .getByTestId("selection-action-check-against-documents"),
  );
  const mark = first.window.getByTestId("claim-mark");
  await expect(mark).toHaveAttribute("data-check", "contradicted");

  // The language changes: the same mark, and its card still says the same.
  await setLanguage(first.window, "zh-CN");
  await expect(mark).toHaveAttribute("data-check", "contradicted");
  await pointAndClick(first.window, mark);
  await expect(first.window.getByTestId("claim-card").getByTestId("claim-quote")).toHaveText(QUOTE);
  await first.window.keyboard.press("Escape");
  await setLanguage(first.window, "en");
  await expect(mark).toHaveAttribute("data-check", "contradicted");
  await first.app.close();

  // Relaunched: the same.
  const again = await launchApp(dataDir, { fakeChat: true, home: sources });
  await setWindowSize(again.app, again.window, 1440, 900);
  await again.window.getByTestId("mind-list-item").filter({ hasText: "Memo" }).click();
  const markAgain = again.window.getByTestId("claim-mark");
  await expect(markAgain).toHaveAttribute("data-check", "contradicted");
  await pointAndClick(again.window, markAgain);
  await expect(again.window.getByTestId("claim-card").getByTestId("claim-quote")).toHaveText(QUOTE);
  await again.window.keyboard.press("Escape");

  // Editing the sentence is what makes it need a re-check.
  const lineAgain = again.window.getByTestId("mind-editor").locator(":scope > p").first();
  await slideSelect(again.window, lineAgain, "6");
  await again.window.keyboard.type("7");
  await expect(markAgain).toHaveAttribute("data-check", "needs-recheck");
  await again.app.close();
});

test("an example's Documents a check relies on stay when the language changes, and so does the check", async () => {
  test.setTimeout(150_000);
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    examples: true,
    home: sources,
  });
  await setWindowSize(app, window, 1440, 900);
  await expect(window.getByTestId("document-list-item")).toHaveCount(2);
  await useLocalChatModel(window);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type(
    "Net debt to EBITDA was 3.1 times at the end of the quarter, up from 2.7 times at the half year.",
  );
  const line = editor.locator(":scope > p").first();
  await slideSelect(window, line, "Net debt", { line, words: "year." });
  await pointAndClick(
    window,
    window.getByTestId("selection-menu").getByTestId("selection-action-check-against-documents"),
  );
  const mark = window.getByTestId("claim-mark");
  await expect(mark).toHaveAttribute("data-check", "supported");

  await setLanguage(window, "zh-CN");
  await expect(mark).toHaveAttribute("data-check", "supported");
  await setLanguage(window, "en");
  await expect(mark).toHaveAttribute("data-check", "supported");
  await app.close();
});
