import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { wordFile, paragraph as wordParagraph } from "../tests/helpers/word";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  importWordFile,
  launchApp,
  motionSettled,
  newMind,
  openDocumentMenu,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/**
 * A Mind's chosen scope narrows its check (#1623): the Search scope set in the composer is
 * where the check looks, every card says where that was, a change of the scope marks the
 * checks made before it for a re-check, and one click widens the scope back to every
 * Document. The scripted model supports a sentence that shares three words with a line it was
 * handed. Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
let shots = 0;
const screenshot = async (window: Page, name: string) => {
  shots++;
  if (!SCREENSHOTS) return;
  await motionSettled(window);
  const file = `${String(shots).padStart(2, "0")}-${name}.png`;
  await window.screenshot({ path: join(SCREENSHOTS, file) });
};

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

async function launched() {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home: sources });
  await setWindowSize(app, window, 900, 800);
  await dismissChatSetup(window);
  return { app, window };
}

async function twoDocumentsAndAMind(window: Page) {
  await writeFile(join(sources, "Harbour times.txt"), `${OWN}\n`);
  await writeFile(join(sources, "Seabirds.txt"), `${ELSEWHERE}\n`);
  await addDocuments(window, [join(sources, "Harbour times.txt"), join(sources, "Seabirds.txt")]);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type(`${OWN} ${ELSEWHERE}`);
  return editor;
}

test("a check looks in the Mind's chosen scope, its card says where, and a change of the scope marks it for a re-check", async () => {
  test.setTimeout(120_000);
  const { app, window } = await launched();
  await useLocalChatModel(window);
  const editor = await twoDocumentsAndAMind(window);

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
  await screenshot(window, "checked-in-en");
  await window.keyboard.press("Escape");

  // Not found where it looked: the line that says so names the scope, and only it does.
  await pointAndClick(window, marks.nth(1));
  await expect(card.getByTestId("claim-nothing")).toHaveText(
    "I didn’t find this in “Harbour times”.",
  );
  await expect(card.getByTestId("claim-checked-in")).toHaveCount(0);
  await screenshot(window, "not-found-in-scope-en");
  await pointAndClick(window, card.getByTestId("claim-search-all"));
  await expect(marks.nth(1)).toHaveAttribute("data-check", "supported");
  await pointAndClick(window, marks.nth(1));
  await expect(card.getByTestId("claim-checked-in")).toHaveText("Checked against “All Documents”");
  await screenshot(window, "widened-en");
  await window.keyboard.press("Escape");

  await setLanguage(window, "zh-CN");
  await pointAndClick(window, marks.first());
  await expect(card.getByTestId("claim-checked-in")).toHaveText("已对照“Harbour times”核对");
  await screenshot(window, "checked-in-zh");
  await window.keyboard.press("Escape");
  await pointAndClick(window, marks.nth(1));
  await expect(card.getByTestId("claim-checked-in")).toHaveText("已对照“全部文档”核对");
  await screenshot(window, "widened-zh");
  await window.keyboard.press("Escape");

  // Widened in the composer: the check made in the one Document needs a re-check; the one made
  // in every Document still holds.
  await pointAndClick(window, composer.getByTestId("scope-chip-remove"));
  await expect(composer.getByTestId("composer-scope-all")).toBeVisible();
  await expect(marks.first()).toHaveAttribute("data-check", "needs-recheck");
  await expect(marks.nth(1)).toHaveAttribute("data-check", "supported");
  await pointAndClick(window, marks.first());
  await expect(card.getByTestId("claim-reason")).toHaveText("这个 Mind 的搜索范围已更改。");
  await screenshot(window, "recheck-zh");
  await app.close();
});

test("“Ask about this Document” sets the Mind's scope, and one click widens it back to every Document, by mouse or keyboard, in English and Chinese", async () => {
  test.setTimeout(120_000);
  const { app, window } = await launched();
  await useLocalChatModel(window);
  await twoDocumentsAndAMind(window);
  const composer = window.getByTestId("composer");
  const chip = composer.getByTestId("scope-chip");
  const widen = composer.getByTestId("scope-chip-remove");
  const askAbout = async () => {
    const row = window.getByTestId("document-list-item").filter({ hasText: "Harbour times" });
    await openDocumentMenu(row);
    await pointAndClick(window, row.getByTestId("document.ask"));
    await expect(chip).toHaveText("Harbour times");
  };

  await askAbout();
  await expect(widen).toHaveAccessibleName("Search all Documents, not only “Harbour times”");
  await screenshot(window, "asked-about-en");
  await pointAndClick(window, widen);
  await expect(chip).toHaveCount(0);
  await expect(composer.getByTestId("composer-scope-all")).toHaveText("All Documents");
  await expect(composer.getByTestId("composer-input")).toBeFocused();

  await askAbout();
  await widen.focus();
  await window.keyboard.press("Enter");
  await expect(chip).toHaveCount(0);
  await expect(composer.getByTestId("composer-input")).toBeFocused();

  await setLanguage(window, "zh-CN");
  await askAbout();
  await expect(widen).toHaveAccessibleName("改为搜索全部文档，不只搜索“Harbour times”");
  await screenshot(window, "asked-about-zh");
  await widen.focus();
  await window.keyboard.press("Space");
  await expect(composer.getByTestId("composer-scope-all")).toHaveText("全部文档");
  await app.close();
});

test("a Mind imported from a Word file isn't supported by its own file, and the card says the copy was left out, in English and Chinese", async () => {
  test.setTimeout(120_000);
  const { app, window } = await launched();
  await useLocalChatModel(window);
  // Something else to check against: the imported file alone is left out of its own check.
  await writeFile(join(sources, "Seabirds.txt"), `${ELSEWHERE}\n`);
  await addDocuments(window, [join(sources, "Seabirds.txt")]);
  await importWordFile(
    app,
    window,
    sources,
    "Draft memo.docx",
    wordFile({ body: wordParagraph(OWN) }),
  );
  const item = window.getByTestId("document-list-item").filter({ hasText: "Draft memo" });
  await expect(item).toHaveAttribute("data-status", "ready");
  const editor = window.getByTestId("mind-editor");
  const line = editor.locator("p", { hasText: "Harbours publish" });
  await slideSelect(window, line, "Harbours publish", { line, words: "week." });
  await pointAndClick(
    window,
    window.getByTestId("selection-menu").getByTestId("selection-action-find-support"),
  );
  const mark = window.getByTestId("claim-mark");
  await expect(mark).toHaveAttribute("data-check", "unsupported");
  await motionSettled(window);
  await pointAndClick(window, mark);
  const card = window.getByTestId("claim-card");
  await expect(card.getByTestId("claim-left-out")).toHaveText(
    "Also in “Draft memo”, a copy of the checked document: not counted",
  );
  await screenshot(window, "left-out-en");
  await window.keyboard.press("Escape");

  await setLanguage(window, "zh-CN");
  await pointAndClick(window, mark);
  await expect(card.getByTestId("claim-left-out")).toHaveText(
    "也出现在“Draft memo”（正在核对的文档的副本）中，未计为依据",
  );
  await screenshot(window, "left-out-zh");
  await app.close();
});
