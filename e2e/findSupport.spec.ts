import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  pointAt,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/*
 * Checking the User's own sentences (#96): select sentences, then Find
 * support or Check against Documents in the selection menu. Each sentence
 * gets a mark in the margin, its card says what was found, and the Mind's
 * health under its title counts them. The scripted model (see
 * src/main/fakeChatModel.ts) finds a sentence supported when it shares three
 * words with a line of a Document, contradicted when it also has a number the
 * line doesn't, and otherwise unsupported. Set INCARNAMIND_SCREENSHOTS to a
 * folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
  { lines: ["Harbour times", "Harbours publish the times of high water every 12 months."] },
]);

const SUPPORTED = "Spring tides happen at new and full moon.";
const UNSUPPORTED = "Penguins nest on the harbour cliffs.";
const CONTRADICTED = "Harbours publish the times of high water every 6 months.";
const BLANK = "This line asks the model for a blank reply.";

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

/** Launches with the scripted model and the Tides PDF to check against. */
async function launchWithTides() {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const running = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(running.window);
  return running;
}

/** A new Mind with one paragraph of these sentences, once the Tides PDF is added. */
async function mindWithSentences(window: Page, sentences: readonly string[]) {
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type(sentences.join(" "));
  const line = editor.locator(":scope > p").first();
  await expect(line).toHaveText(sentences.join(" "));
  return { editor, line };
}

/** Selects from the line's first words to its last, with the mouse, and chooses an action in the menu. */
async function checkAll(window: Page, line: Locator, first: string, last: string, action: string) {
  await slideSelect(window, line, first, { line, words: last });
  const menu = window.getByTestId("selection-menu");
  await expect(menu.getByTestId(`selection-action-${action}`)).toBeVisible();
  await pointAndClick(window, menu.getByTestId(`selection-action-${action}`));
}

const marks = (window: Page) => window.getByTestId("claim-mark");

test("Find support on two sentences: one is supported and its Citation is added, the other is marked unsupported", async () => {
  const { app, window } = await launchWithTides();
  await useLocalChatModel(window);
  const { editor, line } = await mindWithSentences(window, [SUPPORTED, UNSUPPORTED]);
  await setWindowSize(app, window, 1440, 900);
  const health = window.getByTestId("mind-health");
  await expect(health.getByTestId("health-noCitation")).toHaveText("2 with no Citation");

  await checkAll(window, line, "Spring tides", "harbour cliffs.", "find-support");

  await expect(marks(window)).toHaveCount(2);
  const supported = marks(window).and(window.locator('[data-check="supported"]'));
  const unsupported = marks(window).and(window.locator('[data-check="unsupported"]'));
  await expect(supported).toHaveAttribute("data-addable", "true");
  await expect(supported).toHaveAccessibleName("Sentence check: Supported");
  await expect(unsupported).toHaveAccessibleName("Sentence check: Unsupported");
  // Each mark sits in the margin, right of the text, the first above the second.
  const lineBox = await line.boundingBox();
  const supportedBox = await supported.boundingBox();
  const unsupportedBox = await unsupported.boundingBox();
  expect(supportedBox?.x ?? 0).toBeGreaterThan((lineBox?.x ?? 0) + (lineBox?.width ?? 0));
  expect(unsupportedBox?.y ?? 0).toBeGreaterThan(supportedBox?.y ?? 0);
  // The unsupported sentence is washed; the health line counts both.
  await expect(line.locator(".claim-wash--unsupported")).toHaveText(UNSUPPORTED);
  await expect(health.getByTestId("health-unsupported")).toHaveText("1 unsupported");
  await expect(health.getByTestId("health-noCitation")).toHaveText("1 with no Citation");
  await screenshot(window, "check-results-en");

  // The unsupported sentence's card says where it looked.
  await pointAndClick(window, unsupported);
  const card = window.getByTestId("claim-card");
  await expect(card.getByTestId("claim-badge")).toHaveText("Unsupported");
  await expect(card.getByTestId("claim-nothing")).toHaveText(
    "Nothing in your Documents says this.",
  );
  await window.keyboard.press("Escape");
  await expect(card).toBeHidden();

  // The supported one shows the Passage found, and adds it as the sentence's Citation in one click.
  await pointAt(window, supported);
  await expect(card.getByTestId("claim-badge")).toHaveText("Supported");
  await expect(card.getByTestId("claim-quote")).toHaveText(
    "Spring tides happen at new moon and at full moon.",
  );
  await screenshot(window, "check-supported-card-en");
  await pointAndClick(window, card.getByTestId("claim-add-citation"));

  const citation = line.getByTestId("citation");
  await expect(citation).toHaveAttribute("data-check", "found");
  await expect(marks(window)).toHaveCount(1);
  // ⌘Z takes it back in one step: the check's mark returns; ⌘⇧Z adds the Citation again.
  await window.keyboard.press("ControlOrMeta+z");
  await expect(citation).toHaveCount(0);
  await expect(supported).toBeVisible();
  await window.keyboard.press("ControlOrMeta+Shift+z");
  await expect(citation).toHaveAttribute("data-check", "found");
  await expect(line).toHaveText(`${SUPPORTED.slice(0, -1)} 1. ${UNSUPPORTED}`);
  await expect(marks(window)).toHaveCount(1);
  await expect(health.getByTestId("health-found")).toHaveText("1 Citation found");
  await expect(health.getByTestId("health-noCitation")).toHaveCount(0);

  // Editing the unsupported sentence makes its check need a re-check; Check again checks it.
  await slideSelect(window, line, "Penguins");
  await window.keyboard.type("Gulls");
  await expect(marks(window)).toHaveAttribute("data-check", "needs-recheck");
  await expect(health.getByTestId("health-recheck")).toHaveText("1 needs re-check");
  await screenshot(window, "check-recheck-en");
  await pointAndClick(window, marks(window));
  await expect(card.getByTestId("claim-reason")).toHaveText(
    "The sentence has changed since it was checked.",
  );
  await pointAndClick(window, card.getByTestId("claim-check-again"));
  await expect(marks(window)).toHaveAttribute("data-check", "unsupported");
  await expect(health.getByTestId("health-unsupported")).toHaveText("1 unsupported");
  await expect(editor).toContainText("Gulls nest on the harbour cliffs.");
});

test("In Chinese and a narrow window: Check against Documents marks a contradiction and a sentence it couldn't check", async () => {
  const { app, window } = await launchWithTides();
  await useLocalChatModel(window);
  const { line } = await mindWithSentences(window, [CONTRADICTED, BLANK]);
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 900, 720);
  const health = window.getByTestId("mind-health");
  await expect(health.getByTestId("health-noCitation")).toHaveText("2 句没有引用");

  await slideSelect(window, line, "Harbours publish", { line, words: "blank reply." });
  const menu = window.getByTestId("selection-menu");
  await expect(menu.getByTestId("selection-action-find-support")).toHaveText("查找出处");
  await expect(menu.getByTestId("selection-action-check-against-documents")).toHaveText(
    "对照文档核对",
  );
  await screenshot(window, "check-menu-zh-narrow");
  await pointAndClick(window, menu.getByTestId("selection-action-check-against-documents"));

  const contradicted = marks(window).and(window.locator('[data-check="contradicted"]'));
  const notChecked = marks(window).and(window.locator('[data-check="not-checked"]'));
  await expect(contradicted).toBeVisible();
  await expect(notChecked).toBeVisible();
  // In a narrow Mind the marks keep to the gutter right of the text.
  const lineBox = await line.boundingBox();
  const markBox = await contradicted.boundingBox();
  expect(markBox?.x ?? 0).toBeGreaterThanOrEqual((lineBox?.x ?? 0) + (lineBox?.width ?? 0));
  await expect(line.locator(".claim-wash--contradicted")).toHaveText(CONTRADICTED);
  await expect(health.getByTestId("health-contradicted")).toHaveText("1 句与文档矛盾");
  await expect(health.getByTestId("health-notChecked")).toHaveText("1 句未核对");
  await screenshot(window, "check-results-zh-narrow");

  await pointAndClick(window, contradicted);
  const card = window.getByTestId("claim-card");
  await expect(card.getByTestId("claim-badge")).toHaveText("与文档矛盾");
  await expect(card.getByTestId("claim-says")).toContainText(
    "Harbours publish the times of high water every 12 months.",
  );
  await expect(card.getByTestId("claim-quote")).toHaveText(
    "Harbours publish the times of high water every 12 months.",
  );
  await screenshot(window, "check-contradicted-card-zh-narrow");
  await window.keyboard.press("Escape");

  await pointAndClick(window, notChecked);
  await expect(card.getByTestId("claim-reason")).toHaveText("模型的回复是空的。");
  await expect(card.getByTestId("claim-check-again")).toHaveText("重新核对");

  // Each part of the health line moves to its sentence.
  await window.keyboard.press("Escape");
  await pointAndClick(window, health.getByTestId("health-notChecked"));
  await expect(window.getByTestId("mind-editor")).toBeFocused();
});

test("A supported sentence's card names the Document as it is named now, after a rename in IncarnaMind", async () => {
  const { app, window } = await launchWithTides();
  await useLocalChatModel(window);
  const { line } = await mindWithSentences(window, [SUPPORTED]);
  await setWindowSize(app, window, 1440, 900);
  await checkAll(window, line, "Spring tides", "full moon.", "find-support");
  const supported = marks(window).and(window.locator('[data-check="supported"]'));
  const card = window.getByTestId("claim-card");
  await pointAt(window, supported);
  await expect(card.getByTestId("claim-source")).toHaveText(/^Tides · /);

  // Open goes to the Document, whose header renames it.
  await pointAndClick(window, card.getByTestId("claim-open"));
  const viewerTitle = window.getByTestId("viewer-title");
  await expect(viewerTitle).toHaveText("Tides");
  await viewerTitle.dblclick();
  const field = window.getByTestId("viewer-title-rename");
  await expect(field).toBeFocused();
  await field.fill("Tides of the coast");
  await window.keyboard.press("Enter");
  await expect(viewerTitle).toHaveText("Tides of the coast");

  await pointAt(window, supported);
  await expect(card.getByTestId("claim-source")).toHaveText(/^Tides of the coast · /);
});
