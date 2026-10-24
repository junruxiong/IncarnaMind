import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  assistantAnswers,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  expectReadyDocuments,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
} from "./app";

/**
 * "Ask about this" (#101): words selected in a Document go into the composer as a quote chip,
 * from the selection menu, ⌘J or ⌘K; the Question keeps them above its own words, and the chip
 * goes with × or Backspace.
 */

const REPEATED = "Margins recover within two quarters.";
const PAPER = buildPdf([
  { lines: ["Grocers studied across the eleven chains.", REPEATED] },
  { lines: ["Pooling the cuts, the recovery peaks in the second quarter.", REPEATED] },
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

/** The app with the scripted model, the paper beside a new Mind, and a page's text layer. */
async function paperBesideMind() {
  const path = join(sources, "Rate cuts.pdf");
  await writeFile(path, PAPER);
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await connectLocalChatModel(window);
  await addDocuments(window, [path]);
  await expectReadyDocuments(window, 1);
  await newMind(window);
  await window.keyboard.type("Margins");
  await pointAndClick(
    window,
    window.getByTestId("document-list-item").getByTestId("open-document"),
  );
  const layer = (page: number) =>
    window.getByTestId("viewer").locator(`[data-page-number="${page}"] .textLayer`);
  await expect(layer(2)).toContainText(REPEATED);
  return { window, layer };
}

async function select(window: Page, layer: ReturnType<Page["locator"]>, words: string) {
  const all = words.split(" ");
  await slideSelect(window, layer, all.slice(0, 2).join(" "), {
    line: layer,
    words: all.slice(-2).join(" "),
  });
  await expect(window.getByTestId("viewer-selection-menu")).toBeVisible();
}

const chips = (window: Page) => window.getByTestId("composer-quote");

test("Ask about this puts the words in the composer, and the Question keeps them above its own", async () => {
  const { window, layer } = await paperBesideMind();
  // The same words are on page 1 too: the chip must say page 2.
  await select(window, layer(2), "Margins recover within two quarters");

  await pointAndClick(window, window.getByTestId("viewer-selection-ask"));

  await expect(chips(window)).toHaveCount(1);
  await expect(chips(window)).toContainText("“Margins recover within two quarters”");
  await expect(chips(window)).toContainText("Rate cuts p. 2");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("What does this mean?");
  await window.keyboard.press("Enter");

  const question = window.getByTestId("assistant").getByTestId("assistant-question").last();
  await expect(question).toContainText("What does this mean?");
  await expect(question.getByTestId("assistant-question-quote")).toContainText(
    "Margins recover within two quarters",
  );
  await expect(question.getByTestId("assistant-question-quote-source")).toHaveText(
    "Rate cuts p. 2",
  );
  await expect(chips(window)).toHaveCount(0);
  await expect(assistantAnswers(window).last()).toHaveAttribute("data-status", "done", {
    timeout: 45_000,
  });
});

test("⌘J and ⌘K ask about the selection, and Backspace and × take the chip away, in Chinese", async () => {
  const { window, layer } = await paperBesideMind();
  await setLanguage(window, "zh-CN");
  await select(window, layer(1), "Grocers studied across the eleven chains.");
  await expect(window.getByTestId("viewer-selection-ask")).toContainText("就此提问");

  await window.keyboard.press("ControlOrMeta+j");
  await expect(chips(window)).toHaveCount(1);
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.press("Backspace");
  await expect(chips(window)).toHaveCount(0);

  await select(window, layer(1), "Grocers studied across the eleven chains.");
  await window.keyboard.press("ControlOrMeta+k");
  await window.keyboard.type("就所选文字提问");
  await window.keyboard.press("Enter");
  await expect(chips(window)).toHaveCount(1);
  await expect(chips(window)).toContainText("第 1 页");
  await pointAndClick(window, window.getByTestId("composer-quote-remove"));
  await expect(chips(window)).toHaveCount(0);
});
