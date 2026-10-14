import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/*
 * Verify (#97): "Verify a document…" in the + menu takes a PDF someone else
 * wrote. It opens beside the Mind made of its text, whose claims are found;
 * the line under the title checks them, and each gets a mark in the margin.
 * The scripted model (src/main/fakeChatModel.ts) takes every sentence but
 * thanks for a claim, finds one supported when it shares three words with a
 * line of a Document, contradicted when it also has a number the line
 * doesn't, unsupported otherwise, and refuses a sentence with "refuse". Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const TIDES = buildPdf([
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
  { lines: ["Harbour times", "Harbours publish the times of high water every 12 months."] },
]);

const SUPPORTED = "Spring tides happen at new and full moon.";
const CONTRADICTED = "Harbours publish the times of high water every 6 months.";
const UNSUPPORTED = "Penguins nest on the harbour cliffs.";
const REFUSED = "The model should refuse to check this line.";
const THANKS = "Thank you for reading this report.";

/** The report someone else wrote: a title, four claims over two pages, and thanks. */
const REPORT = buildPdf([
  { lines: ["Harbour report", SUPPORTED, CONTRADICTED] },
  { lines: [UNSUPPORTED, REFUSED, THANKS] },
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

/** Launches with the scripted model, and the report and the Tides PDF on disk. */
async function launch() {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await writeFile(join(sources, "Harbour report.pdf"), REPORT);
  const running = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(running.window);
  return running;
}

/** Adds the Tides PDF, if asked, and has the open dialog pick the report. */
async function readyToVerify({ app, window }: Awaited<ReturnType<typeof launch>>, tides = true) {
  if (tides) await addDocuments(window, [join(sources, "Tides.pdf")]);
  await interceptOpenDialog(app, join(sources, "Harbour report.pdf"));
}

/** "Verify a document…" from the + menu. */
async function verifyFromPlusMenu(window: Page, label: string) {
  await pointAndClick(window, window.getByTestId("plus-menu"));
  const item = window.getByTestId("plus-verify-document");
  await expect(item).toHaveText(label);
  await pointAndClick(window, item);
}

const marks = (window: Page) => window.getByTestId("claim-mark");

test("Verify a PDF: its Mind opens beside it, its claims are checked, and the counts say how it went", async () => {
  const running = await launch();
  const { app, window } = running;
  await useLocalChatModel(window);
  await readyToVerify(running);
  await setWindowSize(app, window, 1440, 900);

  await verifyFromPlusMenu(window, "Verify a document…");

  const line = window.getByTestId("verify-line");
  await expect(window.getByTestId("mind-title")).toHaveValue("Checking: Harbour report");
  await expect(line.getByTestId("verify-from")).toHaveText("From Harbour report");
  // The PDF is open beside the Mind, on the right, as a tab of its own.
  const viewer = window.getByTestId("viewer");
  await expect(viewer).toBeVisible();
  const editor = window.getByTestId("mind-editor");
  expect((await viewer.boundingBox())?.x ?? 0).toBeGreaterThan(
    (await editor.boundingBox())?.x ?? 0,
  );
  const check = line.getByTestId("verify-check");
  await expect(check).toHaveText("Check 4 claims");
  await expect(line).toContainText("Check it against your Documents?");
  // A model on this computer costs nothing: no estimate.
  await expect(line.getByTestId("verify-estimate")).toHaveCount(0);
  await screenshot(window, "verify-start-en");

  await pointAndClick(window, check);

  const done = line.getByTestId("verify-done");
  await expect(done).toContainText("Checked 4 claims");
  await expect(done.getByTestId("verify-need-look")).toHaveText("3 need a look");
  await expect(marks(window)).toHaveCount(4);
  await expect(marks(window).and(window.locator('[data-check="supported"]'))).toHaveCount(1);
  await expect(marks(window).and(window.locator('[data-check="contradicted"]'))).toHaveCount(1);
  await expect(marks(window).and(window.locator('[data-check="unsupported"]'))).toHaveCount(1);
  const notChecked = marks(window).and(window.locator('[data-check="not-checked"]'));
  await expect(notChecked).toHaveAccessibleName("Sentence check: Not checked");
  await screenshot(window, "verify-results-en");

  // The claim the model failed on says why, with Check again.
  await pointAndClick(window, notChecked);
  const card = window.getByTestId("claim-card");
  await expect(card.getByTestId("claim-reason")).toHaveText("The model declined to check it.");
  await window.keyboard.press("Escape");

  // "need a look" goes to the next such claim.
  await pointAndClick(window, done.getByTestId("verify-need-look"));
  await expect(editor).toBeFocused();

  // Editing a claim makes its check need a re-check, as Find support's do.
  const first = editor.locator(":scope > p").nth(1);
  await slideSelect(window, first, "Spring");
  await window.keyboard.type("Some spring");
  await expect(marks(window).and(window.locator('[data-check="needs-recheck"]'))).toHaveCount(1);
  await expect(done.getByTestId("verify-need-look")).toHaveText("4 need a look");
});

test("In Chinese and a narrow window: the same, in Chinese", async () => {
  const running = await launch();
  const { app, window } = running;
  await useLocalChatModel(window);
  await readyToVerify(running);
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 900, 720);

  await verifyFromPlusMenu(window, "核对文档…");

  const line = window.getByTestId("verify-line");
  await expect(line.getByTestId("verify-from")).toHaveText("来自 Harbour report");
  await expect(line).toContainText("要对照你的文档核对吗？");
  await expect(line.getByTestId("verify-check")).toHaveText("核对 4 条说法");
  // The line keeps within the Mind's column.
  const lineBox = await line.boundingBox();
  const editorBox = await window.getByTestId("mind-editor").boundingBox();
  expect((lineBox?.x ?? 0) + (lineBox?.width ?? 0)).toBeLessThanOrEqual(
    (editorBox?.x ?? 0) + (editorBox?.width ?? 0) + 1,
  );
  await screenshot(window, "verify-start-zh-narrow");

  await pointAndClick(window, line.getByTestId("verify-check"));
  const done = line.getByTestId("verify-done");
  await expect(done).toContainText("已核对 4 条说法");
  await expect(done.getByTestId("verify-need-look")).toHaveText("3 条需要留意");
  await expect(marks(window)).toHaveCount(4);
  await screenshot(window, "verify-results-zh-narrow");
});

test("Empty states: nothing to check against, then no claims found", async () => {
  const running = await launch();
  const { app, window } = running;
  await useLocalChatModel(window);
  await readyToVerify(running, false);
  await setWindowSize(app, window, 1440, 900);

  await verifyFromPlusMenu(window, "Verify a document…");
  const line = window.getByTestId("verify-line");
  // The PDF is the only Document, and it is never its own evidence.
  await expect(line.getByTestId("verify-empty")).toHaveText(
    "Nothing to check against: add Documents first",
  );
  await expect(line.getByTestId("verify-add-files")).toHaveText("Add files…");

  const thanks = buildPdf([{ lines: ["Thank you all for coming.", "Thanks again for the help."] }]);
  await writeFile(join(sources, "Thanks.pdf"), thanks);
  // The report is listed already: the Tides PDF joins it.
  await window.getByTestId("add-documents-input").setInputFiles([join(sources, "Tides.pdf")]);
  const tides = window.getByTestId("document-list-item").filter({ hasText: "Tides" });
  await expect(tides).toHaveAttribute("data-status", "ready");
  await interceptOpenDialog(app, join(sources, "Thanks.pdf"));
  await verifyFromPlusMenu(window, "Verify a document…");
  // The first Mind's tab stays open behind it.
  await expect(window.locator('[data-testid="mind-title"]:visible')).toHaveValue(
    "Checking: Thanks",
  );
  await expect(
    window.locator('[data-testid="verify-line"]:visible').getByTestId("verify-empty"),
  ).toHaveText("No claims found");
});

test("a provider failure while finding the claims is worded like an Answer's, with its next step (#764)", async () => {
  const running = await launch();
  const { app, window } = running;
  // The scripted model fails as one its provider doesn't know.
  await useLocalChatModel(window, "ghost-claims-not-found");
  await readyToVerify(running);
  await setWindowSize(app, window, 1440, 900);

  await verifyFromPlusMenu(window, "Verify a document…");

  const failure = window.getByTestId("answer-error");
  await expect(failure).toHaveAttribute("data-error-kind", "model");
  await expect(failure).toContainText("doesn't know");
  await expect(failure.getByTestId("answer-error-action")).toBeVisible();
  await expect(window.getByTestId("verify-problem")).toHaveCount(0);
  await app.close();
});
