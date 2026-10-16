import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  checkCard,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * The check badge (#909): at a Mind's bottom right, the margin marks' counts
 * and the claims no check has looked at; nothing on a Mind with nothing to
 * check. Its card opens from it, and from ⌘K. The scripted model
 * (src/main/fakeChat/checks.ts) finds a sentence supported when it shares
 * three words with a line of a Document, and otherwise unsupported. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (!SCREENSHOTS) return;
  await motionSettled(window);
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const TIDES = buildPdf([
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);
const SUPPORTED = "Spring tides happen at new and full moon.";
const UNSUPPORTED = "Penguins nest on the harbour cliffs.";

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

async function launchWithTides() {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const running = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(running.window);
  await connectLocalChatModel(running.window);
  await addDocuments(running.window, [join(sources, "Tides.pdf")]);
  return running;
}

test("a Mind with nothing to check is bare; a claim no check has looked at shows the badge, whose card checks it, says so until you type, and closes with Esc", async () => {
  const { app, window } = await launchWithTides();
  await setWindowSize(app, window, 1440, 900);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("Thanks!");
  const badge = window.getByTestId("check-badge");
  const open = window.getByTestId("check-badge-open");
  await expect(editor).toContainText("Thanks!");
  await expect(badge).toHaveCount(0);

  await window.keyboard.type(` ${SUPPORTED}`);
  await expect(open).toHaveAccessibleName("Sentences and Citations: 1 claim not checked yet");
  await expect(badge).toContainText("to check");
  // It sits at the page's bottom right, over the text, never past the Mind's pane.
  const pane = await window.getByTestId("mind-pane").boundingBox();
  const box = await badge.boundingBox();
  if (!pane || !box) throw new Error("The badge isn't drawn.");
  expect(box.x + box.width).toBeLessThanOrEqual(pane.x + pane.width + 1);
  await screenshot(window, "check-badge-to-check-en");

  const card = await checkCard(window);
  await expect(card.getByTestId("health-unchecked")).toContainText("1 claim not checked yet");
  await screenshot(window, "check-badge-card-en");
  await pointAndClick(window, card.getByTestId("health-check"));
  // A check the User started: the badge says what it checked, until they type.
  await expect(window.getByTestId("claim-mark")).toHaveAttribute("data-check", "supported");
  await expect(badge).toContainText("1 checked");
  await window.keyboard.press("Escape");
  await expect(card).toHaveCount(0);
  await editor.click();
  await window.keyboard.press("End");
  await window.keyboard.type(" More.");
  await expect(badge).toHaveCount(0);

  // ⌘K reaches the card while the badge is hidden; Esc from inside it puts the focus back.
  await window.keyboard.press("ControlOrMeta+k");
  await window.keyboard.type("Sentences and Citations");
  await pointAndClick(
    window,
    window.getByTestId("palette-item").filter({ hasText: "Sentences and Citations" }).first(),
  );
  await expect(card).toBeVisible();
  await expect(card.getByTestId("health-found")).toHaveCount(0);
  await expect(card.getByTestId("badge-check-mind")).toBeVisible();
  await window.keyboard.press("Escape");
  await expect(card).toHaveCount(0);
  await app.close();
});

test("in Chinese and a narrow pane the badge shows only its marks and numbers; Tab reaches its card and Esc brings the focus back", async () => {
  const { app, window } = await launchWithTides();
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 1440, 900);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type(`${SUPPORTED} ${UNSUPPORTED}`);
  const open = window.getByTestId("check-badge-open");
  await expect(open).toHaveAccessibleName("句子和引用：2 句待核对");
  const words = window.getByTestId("check-badge").locator(".check-badge-words");
  await expect(words).toHaveText("句待核对");
  await expect(words).toBeVisible();
  // The Mind beside itself in a narrower window: each page under 400 px keeps the numbers, not the words.
  await pointAndClick(window, window.getByTestId("split-group"));
  const pages = window.locator(".mind-scroller:visible");
  await expect(pages).toHaveCount(2);
  await setWindowSize(app, window, 1340, 900);
  for (const width of await pages.evaluateAll((all) => all.map((page) => page.clientWidth))) {
    expect(width).toBeLessThan(400);
  }
  await expect(words).toHaveCount(2);
  for (const each of await words.all()) await expect(each).toBeHidden();
  await expect(window.getByTestId("check-badge").first()).toContainText("2");
  await screenshot(window, "check-badge-zh-narrow");

  await open.first().focus();
  await window.keyboard.press("Enter");
  // Only the badge's own pane opens its card.
  const card = window.getByTestId("check-card");
  await expect(card).toHaveCount(1);
  await expect(card).toBeVisible();
  await motionSettled(window);
  await window.keyboard.press("Tab");
  await expect(card.locator(":focus")).toHaveCount(1);
  await window.keyboard.press("Escape");
  await expect(card).toHaveCount(0);
  await expect(open.first()).toBeFocused();
  await app.close();
});
