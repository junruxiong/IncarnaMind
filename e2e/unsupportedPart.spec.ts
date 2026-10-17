import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  clickEndOf,
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
 * A cited sentence's Unsupported card says which part its Citation doesn't
 * cover, and never "Nothing in your Documents says this", which a sentence with
 * no Citation keeps. The scripted model (src/main/fakeChat/checks.ts) finds a
 * cited sentence unsupported and names its last clause when the Passage shares
 * none of its words, and gives no reason when the Passage shares almost nothing
 * with it.
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
const COVERED = "Harbours publish the times of high water every 12 months.";
const SECOND_HALF = "creating a scaling constraint";
const UNCITED = "Penguins nest on the harbour cliffs.";

const WORDS = {
  en: {
    part: `Citation 1 doesn't say this part: “${SECOND_HALF}”.`,
    nothing: "Nothing in your Documents says this.",
    noReason: "No reason was given",
    checkAgain: "Check again",
    open: "Open citation 1",
  },
  "zh-CN": {
    part: `引用 1 没有提到这部分：“${SECOND_HALF}”。`,
    nothing: "你的文档中没有内容提到这一点。",
    noReason: "没有给出原因",
    checkAgain: "重新核对",
    open: "打开引用 1",
  },
} as const;

/** Whether the first box fits within the second's width. */
const fitsWithin = async (inner: Locator, outer: Locator) => {
  const [a, b] = [await inner.boundingBox(), await outer.boundingBox()];
  if (!a || !b) return false;
  return a.x >= b.x - 1 && a.x + a.width <= b.x + b.width + 1;
};

for (const [language, width] of [
  ["en", 1440],
  ["zh-CN", 900],
] as const) {
  test(`a cited sentence's Unsupported card names the part its Citation doesn't cover, says when no reason was given, and a sentence with no Citation keeps its line (${language}, ${width}px)`, async () => {
    test.setTimeout(120_000);
    const words = WORDS[language];
    await writeFile(join(sources, "Tides.pdf"), TIDES);
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, width, 900);
    await addDocuments(window, [join(sources, "Tides.pdf")]);
    await newMind(window);
    const editor = window.getByTestId("mind-editor");
    await editor.click();
    await window.keyboard.type(COVERED);
    await window.keyboard.press("Enter");
    await window.keyboard.type(UNCITED);
    const cited = editor.locator(":scope > p").first();
    const uncited = editor.locator(":scope > p").nth(1);
    const marks = window.getByTestId("claim-mark");
    const card = window.getByTestId("claim-card");
    const menu = window.getByTestId("selection-menu");
    /** Opens a mark's card once the marks have settled in the margin. */
    const openCard = async (mark: Locator) => {
      await motionSettled(window);
      await pointAndClick(window, mark);
      await expect(card).toBeVisible();
    };

    // The first sentence is supported and becomes a Citation; the second finds nothing.
    await slideSelect(window, cited, "Harbours publish", { line: uncited, words: "cliffs." });
    await pointAndClick(window, menu.getByTestId("selection-action-find-support"));
    await expect(marks.first()).toHaveAttribute("data-check", "supported");
    await expect(marks.nth(1)).toHaveAttribute("data-check", "unsupported");
    await openCard(marks.first());
    await pointAndClick(window, card.getByTestId("claim-add-citation"));
    await expect(cited.getByTestId("citation")).toHaveAttribute("data-check", "found");
    await expect(marks).toHaveCount(1);

    // The writer adds a second half that no Passage states, after the Citation.
    await clickEndOf(window, cited);
    await window.keyboard.press("ArrowLeft");
    await window.keyboard.type(`, ${SECOND_HALF}`);
    await slideSelect(window, cited, "Harbours publish the times");
    await pointAndClick(window, menu.getByTestId("selection-action-check-against-documents"));
    await expect(marks).toHaveCount(2);
    const mark = marks.first();
    await expect(mark).toHaveAttribute("data-check", "unsupported");

    await openCard(mark);
    await expect(card.getByTestId("claim-badge")).toHaveAttribute("data-check", "unsupported");
    await expect(card.getByTestId("claim-part")).toHaveText(words.part);
    await expect(card.getByTestId("claim-nothing")).toHaveCount(0);
    await expect(card.getByTestId("claim-cited")).toContainText(COVERED);
    await expect(card.getByTestId("claim-open-citation")).toHaveText(words.open);
    expect(await fitsWithin(card.getByTestId("claim-cited"), card)).toBe(true);
    // The card's actions stay as they were.
    await expect(card.getByTestId("claim-find-sources")).toBeVisible();
    await expect(card.getByTestId("claim-search-web")).toBeVisible();
    await expect(card.getByTestId("claim-remove")).toBeVisible();
    await window.keyboard.press("Escape");

    // A sentence with no Citation keeps today's wording.
    await openCard(marks.nth(1));
    await expect(card.getByTestId("claim-nothing")).toHaveText(words.nothing);
    await expect(card.getByTestId("claim-part")).toHaveCount(0);
    await expect(card.getByTestId("claim-no-reason")).toHaveCount(0);
    await expect(card.getByTestId("claim-cited")).toHaveCount(0);
    await window.keyboard.press("Escape");

    // The scripted model gives a sentence that says "no reason" an unsupported verdict and no reason.
    await clickEndOf(window, cited);
    await window.keyboard.press("ArrowLeft");
    await window.keyboard.type(" with no reason");
    await expect(marks.first()).toHaveAttribute("data-check", "needs-recheck");
    await openCard(marks.first());
    await pointAndClick(window, card.getByTestId("claim-check-again"));
    await expect(marks.first()).toHaveAttribute("data-check", "unsupported");
    await openCard(marks.first());
    const noReason = card.getByTestId("claim-no-reason");
    await expect(noReason).toContainText(words.noReason);
    await expect(card.getByTestId("claim-no-reason-check")).toHaveText(words.checkAgain);
    await expect(card.getByTestId("claim-nothing")).toHaveCount(0);
    await expect(card.getByTestId("claim-part")).toHaveCount(0);
    await expect(card.getByTestId("claim-cited")).toContainText(COVERED);
    expect(await fitsWithin(noReason, card)).toBe(true);

    // Check again checks the sentence again: its check is a new one.
    const before = await marks.first().getAttribute("data-claim");
    await pointAndClick(window, card.getByTestId("claim-no-reason-check"));
    await expect(marks.first()).not.toHaveAttribute("data-claim", before ?? "");
    await expect(marks.first()).toHaveAttribute("data-check", "unsupported");

    // The Citation's number opens its Document at the quote.
    await openCard(marks.first());
    await pointAndClick(window, card.getByTestId("claim-open-citation"));
    await expect(card).toHaveCount(0);
    await expect(window.getByTestId("viewer")).toBeVisible();
    await app.close();
  });
}
