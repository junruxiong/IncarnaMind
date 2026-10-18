import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  checkCard,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  newMind,
  paintOf,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  tokenColor,
  useLocalChatModel,
} from "./app";

/*
 * The margin holds one mark per sentence, its verdict (#1112); a Citation's own state shows on
 * its marker in the text: neutral when its quote is found, warning when it isn't, error when the
 * Document now says otherwise. One page holds a supported sentence, an unsupported one with two
 * found Citations, a contradicted one and an unchecked one with a quote that isn't found. The
 * scripted model (src/main/fakeChat/checks.ts) gives the verdicts.
 */

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
  { lines: ["Harbour times", "Harbours publish the times of high water every 12 months."] },
]);

const SENTENCES = {
  supported: "Spring tides happen at new and full moon.",
  unsupported:
    "Harbours publish the times of high water every 12 months, and a scaling constraint applies.",
  contradicted: "Harbours publish the times of high water every 6 months.",
  notFound: "The Moon raises two bulges of water on the Earth.",
};

const LANGUAGES = {
  en: {
    found: /Quote found/,
    notFound: /Quote not found/,
    contradicted: /Contradicted/,
    marks: ["Supported", "Unsupported", "Contradicted"].map((state) => `Sentence check: ${state}`),
    health: { found: "1 Citation found", notFound: "1 Citation not found" },
  },
  "zh-CN": {
    found: /已在.*找到引文/,
    notFound: /未在.*找到引文/,
    contradicted: /与文档矛盾/,
    marks: ["有依据", "无依据", "与文档矛盾"].map((state) => `句子核对：${state}`),
    health: { found: "1 处引用已找到", notFound: "1 处引用未找到" },
  },
} as const;

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

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (!SCREENSHOTS) return;
  await motionSettled(window);
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

async function box(locator: Locator) {
  const found = await locator.boundingBox();
  if (!found) throw new Error("The element isn't visible.");
  return found;
}

for (const language of ["en", "zh-CN"] as const) {
  const words = LANGUAGES[language];
  test(`a sentence has one mark in the margin, its verdict, and a Citation's state shows on its marker, at 1,440 and 900 px (${language})`, async () => {
    test.setTimeout(180_000);
    await writeFile(join(sources, "Tides.pdf"), TIDES);
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1440, 900);
    await addDocuments(window, [join(sources, "Tides.pdf")]);
    const [tides] = await window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments(),
    );
    const cite = (
      page: number,
      quote: string,
      check: string,
      checkReason: string | null = null,
    ) => ({
      type: "citation",
      attrs: {
        passageId: `passage-${page}`,
        documentId: tides?.id,
        documentName: "Tides",
        contentHash: tides?.contentHash,
        pageFrom: page,
        pageTo: page,
        location: null,
        quote,
        check,
        checkReason,
      },
    });
    const text = (value: string) => ({ type: "text", text: value });
    const paragraph = (...content: object[]) => ({ type: "paragraph", content });
    const springQuote = "Spring tides happen at new moon and at full moon.";
    const harbourQuote = "Harbours publish the times of high water every 12 months.";

    await newMind(window);
    const editor = window.getByTestId("mind-editor");
    await editor.evaluate(
      (dom, content) => {
        (dom as unknown as { editor: Editor }).editor.commands.insertContentAt(0, content);
      },
      [
        paragraph(text(SENTENCES.supported), cite(2, springQuote, "found")),
        paragraph(
          text(SENTENCES.unsupported),
          cite(3, harbourQuote, "found"),
          cite(3, harbourQuote, "found"),
        ),
        paragraph(text(SENTENCES.contradicted), cite(3, harbourQuote, "contradicted")),
        paragraph(
          text(SENTENCES.notFound),
          cite(2, "Neap tides pull the sea out.", "not-found", "quote-not-on-pages"),
        ),
      ],
    );
    const lines = editor.locator(":scope > p");
    const [supported, unsupported, contradicted, notFound] = [0, 1, 2, 3].map((at) =>
      lines.nth(at),
    );
    if (!supported || !unsupported || !contradicted || !notFound) throw new Error("No page.");

    // The first three sentences are checked; the fourth is left alone.
    await slideSelect(window, supported, "Spring tides", {
      line: contradicted,
      words: "6 months.",
    });
    await pointAndClick(
      window,
      window.getByTestId("selection-menu").getByTestId("selection-action-check-against-documents"),
    );
    const marks = window.getByTestId("claim-mark");
    await expect(marks).toHaveCount(3);
    for (const [at, check] of ["supported", "unsupported", "contradicted"].entries()) {
      await expect(marks.nth(at)).toHaveAttribute("data-check", check);
      await expect(marks.nth(at)).toHaveAccessibleName(words.marks[at] ?? "");
    }

    const margin = window.getByTestId("margin-checks");
    const [neutral, neutralInk, warningWash, warning, errorWash, error] = await Promise.all(
      [
        "--color-chip",
        "--color-ink-secondary",
        "--color-warning-wash",
        "--color-warning",
        "--color-danger-wash",
        "--color-danger",
      ].map((token) => tokenColor(window, token)),
    );

    for (const width of [1440, 900]) {
      await setWindowSize(app, window, width, 900);
      await window.mouse.move(2, 2);
      await motionSettled(window);

      // The margin holds the three sentences' marks, none for a Citation, and none for the
      // sentence that was never checked; each is level with its sentence.
      await expect(margin.getByRole("button")).toHaveCount(3);
      const column = await editor.evaluate((element) => {
        const rect = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return { right: rect.right - Number.parseFloat(style.paddingRight) };
      });
      const checked = [supported, unsupported, contradicted];
      for (const [at, line] of checked.entries()) {
        const [mark, sentence] = [await box(marks.nth(at)), await box(line)];
        const middle = mark.y + mark.height / 2;
        expect
          .soft(middle, `mark ${at + 1} is level with its sentence`)
          .toBeGreaterThanOrEqual(sentence.y);
        expect
          .soft(middle, `mark ${at + 1} is level with its sentence`)
          .toBeLessThanOrEqual(sentence.y + sentence.height);
        expect
          .soft(mark.x, `mark ${at + 1} is right of the text`)
          .toBeGreaterThanOrEqual(column.right - 1);
      }
      const unchecked = await box(notFound);
      for (const mark of await marks.all()) {
        const rect = await box(mark);
        const middle = rect.y + rect.height / 2;
        expect
          .soft(
            middle < unchecked.y || middle > unchecked.y + unchecked.height,
            "no mark by the unchecked sentence",
          )
          .toBe(true);
      }

      // Found: the neutral chip, with no icon, under a supported, an unsupported and a contradicted
      // sentence alike: the sentence's verdict never changes its Citations' markers.
      const found = [
        supported.getByTestId("citation-chip"),
        ...(await unsupported.getByTestId("citation-chip").all()),
      ];
      expect(found).toHaveLength(3);
      for (const marker of found) {
        await expect(marker).toHaveAttribute("data-check", "found");
        await expect(marker).toHaveAccessibleName(words.found);
        expect(await paintOf(marker)).toEqual({ background: neutral, color: neutralInk });
        await expect(marker.locator(".citation-marker-icon")).toBeHidden();
      }

      // Not found: warning on its wash, and nothing in the margin for it.
      const missing = notFound.getByTestId("citation-chip");
      await expect(missing).toHaveAttribute("data-check", "not-found");
      await expect(missing).toHaveAccessibleName(words.notFound);
      expect(await paintOf(missing)).toEqual({ background: warningWash, color: warning });

      // The Document now says otherwise: error on a light red wash.
      const says = contradicted.getByTestId("citation-chip");
      await expect(says).toHaveAttribute("data-check", "contradicted");
      await expect(says).toHaveAccessibleName(words.contradicted);
      expect(await paintOf(says)).toEqual({ background: errorWash, color: error });
      await screenshot(window, `one-mark-${language}-${width}`);
    }

    // A marker opens its Citation's card; a margin mark opens its sentence's check card.
    await pointAndClick(window, notFound.getByTestId("citation-chip"));
    await expect(window.getByTestId("citation-card")).toHaveAttribute("data-check", "not-found");
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("citation-card")).toHaveCount(0);
    await pointAndClick(window, marks.nth(1));
    await expect(window.getByTestId("claim-card")).toHaveAttribute("data-check", "unsupported");
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("claim-card")).toHaveCount(0);

    // The health card keeps its counts of Citations, and each moves to its marker.
    const card = await checkCard(window);
    await expect(card.getByTestId("health-found")).toContainText(words.health.found);
    await expect(card.getByTestId("health-notFound")).toContainText(words.health.notFound);
    await pointAndClick(window, card.getByTestId("health-notFound"));
    await expect(notFound.getByTestId("citation")).toHaveClass(/ProseMirror-selectednode/);
    await expect(notFound.getByTestId("citation-chip")).toBeInViewport();
    await pointAndClick(window, (await checkCard(window)).getByTestId("health-found"));
    await expect(supported.getByTestId("citation")).toHaveClass(/ProseMirror-selectednode/);
    await app.close();
  });
}
