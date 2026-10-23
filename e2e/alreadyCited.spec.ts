import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CitationLocation, CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  checkCard,
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

/*
 * A sentence that already cites the Passage its check found is never offered
 * it again (#1620): whatever span the check quoted, the margin mark is the
 * plain tick, the card says "Already cited" with the Citation's own number and
 * words, and a check never leaves a second Citation on the same words. A
 * different clause of the Document, or another Document, still offers Add as
 * Citation. The scripted model (src/main/fakeChat/checks.ts) finds a sentence
 * supported by the line of a Document that shares the most words with it.
 * Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (!SCREENSHOTS) return;
  await motionSettled(window);
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const EXACT_LINE = "The Moon raises two bulges of water on the Earth.";
const LONGER_LINE = "Spring tides happen at new moon and at full moon.";
const NEARBY_LINE = "Neap tides happen at the first and last quarter moons.";
const HEADING = "Harbour times";
const SHORTER_LINE = "Harbours publish the times of high water every 12 months.";
const LIGHTHOUSES_LINE = "Lighthouses guide ships past the rocks at night.";

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", EXACT_LINE] },
  { lines: ["Spring and neap tides", LONGER_LINE, NEARBY_LINE] },
  { lines: [HEADING, SHORTER_LINE, LIGHTHOUSES_LINE] },
]);
const ALMANAC = buildPdf([
  { lines: ["Almanac", "Lighthouses guide ships past rocks near the coast."] },
]);

/** The sentences of the note, one paragraph each, and the Citation each holds. */
const CASES = [
  {
    name: "the same words",
    sentence: "The Moon raises bulges of water on the Earth.",
    document: "Tides",
    page: 1,
    quote: EXACT_LINE,
    cited: true,
  },
  {
    name: "a check that quoted more",
    sentence: "Spring tides happen at new and full moon.",
    document: "Tides",
    page: 2,
    quote: "Spring tides happen at new moon",
    cited: true,
  },
  {
    name: "a check that quoted less",
    sentence: "Harbours publish the times of high water.",
    document: "Tides",
    page: 3,
    quote: `${HEADING} ${SHORTER_LINE}`,
    cited: true,
  },
  {
    name: "a different clause of the Document",
    sentence: "Neap tides happen at the first and last quarter moons.",
    document: "Tides",
    page: 2,
    quote: LONGER_LINE,
    cited: false,
  },
  {
    name: "another Document",
    sentence: "Lighthouses guide ships past rocks at night.",
    document: "Almanac",
    page: 1,
    quote: "Lighthouses guide ships past rocks",
    cited: false,
  },
] as const;

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

const inside = async (inner: Locator, outer: Locator) => {
  const [a, b] = [await inner.boundingBox(), await outer.boundingBox()];
  if (!a || !b) return false;
  return (
    a.x >= b.x - 1 &&
    a.x + a.width <= b.x + b.width + 1 &&
    a.y >= b.y - 1 &&
    a.y + a.height <= b.y + b.height + 1
  );
};

/** Writes the note: each sentence with a Citation to the Document, as an AI-written summary has them. */
async function writeNote(
  window: Page,
  notes: readonly {
    sentence: string;
    document: string;
    page: number | null;
    quote: string;
    location?: CitationLocation | null;
  }[],
) {
  const documents = await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments(),
  );
  const content = notes.map((note) => {
    const document = documents.find((each) => each.name === note.document);
    return {
      type: "paragraph",
      content: [
        { type: "text", text: note.sentence },
        {
          type: "citation",
          attrs: {
            passageId: "passage-1",
            documentId: document?.id,
            documentName: note.document,
            contentHash: document?.contentHash,
            pageFrom: note.page,
            pageTo: note.page,
            location: note.location ?? null,
            quote: note.quote,
            check: "found",
            checkReason: null,
          },
        },
      ],
    };
  });
  const editor = window.getByTestId("mind-editor");
  await editor.evaluate((dom, blocks) => {
    (dom as unknown as { editor: Editor }).editor.commands.insertContentAt(0, blocks);
  }, content);
  return editor.locator(":scope > p");
}

const marks = (window: Page) => window.getByTestId("claim-mark");

async function openCard(window: Page, mark: Locator) {
  await motionSettled(window);
  await pointAndClick(window, mark);
  const card = window.getByTestId("claim-card");
  await expect(card).toBeVisible();
  return card;
}

test("Check this Mind on a note whose sentences cite their Passages: the ones that already cite what was found get no +, however much was quoted, and none gets a second Citation; a different clause and another Document are still offered (#1620)", async () => {
  test.setTimeout(180_000);
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await writeFile(join(sources, "Almanac.pdf"), ALMANAC);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 900, 900);
  await addDocuments(window, [join(sources, "Tides.pdf"), join(sources, "Almanac.pdf")]);
  await newMind(window);
  const lines = await writeNote(window, CASES);
  await expect(lines.nth(CASES.length - 1)).toContainText("Lighthouses guide ships");

  await window.getByTestId("mind-tab").first().click({ button: "right" });
  await pointAndClick(window, window.getByRole("menuitem", { name: "Check this Mind" }));
  await checkCard(window);
  await pointAndClick(
    window,
    window.getByTestId("check-card").getByTestId("mind-check-ask").getByTestId("mind-check-check"),
  );
  await expect(marks(window)).toHaveCount(CASES.length);
  await expect(marks(window).and(window.locator('[data-check="supported"]'))).toHaveCount(
    CASES.length,
  );

  for (const [index, each] of CASES.entries()) {
    if (each.cited) await expect(marks(window).nth(index)).not.toHaveAttribute("data-addable");
    else await expect(marks(window).nth(index)).toHaveAttribute("data-addable", "true");
  }
  // Nothing in the run added a Citation.
  for (const index of CASES.keys()) {
    await expect(lines.nth(index).getByTestId("citation")).toHaveCount(1);
  }

  // Quoted more by the check: the card shows the Citation's own words, its number, and no Add.
  const card = window.getByTestId("claim-card");
  await openCard(window, marks(window).nth(1));
  await expect(card.getByTestId("claim-badge")).toHaveText("Supported");
  await expect(card.getByTestId("claim-already-cited")).toContainText("Already cited");
  await expect(card.getByTestId("claim-already-chip")).toHaveText("2");
  await expect(card.getByTestId("claim-quote")).toHaveText("Spring tides happen at new moon");
  await expect(card.getByTestId("claim-add-citation")).toHaveCount(0);
  await expect(card.getByTestId("claim-open")).toHaveClass(/citation-card-action--primary/);
  expect(await inside(card.getByTestId("claim-open"), card)).toBe(true);
  await screenshot(window, "already-cited-en");
  await window.keyboard.press("Escape");
  await expect(card).toBeHidden();

  // Quoted less by the check: the Citation's longer words are shown.
  await openCard(window, marks(window).nth(2));
  await expect(card.getByTestId("claim-already-chip")).toHaveText("3");
  await expect(card.getByTestId("claim-quote")).toHaveText(`${HEADING} ${SHORTER_LINE}`);
  await expect(card.getByTestId("claim-add-citation")).toHaveCount(0);
  await window.keyboard.press("Escape");

  // The same words: no Add either.
  await openCard(window, marks(window).nth(0));
  await expect(card.getByTestId("claim-already-chip")).toHaveText("1");
  await expect(card.getByTestId("claim-add-citation")).toHaveCount(0);
  await window.keyboard.press("Escape");

  // A different clause of the same Document, and another Document: Add as Citation stays, and adds
  // (its mark goes, as the Citation's own check stands for it).
  for (const index of [3, 4]) {
    await openCard(window, marks(window).and(window.locator("[data-addable]")).first());
    await expect(card.getByTestId("claim-already-cited")).toHaveCount(0);
    await expect(card.getByTestId("claim-add-citation")).toBeVisible();
    await expect(card.getByTestId("claim-add-citation")).toHaveClass(
      /citation-card-action--primary/,
    );
    if (index === 3) await screenshot(window, "add-as-citation-en");
    await pointAndClick(window, card.getByTestId("claim-add-citation"));
    await expect(lines.nth(index).getByTestId("citation")).toHaveCount(2);
  }
  for (const index of [0, 1, 2]) {
    await expect(lines.nth(index).getByTestId("citation")).toHaveCount(1);
  }
});

test("In Chinese and a narrow window: a Citation whose quote differs in punctuation and width still counts as already cited, and a different clause is offered (#1620)", async () => {
  test.setTimeout(180_000);
  await writeFile(
    join(sources, "潮汐.md"),
    [
      "# 潮汐",
      "",
      "春季大潮出现在新月和满月，Spring tides happen at new moon and at full moon。",
      "",
      "小潮出现在上弦月和下弦月，Neap tides happen at the first and last quarter moons。",
      "",
    ].join("\n"),
  );
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 900, 720);
  await addDocuments(window, [join(sources, "潮汐.md")]);
  await newMind(window);
  const lines = await writeNote(window, [
    {
      sentence: "Spring tides happen at new and full moon.",
      document: "潮汐",
      page: null,
      quote: "春季大潮出现在新月和满月,Spring tides happen at new moon",
    },
    {
      sentence: "Neap tides happen at the first and last quarter moons.",
      document: "潮汐",
      page: null,
      quote: "春季大潮出现在新月和满月，Spring tides happen at new moon and at full moon。",
    },
  ]);
  await expect(lines.nth(1)).toContainText("Neap tides");

  await slideSelect(window, lines.nth(0), "Spring tides", { line: lines.nth(1), words: "moons." });
  await pointAndClick(
    window,
    window.getByTestId("selection-menu").getByTestId("selection-action-check-against-documents"),
  );
  await expect(marks(window)).toHaveCount(2);
  await expect(marks(window).and(window.locator('[data-check="supported"]'))).toHaveCount(2);
  await expect(marks(window).nth(0)).not.toHaveAttribute("data-addable");
  await expect(marks(window).nth(1)).toHaveAttribute("data-addable", "true");

  const card = await openCard(window, marks(window).nth(0));
  await expect(card.getByTestId("claim-already-cited")).toContainText("已引用");
  await expect(card.getByTestId("claim-already-chip")).toHaveText("1");
  await expect(card.getByTestId("claim-add-citation")).toHaveCount(0);
  expect(await inside(card.getByTestId("claim-open"), card)).toBe(true);
  await screenshot(window, "already-cited-zh-narrow");
  await window.keyboard.press("Escape");

  await openCard(window, marks(window).nth(1));
  await expect(card.getByTestId("claim-add-citation")).toHaveText("添加为引用");
  await screenshot(window, "add-as-citation-zh-narrow");
  await pointAndClick(window, card.getByTestId("claim-add-citation"));
  await expect(lines.nth(1).getByTestId("citation")).toHaveCount(2);
  await expect(lines.nth(0).getByTestId("citation")).toHaveCount(1);
});
