import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  addToNote,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * A Citation where the assistant leaves the card too narrow for two groups (1,100 × 760): the
 * window shows one, the Document comes to the front, and closing it brings the note back with the
 * Citation in view and its card open. Where the groups sit side by side (1,440 px) the note stays.
 * Set INCARNAMIND_SCREENSHOTS to a folder to also keep screenshots.
 */

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

const OTHERS = {
  en: ["Weather log", "Moon notes", "Untitled draft"],
  "zh-CN": ["天气记录", "月相笔记", "未命名草稿"],
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

async function screenshot(window: Page, name: string): Promise<void> {
  const folder = process.env.INCARNAMIND_SCREENSHOTS;
  if (folder) await window.screenshot({ path: join(folder, `${name}.png`) });
}

/** A Mind "Draft" with `lines` paragraphs, then an Answer's Citation added at its end. */
async function draftCiting(window: Page, lines: number): Promise<void> {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await newMind(window);
  await window.getByTestId("mind-title").fill("Draft");
  await window.getByTestId("mind-editor").click();
  if (lines > 0) {
    const text = Array.from({ length: lines }, (_, n) => `Paragraph ${n + 1} of the draft.`);
    await window.keyboard.type(`${text.join("\n")}\n`);
  }
  await addToNote(window, await askInAssistant(window, "When do spring tides happen?"));
}

const editorOf = (window: Page) => window.getByTestId("mind-editor").filter({ visible: true });

async function openOthers(window: Page, names: readonly string[]): Promise<void> {
  for (const name of names) {
    await newMind(window);
    await window.getByTestId("mind-title").fill(name);
  }
  await window.getByRole("tab", { name: "Draft", exact: true }).click();
}

for (const language of ["en", "zh-CN"] as const) {
  test(`a Citation opened at 1,100 × 760 shows the cited page, the tabs keep their names, and the note comes back with the card open (${language})`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1100, 760);
    await draftCiting(window, 0);
    await openOthers(window, OTHERS[language]);
    const editor = editorOf(window);
    await expect(editor.getByTestId("citation")).toHaveCount(1);

    await pointAndClick(window, editor.getByTestId("citation-chip"));

    await expect(window.getByTestId("tab-groups")).toHaveAttribute("data-merged", "true");
    await expect(window.getByTestId("viewer-title")).toHaveText("Tides");
    await expect(
      window.getByTestId("viewer").locator('[data-page-number="2"] [data-quote-highlight]').first(),
    ).toBeVisible();
    const widths = await window
      .getByRole("tab")
      .evaluateAll((tabs) =>
        tabs.map((tab) => [tab.getAttribute("aria-selected"), tab.getBoundingClientRect().width]),
      );
    expect(widths).toHaveLength(5);
    for (const [selected, width] of widths) {
      expect(Number(width)).toBeGreaterThanOrEqual(selected === "true" ? 132 : 100);
    }
    await screenshot(window, `citation-room-document-${language}`);

    await pointAndClick(
      window,
      window.getByRole("tab", { selected: true }).getByTestId("tab-close"),
    );

    await expect(editor).toBeVisible();
    await expect(window.getByTestId("mind-title").filter({ visible: true })).toHaveValue("Draft");
    const card = window.getByTestId("citation-card");
    await expect(card).toBeVisible();
    await expect(card).toHaveAttribute("data-check", "found");
    await expect(card).toContainText("Tides");
    await motionSettled(window);
    await screenshot(window, `citation-room-note-${language}`);

    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
    await app.close();
  });
}

test("a Citation far down a long note is in view, with its card, when the note comes back at 1,100 × 760", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1100, 760);
  await draftCiting(window, 40);
  const editor = editorOf(window);
  const chip = editor.getByTestId("citation-chip");
  await window
    .locator(".mind-scroller:visible")
    .evaluate((scroller) => scroller.scrollTo({ top: 0 }));
  await expect(chip).not.toBeInViewport();
  await chip.scrollIntoViewIfNeeded();

  await pointAndClick(window, chip);
  await expect(window.getByTestId("viewer-title")).toHaveText("Tides");
  await pointAndClick(window, window.getByRole("tab", { selected: true }).getByTestId("tab-close"));

  const card = window.getByTestId("citation-card");
  await expect(card).toBeVisible();
  await expect(chip).toBeInViewport({ ratio: 1 });
  await expect(card).toBeInViewport({ ratio: 1 });
  await screenshot(window, "citation-room-long-note");
  await app.close();
});

test("a Citation opened at 1,440 px opens its page beside the note, which stays", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setWindowSize(app, window, 1440, 900);
  await draftCiting(window, 0);
  const editor = editorOf(window);

  await pointAndClick(window, editor.getByTestId("citation-chip"));

  await expect(window.getByTestId("tab-groups")).not.toHaveAttribute("data-merged", "true");
  await expect(window.getByTestId("viewer-title")).toHaveText("Tides");
  await expect(editor).toBeVisible();
  await expect(window.getByTestId("citation-card")).toBeVisible();
  await motionSettled(window);
  await screenshot(window, "citation-room-wide");
  await app.close();
});
