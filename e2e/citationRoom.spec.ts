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
 * A Citation at 1,100 × 760, where the assistant leaves the card too narrow for two groups, so
 * the window shows one: the Document comes to the front, its tab keeps its name beside the
 * Minds' tabs, and closing it brings the note back with the Citation's card still open. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also keep screenshots.
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

for (const language of ["en", "zh-CN"] as const) {
  test(`a Citation opened at 1,100 × 760 shows the cited page, the tabs keep their names, and the note comes back with the card open (${language})`, async () => {
    await writeFile(join(sources, "Tides.pdf"), TIDES);
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1100, 760);
    await addDocuments(window, [join(sources, "Tides.pdf")]);
    await newMind(window);
    await window.getByTestId("mind-title").fill("Draft");
    await window.getByTestId("mind-editor").click();
    await addToNote(window, await askInAssistant(window, "When do spring tides happen?"));
    for (const name of OTHERS[language]) {
      await newMind(window);
      await window.getByTestId("mind-title").fill(name);
    }
    await window.getByRole("tab", { name: "Draft", exact: true }).click();
    const editor = window.getByTestId("mind-editor").filter({ visible: true });
    await expect(editor.getByTestId("citation")).toHaveCount(1);

    await pointAndClick(window, editor.getByTestId("citation-chip"));

    // The cited page takes the pane, with the quote washed; the window shows one group.
    await expect(window.getByTestId("tab-groups")).toHaveAttribute("data-merged", "true");
    const viewer = window.getByTestId("viewer");
    await expect(window.getByTestId("viewer-title")).toHaveText("Tides");
    await expect(
      viewer.locator('[data-page-number="2"] [data-quote-highlight]').first(),
    ).toBeVisible();

    // Every tab keeps enough of its name to tell it from the others, and the one in front is at
    // least 132 px (DESIGN.md, Tabs).
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

    // Closing the preview is one click, and the card is open where the Citation is.
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

    // Esc puts the card away, and it stays away.
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
    await app.close();
  });
}
