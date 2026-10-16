import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/**
 * A sentence that cites a court decision no Document holds says so on its
 * card, instead of treating it as one more unsupported sentence, and offers no
 * web search to settle it. With the opinion among the Documents it is checked
 * against it as usual.
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
const OPINION = buildPdf([
  {
    lines: [
      "Hallmark Builders, Inc. v. Rio Grande Supply Co.",
      "512 S.W.3d 101 (Tex. 2017)",
      "A supplier lien attaches when the materials are delivered to the job site.",
    ],
  },
]);
const CITATION = "Hallmark Builders v. Rio Grande Supply, 512 S.W.3d 101 (Tex. 2017)";
const INVENTED = `See ${CITATION}.`;
const BESIDE = "Penguins nest on the harbour cliffs.";
const HELD = `A supplier lien attaches when the materials are delivered to the job site, see ${CITATION}.`;

const WORDS = {
  en: {
    title: "This cites a court decision.",
    body: "IncarnaMind can't check that the case exists or that it says this. Look it up in your legal research service before you rely on it.",
    nothing: "Nothing in your Documents says this.",
  },
  "zh-CN": {
    title: "本句引用了法院判决。",
    body: "IncarnaMind 无法核实这个判决是否存在、是否这样说。依赖它之前，请在你使用的法律数据库中查一下。",
    nothing: "你的文档中没有内容提到这一点。",
  },
} as const;

for (const language of ["en", "zh-CN"] as const) {
  test(`the card of a cited court decision no Document holds says it can't be checked (${language})`, async () => {
    test.setTimeout(120_000);
    const words = WORDS[language];
    await writeFile(join(sources, "Tides.pdf"), TIDES);
    const { app, window } = await launchApp(dataDir, { fakeChat: true, home: sources });
    await setWindowSize(app, window, 1440, 900);
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await addDocuments(window, [join(sources, "Tides.pdf")]);
    await newMind(window);
    const editor = window.getByTestId("mind-editor");
    await editor.click();
    await window.keyboard.type(INVENTED);
    await window.keyboard.press("Enter");
    await window.keyboard.type(BESIDE);
    const lines = editor.locator(":scope > p");
    await slideSelect(window, lines.first(), "See Hallmark", {
      line: lines.nth(1),
      words: "cliffs.",
    });
    await pointAndClick(
      window,
      window.getByTestId("selection-menu").getByTestId("selection-action-find-support"),
    );
    const marks = window.getByTestId("claim-mark");
    await expect(marks).toHaveCount(2);
    const card = window.getByTestId("claim-card");

    await pointAndClick(window, marks.first());
    await expect(marks.first()).toHaveAttribute("data-check", "unsupported");
    await expect(card.getByTestId("claim-badge")).toHaveAttribute("data-check", "unsupported");
    const notice = card.getByTestId("claim-court-decision");
    await expect(notice).toContainText(words.title);
    await expect(notice).toContainText(words.body);
    await expect(card.getByTestId("claim-nothing")).toHaveCount(0);
    await expect(card.getByTestId("claim-search-web")).toHaveCount(0);
    await expect(card.getByTestId("claim-remove")).toBeVisible();

    // The sentence beside it keeps the plain card, with every way to look.
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
    await pointAndClick(window, marks.nth(1));
    await expect(card.getByTestId("claim-nothing")).toHaveText(words.nothing);
    await expect(card.getByTestId("claim-court-decision")).toHaveCount(0);
    await expect(card.getByTestId("claim-search-web")).toBeVisible();

    // Remove the mark takes the cited sentence's mark away.
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
    await pointAndClick(window, marks.first());
    await pointAndClick(window, card.getByTestId("claim-remove"));
    await expect(marks).toHaveCount(1);
    await app.close();
  });
}

test("a cited court decision whose opinion is among the Documents is checked against it as usual", async () => {
  test.setTimeout(120_000);
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await writeFile(join(sources, "Hallmark Builders v Rio Grande Supply.pdf"), OPINION);
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home: sources });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [
    join(sources, "Tides.pdf"),
    join(sources, "Hallmark Builders v Rio Grande Supply.pdf"),
  ]);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type(HELD);
  const line = editor.locator(":scope > p").first();
  await slideSelect(window, line, "A supplier", { line, words: "2017)." });
  await pointAndClick(
    window,
    window.getByTestId("selection-menu").getByTestId("selection-action-find-support"),
  );
  const marks = window.getByTestId("claim-mark");
  await expect(marks).toHaveCount(1);
  await expect(marks.first()).toHaveAttribute("data-check", "supported");
  await pointAndClick(window, marks.first());
  const card = window.getByTestId("claim-card");
  await expect(card.getByTestId("claim-badge")).toHaveAttribute("data-check", "supported");
  await expect(card.getByTestId("claim-court-decision")).toHaveCount(0);
  await app.close();
});
