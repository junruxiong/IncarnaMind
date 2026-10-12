import { readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import type { FakeScanScript } from "../src/core/documents/scan/fake";
import { buildPdf } from "../tests/helpers/pdf";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

const SCANS = join(__dirname, "../tests/fixtures/scans");
const picture = (id: string) => readFileSync(join(SCANS, `${id}.jpg`));
/** What the real model read on each page of the set, with each line's box (see eval/ocr/record.check.ts). */
const read = JSON.parse(readFileSync(join(SCANS, "read.json"), "utf8")) as FakeScanScript;
/** Its lines and their boxes, without the words the model happened to be unsure of. */
const linesOf = (id: string) =>
  (read.pages[id]?.lines ?? []).map((line) =>
    typeof line === "string" ? line : { text: line.text, quad: line.quad },
  );

/** Each language's scan, the Question about it, and the words the app shows. */
const CASES = {
  en: {
    pages: ["en-clean", "en-noisy"],
    unsure: ["convenience"],
    marked: ["convenience"],
    washed: "Either party may terminate this Agreement for convenience",
    question: "Can either party terminate for convenience?",
    reading: "Reading 2 scanned pages…",
    scan: "Text read from a scan · 1 word may be misread",
  },
  "zh-CN": {
    pages: ["zh-clean", "zh-noisy"],
    unsure: ["租赁"],
    // Each Chinese character is a word of its own.
    marked: ["租", "赁"],
    washed: "房屋租赁合同",
    question: "押金什么时候退还？",
    reading: "正在识别 2 页扫描件…",
    scan: "从扫描件识别的文字 · 2 处可能识别有误",
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

/** Asks a Question in a new Mind and waits for its Answer to finish. */
async function ask(window: Page, text: string) {
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type(text);
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  return answer;
}

for (const language of ["en", "zh-CN"] as const) {
  test(`a scanned PDF is read page by page, a Question about it gets a checked Citation, and the viewer washes the quote on the page's picture (${language})`, async () => {
    const chosen = CASES[language];
    // What the stand-in for the text recognition model reads on each page.
    const script = join(sources, "scans.json");
    await writeFile(
      script,
      JSON.stringify({
        pages: Object.fromEntries(
          chosen.pages.map((id, index) => [
            index + 1,
            { lines: linesOf(id), unsure: [...chosen.unsure] },
          ]),
        ),
      }),
    );
    const file = join(sources, "Scan.pdf");
    await writeFile(file, buildPdf(chosen.pages.map((id) => ({ scan: picture(id) }))));
    const { app, window } = await launchApp(dataDir, { fakeChat: true, scans: script });
    await dismissChatSetup(window);
    await window.evaluate(async (chosenLanguage) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await bridge.updateSettings({ user: { language: chosenLanguage } });
    }, language);
    await useLocalChatModel(window);

    // Its status says its scanned pages are being read, then it is ready.
    await window.getByTestId("add-documents-input").setInputFiles([file]);
    const row = window.getByTestId("document-list-item");
    await expect(row.getByTestId("document-status")).toContainText(chosen.reading);
    await expect(row).toHaveAttribute("data-status", "ready", { timeout: 20_000 });

    const answer = await ask(window, chosen.question);
    const citation = answer.getByTestId("citation");
    await expect(citation).toHaveAttribute("data-check", "found");

    // Its card says the quote was read from a scan, and marks a word that may be misread.
    await citation.getByTestId("citation-chip").hover();
    const card = window.getByTestId("citation-card");
    await expect(card.getByTestId("citation-scan")).toHaveText(chosen.scan);
    await expect(card.getByTestId("citation-unsure-word")).toHaveText([...chosen.marked]);

    // Opened, the viewer washes the quote on the page's picture, over the line it was read from.
    await citation.getByTestId("citation-chip").click();
    const viewer = window.getByTestId("viewer");
    const page = viewer.locator('[data-page-number="1"]');
    const highlights = page.locator("[data-quote-highlight]");
    await expect(highlights.first()).toBeVisible();
    await expect(highlights.first()).toBeInViewport();
    const box = await highlights.first().boundingBox();
    const pageBox = await page.boundingBox();
    expect(box && pageBox).toBeTruthy();
    if (box && pageBox) {
      expect(box.x).toBeGreaterThanOrEqual(pageBox.x);
      expect(box.y).toBeGreaterThanOrEqual(pageBox.y);
      expect(box.x + box.width).toBeLessThanOrEqual(pageBox.x + pageBox.width + 1);
    }
    // It lies over the line read, as the model placed it on the picture.
    expect((await highlights.allTextContents()).join("")).toContain(chosen.washed);
    // The text read lies over the picture as a PDF's own text does: it can be selected.
    await highlights.first().dblclick();
    const selected = await window.evaluate(() => globalThis.getSelection()?.toString() ?? "");
    expect(selected.trim().length).toBeGreaterThan(0);
    expect(chosen.washed).toContain(selected.trim());
    await app.close();
  });
}
