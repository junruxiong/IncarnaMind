import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import {
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
 * How a change reads (#1028; DESIGN.md, AI edits), in Review changes after ⌘J:
 * a word replaced is struck, its replacement underlined, a thin space apart;
 * a paragraph rewritten by more than half shows its new text whole and one
 * "Was: …" (原文：…) line above it, which shows the old text, struck, in place.
 * The scripted model does what "replace X with Y" (把 X 换成 Y) says. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

/** A résumé's summary and a job line, with invented names. */
const SUMMARY = "林晓的主要优势是跨医疗、金融两个行业的数据分析经验，熟悉财务建模与估值。";
const OLD_PART = "跨医疗、金融两个行业的数据分析经验，熟悉财务建模与估值";
const NEW_PART = "十年咨询与审计项目经验，擅长跨部门沟通与团队管理";
const JOB = "2017.05—2017.04 某咨询公司 分析师";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function mindWith(window: Page, lines: readonly string[]): Promise<Locator> {
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  for (const [index, line] of lines.entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await window.keyboard.type(line);
  }
  await expect(editor.locator(":scope > p")).toHaveText(lines);
  return editor;
}

/** ⌘J over `line`, all of it selected, saying what to change; then Review changes. */
async function changeAndReview(window: Page, editor: Locator, line: Locator, instruction: string) {
  const text = (await line.textContent()) ?? "";
  // From its first words to its last, on whichever line they wrap to.
  await slideSelect(window, line, text.slice(0, 3), { line, words: text.slice(-3) });
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type(instruction);
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("edit-pending")).toHaveCount(0);
  await pointAndClick(window, editor.getByTestId("applied-line").getByTestId("applied-review"));
}

/** How far apart, in px, the end of the struck words is from the start of the new ones. */
const gapBetween = async (old: Locator, added: Locator) => {
  const [before, after] = await Promise.all([old.boundingBox(), added.boundingBox()]);
  if (!before || !after) throw new Error("The change isn't drawn.");
  return after.x - (before.x + before.width);
};

for (const language of ["en", "zh-CN"] as const) {
  test(`a word replaced reads word by word: struck, then the new word underlined, a thin space apart (${language})`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1440, 900);
    const [line, instruction, old, added] =
      language === "en"
        ? [
            "The tides here are strong and regular.",
            "replace regular with punctual",
            "regular",
            "punctual",
          ]
        : ["潮水每天非常规律地涨落两次。", "把规律换成准时", "规律", "准时"];
    const editor = await mindWith(window, [line]);
    const paragraph = editor.locator(":scope > p").first();

    await changeAndReview(window, editor, paragraph, instruction);

    const struck = paragraph.locator("del.applied-old");
    const underlined = paragraph.locator(".applied-new");
    await expect(struck).toHaveText(old);
    await expect(underlined).toHaveText(added);
    await expect(struck).toHaveCSS("text-decoration-line", "line-through");
    expect(await gapBetween(struck, underlined)).toBeGreaterThanOrEqual(2);
    await expect(editor.getByTestId("change-was-line")).toHaveCount(0);
    await motionSettled(window);
    await screenshot(window, `changes-word-${language}`);
    await app.close();
  });
}

test("a résumé's summary rewritten by more than half reads whole, with one 原文 line that shows the old text, struck; a date range changed is replaced whole", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 1100, 760);
  const editor = await mindWith(window, [SUMMARY, JOB]);
  const summary = editor.locator(":scope > p").first();
  const job = editor.locator(":scope > p").nth(1);

  await changeAndReview(window, editor, summary, `把${OLD_PART}换成${NEW_PART}`);

  // The new paragraph whole, its old text one line above it, no old and new words between.
  const rewritten = SUMMARY.replace(OLD_PART, NEW_PART);
  await expect(summary).toHaveText(rewritten);
  await expect(summary.locator("del")).toHaveCount(0);
  await expect(summary.locator(".applied-new")).toHaveText(rewritten);
  const was = editor.getByTestId("change-was-line");
  await expect(was).toHaveCount(1);
  await expect(was).toHaveText(`原文：${SUMMARY}`);
  const wasBox = await was.boundingBox();
  const summaryBox = await summary.boundingBox();
  expect((wasBox?.y ?? 0) + (wasBox?.height ?? 0)).toBeLessThanOrEqual((summaryBox?.y ?? 0) + 1);
  await motionSettled(window);
  await screenshot(window, "changes-rewritten-zh");

  // A click shows the old text, struck, in place, before the new; another hides it again.
  await pointAndClick(window, was.getByTestId("change-was"));
  await expect(was.getByTestId("change-was")).toHaveAttribute("aria-expanded", "true");
  await expect(summary.locator("del.applied-old")).toHaveText(SUMMARY);
  await expect(summary.locator("del")).toHaveCount(1);
  await screenshot(window, "changes-rewritten-open-zh");
  await pointAndClick(window, was.getByTestId("change-was"));
  await expect(summary.locator("del")).toHaveCount(0);

  // By keyboard: ⌘K "Show original" with the cursor in the paragraph.
  await pointAndClick(window, summary);
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type("显示原文");
  await pointAndClick(
    window,
    window.locator('[data-testid="palette"] [data-key="command:change.original"]'),
  );
  await expect(summary.locator("del.applied-old")).toHaveText(SUMMARY);
  await pointAndClick(window, editor.getByTestId("applied-line").getByTestId("applied-done"));

  // The date range is one word: struck whole, the new range whole.
  await changeAndReview(window, editor, job, "把2017.04换成2018.09");
  await expect(job.locator("del.applied-old")).toHaveText("2017.05—2017.04");
  await expect(job.locator(".applied-new")).toHaveText("2017.05—2018.09");
  await screenshot(window, "changes-range-zh");
  await app.close();
});

test("a résumé's section rewritten as a whole reads as one line in Review changes, 重写了“核心优势” · 显示原文, which shows its old text, struck", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 1100, 760);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("## 履历亮点与待核实事项");
  for (const line of [
    "跨医疗、金融两个行业的数据分析经验。",
    "熟悉财务建模与估值。",
    "英语 CET-6，雅思 7.5。",
  ]) {
    await window.keyboard.press("Enter");
    await window.keyboard.type(line);
  }
  await expect(editor.locator(":scope > h2")).toHaveText("履历亮点与待核实事项");
  await window.keyboard.press("ControlOrMeta+a");
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("重写这一节");
  await window.keyboard.press("Enter");

  // Applied, counted by paragraph; Review changes reads the section as one line, nothing interleaved.
  const line = editor.getByTestId("applied-line");
  await expect(line.getByTestId("applied-count")).toHaveText("修改了 4 个段落");
  await expect(editor.locator(":scope > h2")).toHaveText("核心优势");
  await pointAndClick(window, line.getByTestId("applied-review"));
  const section = editor.getByTestId("change-section-line");
  await expect(section).toHaveCount(1);
  await expect(section).toContainText("重写了“核心优势”");
  await expect(editor.getByTestId("change-was-line")).toHaveCount(0);
  await expect(editor.locator("del.applied-old")).toHaveCount(0);
  await motionSettled(window);
  await screenshot(window, "changes-section-zh");

  // The cursor in the last change: its bar floats over that one, clear of the section's line (#1018 puts it inline).
  await pointAndClick(window, editor.locator(":scope > p").last());
  const original = section.getByTestId("change-section-original");
  await expect(original).toHaveText("显示原文");
  await pointAndClick(window, original);
  await expect(original).toHaveText("隐藏原文");
  await expect(editor.locator("del.applied-old")).toHaveText([
    "履历亮点与待核实事项",
    "跨医疗、金融两个行业的数据分析经验。",
    "熟悉财务建模与估值。",
    "英语 CET-6，雅思 7.5。",
  ]);
  await screenshot(window, "changes-section-open-zh");
  await app.close();
});
