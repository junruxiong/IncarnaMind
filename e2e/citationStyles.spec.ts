import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge, ReferencePatch } from "../src/core/api";
import { paragraphsOf, part, unzip } from "../tests/helpers/docx";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  addToNote,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  documentIdOf,
  exportFromMenu,
  launchApp,
  motionSettled,
  newMind,
  pointAndClick,
  removeDataFolder,
  setWindowSize,
  sidebarFolder,
  useLocalChatModel,
  useOutputLocation,
} from "./app";

/** Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots. */
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  const folder = process.env.INCARNAMIND_SCREENSHOTS;
  if (folder) await target.screenshot({ path: join(folder, `${name}.png`) });
}

/** Spring tides are on page 2. */
const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

const TIDES_RECORD: ReferencePatch = {
  type: "article-journal",
  title: "Tides and the Moon",
  authors: [
    { family: "Bowler", given: "Diana E." },
    { family: "Buyung-Ali", given: "Lisette" },
  ],
  issued: "2010",
  containerTitle: "Ocean Review",
  volume: "12",
  pages: "1–20",
  language: "en",
};

/** Sets a Document's reference record, as its "Cite as" panel would. */
const setRecord = (window: Page, documentId: string, patch: ReferencePatch) =>
  window.evaluate(
    ([id, fields]) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateReference(
        id,
        fields,
      ),
    [documentId, patch] as const,
  );

/** A master's thesis: its cover, set as Chinese universities set one, then a page of its text. */
const THESIS = buildPdf([
  {
    printed: [
      { text: "硕士学位论文", size: 26 },
      { text: "城市绿地与夏季降温调查", size: 20 },
      { text: "作者姓名：林小雨", size: 14 },
      { text: "指导教师：王建国 教授", size: 14 },
      { text: "培养单位：华东师范大学地理科学学院", size: 14 },
      { text: "二〇二六年五月", size: 14 },
    ],
  },
  { chineseLines: ["城市绿地使夏季夜间平均降温一摄氏度。"] },
]);

/** A Chinese journal article: its title large, its authors under it, and its DOI. */
const ARTICLE = buildPdf([
  {
    printed: [
      { text: "地理学报 2024 年 3 月", size: 9 },
      { text: "城市绿地对热岛效应的缓解作用", size: 18 },
      { text: "张三 1，李四 2", size: 11 },
      { text: "（1. 北京大学城市与环境学院，北京 100871）", size: 9 },
      { text: "摘要：城市绿地在夏季夜间缓解城市热岛效应。", size: 10 },
      { text: "文献标识码：A", size: 9 },
      { text: "DOI: 10.5555/greenspace-2024", size: 9 },
    ],
  },
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

/**
 * Asks a Question from a new Mind, in the assistant (#201), and adds its
 * Answer to the Mind, as a person keeps one; resolves with the Mind's text,
 * where its Citations now are.
 */
async function ask(window: Page, text: string): Promise<Locator> {
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, editor);
  await addToNote(window, await askInAssistant(window, text));
  await expect(editor.getByTestId("citation").first()).toBeVisible();
  return editor;
}

/** The checks in the Mind's margin, not the assistant's. */
const mindChecks = (window: Page) => window.getByTestId("card").getByTestId("margin-check");

/** Chooses a style in the open picker, typing to find it and pressing Enter, as a person would. */
async function chooseStyle(window: Page, search: string, label: string): Promise<void> {
  const picker = window.getByTestId("citation-style-picker");
  await expect(picker.getByTestId("citation-style-search")).toBeFocused();
  await window.keyboard.type(search);
  await expect(picker.getByTestId("citation-style-option").first()).toHaveText(label);
  await window.keyboard.press("Enter");
  await expect(picker).toBeHidden();
}

test("a Mind citing a paper offers a citation style; chosen, its Citations and References read in it, the References' picker switches it, with Undo, and the export follows", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await writeFile(join(sources, "Lease agreement.txt"), "Spring rent is due at the new moon.");
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf"), join(sources, "Lease agreement.txt")]);
  await setRecord(window, await documentIdOf(window, "Tides"), TIDES_RECORD);

  // "each": one Citation of each Document, side by side. A Mind is Numbered until it chooses:
  // numbered markers, and no References.
  const answer = await ask(window, "When do spring tides happen in each?");
  const chips = answer.getByTestId("citation-chip");
  await expect(chips).toHaveText(["1", "2"]);
  const references = window.getByTestId("references");
  await expect(references).toHaveCount(0);

  // It cites a paper (Tides has a record with its authors), so its end offers a style, APA
  // first in English. Chosen, APA 7 writes the two side by side as one.
  const prompt = window.getByTestId("citation-style-prompt");
  await expect(prompt).toHaveText("Choose a citation style? APA · GB/T 7714 · More…");
  await screenshot(window, "citation-style-prompt-en");
  await pointAndClick(window, prompt.getByTestId("citation-style-prompt-choice").first());
  await expect(prompt).toHaveCount(0);
  await expect(window.getByTestId("undo-line")).toContainText("Citation style changed to APA 7");
  const styled = answer.locator('[data-testid="citation"][data-styled="runs"]');
  await expect(styled).toHaveCount(1);
  await expect(styled.getByTestId("citation-chip")).toHaveText(
    /^\(Bowler & Buyung-Ali, 2010, p\. 2; Lease Agreement, n\.d\., Line 1\)$/,
  );
  // The second Citation is drawn by the first, but keeps its own check in the margin.
  await expect(answer.locator('[data-testid="citation"][data-styled="joined"]')).toHaveCount(1);
  await expect(mindChecks(window)).toHaveCount(2);

  // Over a work's words, the pointer shows that Citation's card.
  const chip = styled.getByTestId("citation-chip");
  await chip.locator('[data-work="1"]').first().hover();
  const card = window.getByTestId("citation-card");
  await expect(card).toContainText("Lease agreement");
  // Moving on to the first work's words: its card shows, and the other's goes.
  await chip.locator('[data-work="0"]').first().hover();
  await expect(card.filter({ hasText: "Tides" })).toBeVisible();
  await expect(card).toHaveCount(1);

  // The References: each Document once, in APA 7, the thin one by its name, needing details.
  await expect(references).toHaveAttribute("data-style", "apa");
  await expect(references.getByTestId("references-style")).toHaveText("APA 7");
  const entries = references.getByTestId("reference-entry");
  await expect(entries).toHaveCount(2);
  await expect(entries.first()).toContainText(
    "Bowler, D. E., & Buyung-Ali, L. (2010). Tides and the Moon. Ocean Review, 12, 1–20.",
  );
  await expect(entries.nth(1)).toHaveAttribute("data-needs-details", "true");
  await expect(entries.nth(1)).toContainText("Lease agreement.");
  await expect(entries.nth(1)).not.toContainText("added");
  await screenshot(window, "references-apa-en");

  // "Needs details" opens the Document's "Cite as" right there, at its first gap, its title.
  // Esc leaves the field, then closes the panel, giving the focus back.
  const needs = entries.nth(1).getByTestId("reference-needs-details");
  await pointAndClick(window, needs);
  const citeAs = window.getByTestId("reference-cite-as");
  await expect(citeAs).toBeVisible();
  await expect(citeAs.locator("input:focus, textarea:focus")).toHaveCount(1);
  await screenshot(window, "references-cite-as-en");
  await window.keyboard.press("Escape");
  await expect(citeAs.locator("input:focus, textarea:focus")).toHaveCount(0);
  await window.keyboard.press("Escape");
  await expect(citeAs).toBeHidden();
  await expect(needs).toBeFocused();

  // An edit to a record reaches the References and the Citation.
  await setRecord(window, await documentIdOf(window, "Tides"), { issued: "2011" });
  await expect(chip).toContainText("Bowler & Buyung-Ali, 2011, p. 2");
  await expect(entries.first()).toContainText("(2011)");

  // The style button opens the picker: IEEE numbers them, in the text and the References.
  await pointAndClick(window, references.getByTestId("references-style"));
  await chooseStyle(window, "ieee", "IEEE");
  await expect(references).toHaveAttribute("data-style", "ieee");
  await expect(chip).toHaveText("[1, p. 2], [2, L. 1]");
  await expect(entries.first().locator(".references-label")).toHaveText("[1]");
  await expect(window.getByTestId("undo-line")).toContainText("Citation style changed to IEEE");
  await screenshot(window, "references-ieee-en");

  // ⌘Z takes it back.
  await references.getByTestId("references-style").focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(references).toHaveAttribute("data-style", "apa");
  await expect(chip).toContainText("Bowler & Buyung-Ali, 2011, p. 2");

  // A note style: the chip is the note's number, and the card shows the note.
  await pointAndClick(window, references.getByTestId("references-style"));
  await chooseStyle(window, "notes", "Chicago (notes and bibliography)");
  await expect(answer.locator('[data-testid="citation"][data-styled="note"]')).toHaveCount(1);
  const note = answer.locator('[data-testid="citation"][data-styled="note"]');
  await expect(note.getByTestId("citation-chip")).toHaveText("1");
  await note.getByTestId("citation-chip").hover();
  await expect(card.getByTestId("citation-styled-note")).toContainText(
    "Diana E. Bowler and Lisette Buyung-Ali",
  );
  await screenshot(window, "references-chicago-notes-en");

  // The export is in the Mind's style: Chicago notes become Word footnotes, with the Bibliography.
  const target = join(sources, "Untitled.docx");
  await useOutputLocation(window, sources);
  await window.keyboard.press("Escape");
  await pointAndClick(window, references.getByTestId("references-export"));
  const dialog = window.getByTestId("export-dialog");
  await expect(dialog.getByTestId("export-citation-style")).toHaveText(
    "Chicago (notes and bibliography)",
  );
  // The options sheet's own style button switches it back to APA 7, for the Mind and so its
  // exports; Save options keeps Word, and Export makes the file.
  await pointAndClick(window, dialog.getByTestId("export-citation-style"));
  await chooseStyle(window, "apa", "APA 7");
  await expect(dialog.getByTestId("export-citation-style")).toHaveText("APA 7");
  await screenshot(window, "export-dialog-style-en");
  await pointAndClick(window, dialog.getByTestId("export-save"));
  await expect(dialog).toBeHidden();
  await exportFromMenu(window);
  const done = window.getByTestId("export-result").getByTestId("export-done");
  await expect(done).toHaveText(/^Saved “.+\.docx” in .+$/);
  const saved = (await done.getAttribute("title")) ?? target;
  const text = paragraphsOf(part(unzip(await readFile(saved)), "word/document.xml")).map(
    (each) => each.text,
  );
  expect(text).toContain("References");
  expect(text.some((line) => line.includes("(Bowler & Buyung-Ali, 2011, p. 2; "))).toBe(true);
  await app.close();
});

test("in Chinese: the offer puts GB/T 7714 first; chosen, raised numbers and 参考文献, and the picker finds 国标 styles; a narrow window keeps the header on its lines", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await writeFile(join(sources, "复杂系统.txt"), "开放的复杂巨系统是一个科学新领域。");
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf"), join(sources, "复杂系统.txt")]);
  await setRecord(window, await documentIdOf(window, "Tides"), TIDES_RECORD);
  await setRecord(window, await documentIdOf(window, "复杂系统"), {
    type: "article-journal",
    title: "一个科学新领域——开放的复杂巨系统及其方法论",
    authors: [{ literal: "钱学森" }, { literal: "于景元" }, { literal: "戴汝为" }],
    issued: "1990",
    containerTitle: "自然杂志",
    language: "zh",
  });

  const answer = await ask(window, "开放的复杂巨系统 each");
  const prompt = window.getByTestId("citation-style-prompt");
  await expect(prompt).toHaveText("选择引用格式？GB/T 7714 · APA · 更多…");
  await screenshot(window, "citation-style-prompt-zh");
  await pointAndClick(window, prompt.getByTestId("citation-style-prompt-choice").first());
  const styled = answer.locator('[data-testid="citation"][data-styled="runs"]').first();
  // A page raised after its number, as GB/T 7714 prints one; lines in brackets: "[1]（第 1 行）".
  await expect(styled.getByTestId("citation-chip")).toHaveText(/^(\[\d\](\d+|（第 \d+ 行）)?)+$/);
  // The numbers raised; a chip with a Location on the line stays on the line itself.
  await expect(
    styled.getByTestId("citation-chip").locator("sup").filter({ hasText: /\d/ }).first(),
  ).toBeVisible();

  const references = window.getByTestId("references");
  await expect(references).toHaveAttribute("data-style", "gb-t-7714-2015-numeric");
  await expect(references.getByRole("heading")).toHaveText("参考文献");
  await expect(references.getByTestId("references-style")).toHaveText(
    "GB/T 7714—2015（顺序编码制）",
  );
  await expect(
    references.getByTestId("reference-entry").filter({ hasText: "钱学森" }),
  ).toContainText(
    "钱学森, 于景元, 戴汝为. 一个科学新领域——开放的复杂巨系统及其方法论[J]. 自然杂志, 1990.",
  );
  await screenshot(window, "references-gbt-zh");

  // Typed in Chinese, the picker finds the national standard's styles; the author-date one
  // writes an English work with "et al." and a Chinese one with 等.
  await pointAndClick(window, references.getByTestId("references-style"));
  const picker = window.getByTestId("citation-style-picker");
  await window.keyboard.type("国标");
  await expect(picker.getByTestId("citation-style-option")).toHaveCount(4);
  await screenshot(window, "citation-style-picker-zh");
  await window.keyboard.press("ArrowDown");
  await window.keyboard.press("Enter");
  await expect(references).toHaveAttribute("data-style", "gb-t-7714-2015-author-date");
  await expect(window.getByTestId("undo-line")).toContainText(
    "引用格式已改为 GB/T 7714—2015（著者-出版年制）",
  );

  // At the narrowest window the header wraps by word and nothing spills out of the column.
  await setWindowSize(app, window, 900, 660);
  await expect(window.getByTestId("assistant")).toHaveAttribute("data-dock", "bottom");
  const column = await window.getByTestId("mind-pane").boundingBox();
  const head = await references.locator(".references-head").boundingBox();
  if (!column || !head) throw new Error("The References or the Mind aren't on screen.");
  expect(head.x + head.width).toBeLessThanOrEqual(column.x + column.width + 1);
  await screenshot(window, "references-narrow-zh");
  await app.close();
});

test("in Chinese, a thesis Mind: 引用信息 read from the PDFs waits to be checked; GB/T 7714 by default, in the Export menu; pages raised, and a thin entry flagged in the Word file and named after the export", async () => {
  await writeFile(join(sources, "林小雨学位论文.pdf"), THESIS);
  await writeFile(join(sources, "绿地与热岛.pdf"), ARTICLE);
  await writeFile(join(sources, "城市绿地调查笔记.txt"), "城市绿地调查：公园比周围街道凉快。");
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await useLocalChatModel(window);
  await addDocuments(
    window,
    ["林小雨学位论文.pdf", "绿地与热岛.pdf", "城市绿地调查笔记.txt"].map((name) =>
      join(sources, name),
    ),
  );

  // The thesis's 引用信息 came from its cover, and waits in the Library to be checked.
  await window.getByTestId("open-library").click();
  const thesisRow = window.getByTestId("library-document").filter({ hasText: "林小雨学位论文" });
  const toggle = thesisRow.getByTestId("cite-as-toggle");
  await expect(toggle).toHaveText("需要核对");
  await pointAndClick(window, toggle);
  const panel = thesisRow.getByTestId("cite-as");
  await expect(panel.getByTestId("reference-type")).toHaveValue("thesis");
  await expect(panel.getByTestId("reference-title")).toHaveText("城市绿地与夏季降温调查");
  await expect(panel.getByTestId("reference-authors")).toHaveText("林小雨");
  await expect(panel.getByTestId("reference-issued")).toHaveText("2026");
  await expect(panel.getByTestId("reference-publisher")).toHaveText("华东师范大学");
  await expect(panel.locator("dt[data-unchecked]")).toHaveCount(5);
  await screenshot(window, "cite-as-check-zh");
  await pointAndClick(window, panel.getByTestId("reference-confirm"));
  await expect(panel.locator("dt[data-unchecked]")).toHaveCount(0);
  await expect(toggle).toHaveText("引用信息");
  await expect(window.getByTestId("undo-line")).toContainText("已确认“林小雨学位论文”的引用信息");

  // A Mind in Chinese citing the three: GB/T 7714 by default, and it says so.
  const answer = await ask(window, "城市绿地 each");
  await pointAndClick(window, answer.locator("p").last());
  await window.keyboard.press("ControlOrMeta+End");
  await window.keyboard.press("Enter");
  await window.keyboard.type("本文梳理城市绿地对夏季降温的作用，并比较不同城市的热岛效应。");
  const references = window.getByTestId("references");
  await expect(references).toHaveAttribute("data-style", "gb-t-7714-2015-numeric");
  await expect(references.getByTestId("references-style")).toHaveText(
    "GB/T 7714—2015（顺序编码制）",
  );
  await expect(references.getByTestId("references-style-default")).toHaveText("默认");
  await expect(window.getByTestId("citation-style-prompt")).toHaveCount(0);
  const entries = references.getByTestId("reference-entry");
  await expect(entries.filter({ hasText: "林小雨" })).toContainText(
    "林小雨. 城市绿地与夏季降温调查[D]. 华东师范大学, 2026.",
  );
  await expect(entries.filter({ hasText: "张三" })).toContainText(
    "张三, 李四. 城市绿地对热岛效应的缓解作用[J/OL]. 2024. DOI:10.5555/greenspace-2024.",
  );
  const thin = entries.filter({ hasText: "城市绿地调查笔记" });
  await expect(thin).toHaveAttribute("data-needs-details", "true");
  // Each cited work on its own, its page raised after its number, lines in brackets: "[1]（第 1 行）[2]1[3]1".
  await expect(answer.getByTestId("citation-chip").first()).toHaveText(
    /^(\[\d\](\d+|（第 \d+ 行）)){3}$/,
  );
  await screenshot(window, "references-gbt-default-zh");

  // Where the User exports: the Export menu names the style, and opens the picker.
  await useOutputLocation(window, sources);
  await pointAndClick(window, window.getByTestId("export-mind"));
  const styleItem = window.getByTestId("export-citation-style-menu");
  await expect(styleItem).toHaveText("引用格式GB/T 7714—2015（顺序编码制）");
  await screenshot(window, "export-menu-style-zh");
  await pointAndClick(window, styleItem);
  const picker = window.getByTestId("citation-style-picker");
  await expect(picker).toBeVisible();
  await window.keyboard.press("Escape");
  await expect(picker).toBeHidden();

  // Exported to Word: raised pages, 参考文献, and the thin entry flagged, never "添加于".
  await exportFromMenu(window, "docx");
  const result = window.getByTestId("export-result");
  const done = result.getByTestId("export-done");
  await expect(done).toHaveText(/\.docx/);
  const needs = result.getByTestId("export-needs-details");
  await expect(needs).toContainText("“城市绿地调查笔记”的引用信息待补充，已在文件中标出");
  await screenshot(window, "export-needs-details-zh");
  const saved = (await done.getAttribute("title")) ?? "";
  const paragraphs = paragraphsOf(part(unzip(await readFile(saved)), "word/document.xml")).map(
    (each) => each.text,
  );
  expect(paragraphs).toContain("参考文献");
  expect(paragraphs.some((line) => /(\[\d\](\d+|（第 \d+ 行）)){3}/.test(line))).toBe(true);
  expect(
    paragraphs.some((line) => line.includes("林小雨. 城市绿地与夏季降温调查[D]. 华东师范大学")),
  ).toBe(true);
  expect(paragraphs.some((line) => line.endsWith("城市绿地调查笔记. [信息待补充]"))).toBe(true);
  expect(paragraphs.join("\n")).not.toContain("添加于");

  // Show goes to the thin entry, whose 信息待补充 fills it in right there.
  await pointAndClick(window, needs.getByTestId("export-needs-details-show"));
  await expect(thin.getByTestId("reference-needs-details")).toBeFocused();
  await app.close();
});

test("in Chinese, a Mind that takes GB/T 7714 by default says so once on the undo line: Change opens the picker, Undo puts Numbered back, and it never shows again in that Mind", async () => {
  await writeFile(join(sources, "绿地与热岛.pdf"), ARTICLE);
  let session = await launchApp(dataDir, { fakeChat: true });
  let { window } = session;
  await dismissChatSetup(window);
  await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "绿地与热岛.pdf")]);
  try {
    /** A new Mind citing the article, in Chinese, titled `title`. */
    const chineseMind = async (title: string) => {
      const editor = await ask(window, "城市绿地 each");
      await window.getByTestId("mind-title").fill(title);
      await pointAndClick(window, editor.locator("p").last());
      await window.keyboard.press("ControlOrMeta+End");
      await window.keyboard.press("Enter");
      await window.keyboard.type("本文梳理城市绿地对夏季降温的作用，并比较不同城市的热岛效应。");
      return editor;
    };
    const message = window.getByTestId("undo-line").getByTestId("undo-message");
    const notice = "此 Mind 引用了论文，引用格式已设为 GB/T 7714—2015（顺序编码制）";

    // The first Mind: the line says so, with Change and Undo; Change opens the picker.
    await chineseMind("第一章");
    const references = window.getByTestId("references");
    await expect(references).toHaveAttribute("data-style", "gb-t-7714-2015-numeric");
    await expect(message).toHaveText(notice);
    await motionSettled(window);
    await screenshot(window, "default-style-notice-zh");
    await pointAndClick(window, window.getByTestId("undo-change-style"));
    const picker = window.getByTestId("citation-style-picker");
    await expect(picker).toBeVisible();
    await expect(message).toHaveCount(0);
    await window.keyboard.press("Escape");
    await expect(picker).toBeHidden();
    // Writing on, it doesn't say it again.
    await window.keyboard.type("城市绿地使夜间降温。");
    await expect(references).toHaveAttribute("data-style", "gb-t-7714-2015-numeric");
    await expect(message).toHaveCount(0);

    // The second Mind says it once too; Undo puts Numbered back, as the Mind's own choice.
    const second = await chineseMind("第二章");
    await expect(message).toHaveText(notice);
    await pointAndClick(window, window.getByTestId("undo-action"));
    await expect(window.getByTestId("references")).toHaveCount(0);
    await expect(second.getByTestId("citation-chip").first()).toHaveText("1");
    await window.keyboard.type("再写一句。");
    await expect(message).toHaveCount(0);

    // After a restart, the first Mind is still in GB/T 7714 by default, and says nothing.
    await session.app.close();
    session = await launchApp(dataDir, { fakeChat: true });
    window = session.window;
    await pointAndClick(window, window.getByTestId("mind-list-item").filter({ hasText: "第一章" }));
    await expect(window.getByTestId("references")).toHaveAttribute(
      "data-style",
      "gb-t-7714-2015-numeric",
    );
    await expect(window.getByTestId("references-style-default")).toBeVisible();
    await pointAndClick(window, window.getByTestId("export-mind"));
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("undo-line").getByTestId("undo-message")).toHaveCount(0);
  } finally {
    await session.app.close();
  }
});

test("a Mind citing files with no details isn't offered a style until one is a paper; closed, the offer stays closed there, with Undo; More… opens the picker", async () => {
  await writeFile(join(sources, "Lease agreement.txt"), "Spring rent is due at the new moon.");
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Lease agreement.txt")]);

  // A contract: a numbered marker, as before styles, and nothing offered.
  const answer = await ask(window, "When is spring rent due?");
  await expect(answer.getByTestId("citation-chip")).toHaveText(["1"]);
  const prompt = window.getByTestId("citation-style-prompt");
  await expect(mindChecks(window)).toHaveCount(1);
  await expect(prompt).toHaveCount(0);
  await expect(window.getByTestId("references")).toHaveCount(0);

  // Once its record says it is a paper, the offer shows at the Mind's end.
  await setRecord(window, await documentIdOf(window, "Lease agreement"), {
    type: "article-journal",
    title: "Rent and the Moon",
    authors: [{ family: "Oke", given: "T. R." }],
    issued: "1982",
  });
  await expect(prompt).toBeVisible();

  // Closed, it doesn't come back in this Mind; Undo brings it back.
  await pointAndClick(window, prompt.getByTestId("citation-style-prompt-dismiss"));
  await expect(prompt).toHaveCount(0);
  await expect(window.getByTestId("undo-line")).toContainText(
    "Won't ask about a citation style in this Mind",
  );
  await pointAndClick(window, window.getByTestId("undo-action"));
  await expect(prompt).toBeVisible();

  // More… opens the picker, for any style.
  await pointAndClick(window, prompt.getByTestId("citation-style-prompt-more"));
  await chooseStyle(window, "harvard", "Harvard (Cite Them Right)");
  await expect(prompt).toHaveCount(0);
  await expect(window.getByTestId("references")).toHaveAttribute("data-style", "harvard");
  await expect(answer.getByTestId("citation-chip")).toHaveText("(Oke, 1982, l. 1)");
  await app.close();
});

test("the style chosen for a Mind in a Folder, from its menu, is the one a new Mind there starts in", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const folder = await core.createLibraryGroup({ name: "Thesis", description: "" });
    await core.createMind({ title: "Chapter 1", folderId: folder.id });
  });
  const folder = sidebarFolder(window, "Thesis");
  await pointAndClick(window, folder.getByTestId("folder-row"));
  await expect(folder).toHaveAttribute("data-expanded", "true");

  /** Opens a Mind's citation style picker from its row's ⋯ menu. */
  const openPicker = async (row: Locator) => {
    await row.hover();
    await pointAndClick(window, row.getByTestId("mind-menu"));
    await pointAndClick(window, row.getByTestId("mind.citationStyle"));
  };
  await openPicker(folder.getByTestId("mind-row").filter({ hasText: "Chapter 1" }));
  await chooseStyle(window, "ieee", "IEEE");

  // "New Mind here", from the Folder's menu: it starts in IEEE, ticked in its picker.
  await pointAndClick(window, folder.getByTestId("folder-row"), { button: "right" });
  await pointAndClick(window, folder.getByTestId("folder.newMind"));
  const rows = folder.getByTestId("mind-row");
  await expect(rows).toHaveCount(2);
  await openPicker(rows.filter({ hasNotText: "Chapter 1" }));
  const current = window.locator('[data-testid="citation-style-option"][data-current="true"]');
  await expect(current).toHaveText("IEEE");
  await window.keyboard.press("Escape");
  await expect(window.getByTestId("citation-style-picker")).toBeHidden();
  await app.close();
});
