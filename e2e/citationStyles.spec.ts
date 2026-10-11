import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge, ReferencePatch } from "../src/core/api";
import { paragraphsOf, part, unzip } from "../tests/helpers/docx";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  documentIdOf,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
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
async function ask(window: Page, text: string): Promise<Locator> {
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

/** Chooses a style in the open picker, typing to find it and pressing Enter, as a person would. */
async function chooseStyle(window: Page, search: string, label: string): Promise<void> {
  const picker = window.getByTestId("citation-style-picker");
  await expect(picker.getByTestId("citation-style-search")).toBeFocused();
  await window.keyboard.type(search);
  await expect(picker.getByTestId("citation-style-option").first()).toHaveText(label);
  await window.keyboard.press("Enter");
  await expect(picker).toBeHidden();
}

test("a Mind's Citations and References read in its citation style, which the References' picker switches, with Undo, and the export follows", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  await writeFile(join(sources, "Lease agreement.txt"), "Spring rent is due at the new moon.");
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf"), join(sources, "Lease agreement.txt")]);
  await setRecord(window, await documentIdOf(window, "Tides"), TIDES_RECORD);

  // "each": one Citation of each Document, side by side, which APA 7 (the default) writes as one.
  const answer = await ask(window, "When do spring tides happen in each?");
  const chips = answer.getByTestId("citation-chip");
  await expect(chips).toHaveCount(2);
  const styled = answer.locator('[data-testid="citation"][data-styled="runs"]');
  await expect(styled).toHaveCount(1);
  await expect(styled.getByTestId("citation-chip")).toHaveText(
    /^\(Bowler & Buyung-Ali, 2010, p\. 2; Lease Agreement, n\.d\., Line 1\)$/,
  );
  // The second Citation is drawn by the first, but keeps its own check in the margin.
  await expect(answer.locator('[data-testid="citation"][data-styled="joined"]')).toHaveCount(1);
  await expect(window.getByTestId("margin-check")).toHaveCount(2);

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
  const references = window.getByTestId("references");
  await expect(references).toHaveAttribute("data-style", "apa");
  await expect(references.getByTestId("references-style")).toHaveText("APA 7");
  const entries = references.getByTestId("reference-entry");
  await expect(entries).toHaveCount(2);
  await expect(entries.first()).toContainText(
    "Bowler, D. E., & Buyung-Ali, L. (2010). Tides and the Moon. Ocean Review, 12, 1–20.",
  );
  await expect(entries.nth(1)).toHaveAttribute("data-needs-details", "true");
  await expect(entries.nth(1)).toContainText("Lease agreement");
  await screenshot(window, "references-apa-en");

  // "Needs details" opens the Document's "Cite as" right there, at its first gap, its title.
  // Esc leaves the field, then closes the panel, giving the focus back.
  const needs = entries.nth(1).getByTestId("reference-needs-details");
  await needs.click();
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
  await references.getByTestId("references-style").click();
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
  await references.getByTestId("references-style").click();
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
  await references.getByTestId("references-export").click();
  const dialog = window.getByTestId("export-dialog");
  await expect(dialog.getByTestId("export-citation-style")).toHaveText(
    "Chicago (notes and bibliography)",
  );
  // The dialog's own style button switches it back to APA 7 for this export, and for the Mind.
  await dialog.getByTestId("export-citation-style").click();
  await chooseStyle(window, "apa", "APA 7");
  await expect(dialog.getByTestId("export-citation-style")).toHaveText("APA 7");
  await screenshot(window, "export-dialog-style-en");
  await dialog.getByTestId("export-save").click();
  await expect(dialog.getByTestId("export-done")).toBeVisible();
  const saved = (await dialog.getByTestId("export-done").getAttribute("title")) ?? target;
  await dialog.getByTestId("export-close").click();
  const text = paragraphsOf(part(unzip(await readFile(saved)), "word/document.xml")).map(
    (each) => each.text,
  );
  expect(text).toContain("References");
  expect(text.some((line) => line.includes("(Bowler & Buyung-Ali, 2011, p. 2; "))).toBe(true);
  await app.close();
});

test("in Chinese: GB/T 7714—2015 by default, raised numbers, 参考文献, and the picker finds 国标 styles; a narrow window keeps the header on its lines", async () => {
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
  const styled = answer.locator('[data-testid="citation"][data-styled="runs"]').first();
  await expect(styled.getByTestId("citation-chip")).toHaveText(/^\[\d(,\d)?\]$/);
  await expect(styled.getByTestId("citation-chip")).toHaveClass(/citation-marker--raised/);

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
  await references.getByTestId("references-style").click();
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
  const column = await window.getByTestId("mind-pane").boundingBox();
  const head = await references.locator(".references-head").boundingBox();
  if (!column || !head) throw new Error("The References or the Mind aren't on screen.");
  expect(head.x + head.width).toBeLessThanOrEqual(column.x + column.width + 1);
  await screenshot(window, "references-narrow-zh");
  await app.close();
});
