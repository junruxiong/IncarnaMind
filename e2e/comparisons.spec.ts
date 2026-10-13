import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  acceptPlainTextKeys,
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * Compare Documents (#98): from "/", the + menu or ⌘K, the sheet at the foot
 * of the Mind takes Documents and Questions and makes a comparison table,
 * which arrives as a change to accept. Its cells fill with cited Answers, a
 * busy provider waited out; each cell without one says why. "+ Question" adds
 * a column, Expand fills the card, a deleted Document's row says its source
 * was removed, and the table is there after a reload. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** Three suppliers' terms, a line each. */
const SUPPLIERS: Record<string, string> = {
  "Corvane Agreement.txt":
    "Corvane Agreement.\nTermination notice: 90 days.\nLiability cap: 12 months of fees.\n",
  "Halvard MSA.txt":
    "Halvard MSA.\nTermination notice: 6 months.\nLiability cap: twice the annual fees.\n",
  "Lumeric Order Form.txt":
    "Lumeric Order Form.\nTermination notice: 30 days.\nGoverning law: New York.\n",
};

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

const bridge = (window: Page) => ({
  language: (language: "en" | "zh-CN") =>
    window.evaluate(
      (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    ),
  /** Claude Haiku 5.5 on an Anthropic key, billed per token, allowed to be sent to. */
  cloudModel: () =>
    window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const provider = await core.saveChatProvider({
        kind: "anthropic",
        apiKey: "sk-ant-test",
        modelId: "claude-haiku-5-5",
      });
      if (provider.service) {
        await core.allowDataFlow("chat", provider.service.id);
        await core.allowDataFlow("tagging", provider.service.id);
      }
    }),
  documentId: (name: string) =>
    window.evaluate(
      async (wanted) =>
        (
          await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments()
        ).find((document) => document.name === wanted)?.id ?? "",
      name,
    ),
  deleteDocument: (id: string) =>
    window.evaluate(
      (documentId) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.deleteDocument(
          documentId,
        ),
      id,
    ),
});

async function addSuppliers(window: Page): Promise<void> {
  const paths: string[] = [];
  for (const [name, text] of Object.entries(SUPPLIERS)) {
    const path = join(sources, name);
    await writeFile(path, text);
    paths.push(path);
  }
  await addDocuments(window, paths);
}

/** A new Mind with a line of the User's, and the sheet opened from "/" on the line after it. */
async function openSheetFromSlash(window: Page): Promise<Locator> {
  await newMind(window);
  await window.keyboard.type("Shortlist: contract terms");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Three suppliers made the shortlist.");
  await window.keyboard.press("Enter");
  await window.keyboard.type("/compare");
  await expect(window.getByTestId("slash-item-compare")).toBeVisible();
  await window.keyboard.press("Enter");
  const sheet = window.getByTestId("compare-sheet");
  await expect(sheet).toBeVisible();
  // The first Question's field has the focus.
  await expect(sheet.getByTestId("compare-sheet-question").first()).toBeFocused();
  return sheet;
}

/** Types the Questions into the sheet, one per line, as a person does. */
async function typeQuestions(window: Page, questions: readonly string[]): Promise<void> {
  for (const [index, question] of questions.entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await window.keyboard.type(question);
  }
}

const cellTexts = (table: Locator) =>
  table
    .getByTestId("comparison-row")
    .evaluateAll((rows) =>
      rows.map((row) =>
        [...row.querySelectorAll('[data-testid="comparison-cell"]')].map((cell) =>
          (cell as HTMLElement).innerText.replace(/\s+/g, " ").trim(),
        ),
      ),
    );

test("Compare Documents from “/”: 3 Documents × 2 Questions fill as a change to accept, a busy provider waited out, every cell cited and checked; Accept all, Reject all and ⌘Z", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1440, 860);
  await acceptPlainTextKeys(window);
  await bridge(window).cloudModel();
  await addSuppliers(window);

  const sheet = await openSheetFromSlash(window);
  await expect(sheet.getByTestId("compare-sheet-counted")).toHaveText("3 of 3");
  await expect(sheet.getByTestId("compare-sheet-document")).toHaveCount(3);
  await typeQuestions(window, ["Termination notice", "Liability cap (busy)"]);
  // A cloud model billed per token: the estimate shows before the run.
  await expect(sheet.getByTestId("compare-sheet-note")).toContainText("About $0.0095 · 18 model calls");
  await screenshot(window, "compare-sheet-en-1440");
  await pointAndClick(window, sheet.getByTestId("compare-sheet-make"));

  const table = window.getByTestId("comparison-table");
  await expect(table).toBeVisible();
  await expect(sheet).toHaveCount(0);
  await expect(table).toHaveAttribute("data-pending", "true");
  await expect(table.getByTestId("comparison-title")).toHaveText(
    "Comparison table · 3 Documents × 2 Questions",
  );
  // The busy provider refused each "busy" cell once; the fill waits and tries again.
  await expect(table.getByTestId("comparison-progress")).toHaveCount(0, { timeout: 30_000 });
  expect(await cellTexts(table)).toEqual([
    ["90 days 1", "12 months of fees 2"],
    ["6 months 3", "twice the annual fees 4"],
    ["30 days 5", "! not found"],
  ]);
  await expect(table.getByTestId("comparison-row").first()).toContainText("Corvane Agreement");
  // Each cell's Citation is checked, as any Answer's, and none is drawn in the margin.
  await expect(table.locator('.citation-marker[data-check="found"]')).toHaveCount(5);
  await expect(window.getByTestId("margin-check")).toHaveCount(0);
  await screenshot(window, "compare-filled-en-1440");

  // Accept all keeps it; ⌘Z takes that back; Reject all takes it out; ⌘Z brings it back.
  await pointAndClick(window, table.getByTestId("comparison-accept-all"));
  await expect(table).not.toHaveAttribute("data-pending", "true");
  await expect(window.getByTestId("proposals-bar")).toHaveCount(0);
  await window.keyboard.press("ControlOrMeta+z");
  await expect(table).toHaveAttribute("data-pending", "true");
  await pointAndClick(window, table.getByTestId("comparison-reject-all"));
  await expect(window.getByTestId("comparison-table")).toHaveCount(0);
  await window.keyboard.press("ControlOrMeta+z");
  await expect(window.getByTestId("comparison-table")).toHaveAttribute("data-pending", "true");
  expect(await cellTexts(window.getByTestId("comparison-table"))).toHaveLength(3);
});

test("“+ Question” adds a column that fills; Expand fills the card with the Document column held; a deleted Document's row says so; it is there after a reload and in the export", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1600, 860);
  await useLocalChatModel(window);
  await addSuppliers(window);

  const sheet = await openSheetFromSlash(window);
  await typeQuestions(window, ["Termination notice"]);
  // A model on this computer: no estimate.
  await expect(sheet.getByTestId("compare-sheet-note")).toHaveText("");
  await window.keyboard.press("ControlOrMeta+Enter");
  const table = window.getByTestId("comparison-table");
  await expect(table.getByTestId("comparison-progress")).toHaveCount(0, { timeout: 30_000 });
  await pointAndClick(window, table.getByTestId("comparison-accept-all"));
  await expect(table).not.toHaveAttribute("data-pending", "true");

  // "+ Question": a column, written in place, then filled.
  await pointAndClick(window, table.getByTestId("comparison-add-question").locator("visible=true"));
  await expect(table.getByTestId("comparison-new-question")).toBeFocused();
  await window.keyboard.type("Governing law");
  await window.keyboard.press("Enter");
  await expect(table.getByTestId("comparison-question")).toHaveText([
    "Termination notice",
    "Governing law",
  ]);
  await expect(table.getByTestId("comparison-progress")).toHaveCount(0, { timeout: 30_000 });
  await expect
    .poll(() => cellTexts(table))
    .toEqual([
      ["90 days 1", "! not found"],
      ["6 months 2", "! not found"],
      ["30 days 3", "New York 4"],
    ]);

  // Expand: the table fills the card; Esc goes back to the note.
  const scroller = window.locator(".mind-scroller").first();
  await pointAndClick(window, table.getByTestId("comparison-expand"));
  await expect(table).toHaveAttribute("data-expanded", "true");
  await expect(table.getByTestId("comparison-back")).toBeVisible();
  const [card, expanded] = [await scroller.boundingBox(), await table.boundingBox()];
  if (!card || !expanded) throw new Error("The card or the table isn't on screen.");
  expect(Math.abs(expanded.x - card.x)).toBeLessThan(2);
  expect(Math.abs(expanded.y - card.y)).toBeLessThan(2);
  expect(expanded.height).toBeGreaterThan(card.height - 2);
  await screenshot(window, "compare-expanded-en-1600");
  await window.keyboard.press("Escape");
  await expect(table).not.toHaveAttribute("data-expanded", "true");

  // Scrolled sideways, the Document column stays where it was; the Questions move.
  await setWindowSize(app, window, 900, 860);
  for (const question of ["Liability cap", "Auto-renewal", "Price increase limit"]) {
    await pointAndClick(
      window,
      table.getByTestId("comparison-add-question").locator("visible=true"),
    );
    await window.keyboard.type(question);
    await window.keyboard.press("Enter");
  }
  await expect(table.getByTestId("comparison-progress")).toHaveCount(0, { timeout: 30_000 });
  const frame = table.locator(".comparison-scroll");
  const documentCell = table.locator(".comparison-document").first();
  const firstQuestion = table.getByTestId("comparison-question").first();
  // From its start: the last column's field had scrolled it to the end.
  await frame.evaluate((element) => {
    element.scrollLeft = 0;
  });
  await expect.poll(() => frame.evaluate((element) => element.scrollLeft)).toBe(0);
  const before = {
    document: await documentCell.boundingBox(),
    question: await firstQuestion.boundingBox(),
  };
  await frame.evaluate((element) => {
    element.scrollLeft = 200;
  });
  await expect
    .poll(async () => (await firstQuestion.boundingBox())?.x)
    .toBeLessThan((before.question?.x ?? 0) - 100);
  expect((await documentCell.boundingBox())?.x).toBe(before.document?.x);
  await screenshot(window, "compare-scrolled-en-900");

  // A deleted Document: its row says its source was removed.
  await bridge(window).deleteDocument(await bridge(window).documentId("Lumeric Order Form"));
  const lumeric = table.getByTestId("comparison-row").filter({ hasText: "Lumeric Order Form" });
  await expect(lumeric.getByTestId("comparison-source-removed")).toHaveText("source removed");

  // After a reload the table is there as it was.
  await window.reload();
  await expect(window.getByTestId("comparison-table")).toBeVisible({ timeout: 20_000 });
  expect((await cellTexts(window.getByTestId("comparison-table")))[0]?.slice(0, 2)).toEqual([
    "90 days 1",
    "! not found",
  ]);

  // The Markdown export holds it as a table, its Citations as footnotes.
  const markdown = await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const [mind] = await core.listMinds();
    const exported = await core.exportMind(mind?.id ?? "", { format: "markdown" });
    return new TextDecoder().decode(exported.data);
  });
  expect(markdown).toContain(
    "| Document | Termination notice | Governing law | Liability cap | Auto-renewal | Price increase limit |",
  );
  expect(markdown).toContain("| Corvane Agreement | 90 days[^1] | not found |");
  expect(markdown).toMatch(/\n\[\^1\]: Corvane Agreement, /);
});

test("empty states from the + menu, and the monthly cap: every cell left says it wasn't checked, in Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await bridge(window).language("zh-CN");
  await setWindowSize(app, window, 900, 860);
  await acceptPlainTextKeys(window);
  await bridge(window).cloudModel();

  // No Documents: the sheet says so, and nothing can be made.
  await newMind(window);
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-compare-documents"));
  const sheet = window.getByTestId("compare-sheet");
  await expect(sheet.getByTestId("compare-sheet-no-documents")).toContainText("这里还没有文档。");
  await expect(sheet.getByTestId("compare-sheet-make")).toBeDisabled();
  await pointAndClick(window, sheet.getByTestId("compare-sheet-close"));
  await expect(sheet).toHaveCount(0);

  // From the + menu once there are Documents: no Question yet, so it says what's missing.
  await addSuppliers(window);
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-compare-documents"));
  await expect(sheet).toBeVisible();
  await expect(sheet.getByTestId("compare-sheet-note")).toHaveText("每列一个问题，先添加一个。");
  await expect(sheet.getByTestId("compare-sheet-make")).toBeDisabled();
  await typeQuestions(window, ["Termination notice", "Liability cap"]);
  await screenshot(window, "compare-sheet-zh-900");

  // A cap too small for one request: nothing is sent, and every cell says why.
  await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { monthlyCap: { amount: 0.000001, currency: "USD" } },
    }),
  );
  await pointAndClick(window, sheet.getByTestId("compare-sheet-make"));
  const table = window.getByTestId("comparison-table");
  await expect(table.getByTestId("comparison-title")).toHaveText("对比表 · 3 个文档 × 2 个问题");
  await expect(table.getByTestId("comparison-progress")).toHaveCount(0, { timeout: 30_000 });
  await expect(table.getByTestId("comparison-cell-problem")).toHaveText(
    Array(6).fill("!未核对：已达花费上限"),
  );
  await expect(table.getByTestId("comparison-fill")).toHaveText("补全 6 个单元格");
  // The table says what stopped it, once, with the one thing that helps.
  await expect(table.getByTestId("comparison-stopped")).toContainText("已达到每月花费上限");
  await expect(table.getByTestId("comparison-stopped")).toContainText("更改上限");
  await screenshot(window, "compare-cost-cap-zh-900");
});
