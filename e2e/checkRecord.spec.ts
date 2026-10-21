import { createHash } from "node:crypto";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { parseCsv } from "../tests/helpers/csv";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  nameMind,
  newMind,
  pointAndClick,
  pointAt,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
  useOutputLocation,
} from "./app";

/*
 * Export check record (#738): from a Mind's ⋯ menu and from ⌘K, a record of what was checked in
 * it, as a CSV for the User's records and a PDF to file with the work, saved in the Output
 * location. The scripted model (src/main/fakeChat/checks.ts) finds a sentence supported when it
 * shares three words with a line of a Document, and otherwise unsupported. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also keep the files.
 */

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);
const SENTENCES = [
  "Spring tides happen at new and full moon.",
  "Penguins nest on the harbour cliffs.",
];

const TEXTS = {
  en: {
    csv: "Export check record as CSV",
    pdf: "Export check record as PDF",
    csvFile: "Tides - check record.csv",
    pdfFile: "Tides - check record.pdf",
    columns: ["Kind", "Sentence", "Result"],
    supported: "Supported",
    unsupported: "Unsupported",
    sentence: "Sentence",
    model: "Model",
    service: "Service",
    checked: "Checked (UTC)",
    thisComputer: "On this computer",
  },
  "zh-CN": {
    csv: "将核对记录导出为 CSV",
    pdf: "将核对记录导出为 PDF",
    csvFile: "Tides - 核对记录.csv",
    pdfFile: "Tides - 核对记录.pdf",
    columns: ["类型", "句子", "核对结果"],
    supported: "有依据",
    unsupported: "无依据",
    sentence: "句子",
    model: "模型",
    service: "服务",
    checked: "核对时间（UTC）",
    thisComputer: "在这台电脑上",
  },
} as const;

let dataDir: string;
let sources: string;
let out: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
  out = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
  await removeDataFolder(out);
});

/** The file the result line says it saved. */
async function saved(window: Page, name: string): Promise<string> {
  const done = window.getByTestId("export-result").getByTestId("export-done");
  await expect(done).toContainText(name, { timeout: 30_000 });
  await expect(done).toContainText(`${basename(out)}`);
  return (await done.getAttribute("title")) ?? "";
}

for (const language of ["en", "zh-CN"] as const) {
  test(`the Mind's menu and ⌘K save the check record as CSV and as PDF, listing each checked sentence (${language})`, async () => {
    const text = TEXTS[language];
    await writeFile(join(sources, "Tides.pdf"), TIDES);
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1440, 900);
    await addDocuments(window, [join(sources, "Tides.pdf")]);
    await newMind(window);
    await nameMind(window, "Tides");
    const editor = window.getByTestId("mind-editor").filter({ visible: true });
    await editor.click();
    await window.keyboard.type(SENTENCES.join(" "));
    const line = editor.locator(":scope > p").first();
    await slideSelect(window, line, "Spring tides", { line, words: "harbour cliffs." });
    await pointAndClick(
      window,
      window.getByTestId("selection-menu").getByTestId("selection-action-find-support"),
    );
    const marks = window.getByTestId("claim-mark");
    await expect(marks.and(window.locator("[data-check=supported]"))).toHaveCount(1);
    await expect(marks.and(window.locator("[data-check=unsupported]"))).toHaveCount(1);
    await useOutputLocation(window, out);

    // From the Mind's ⋯ menu, as a CSV.
    const row = window.getByTestId("mind-row").filter({ hasText: "Tides" });
    await pointAt(window, row);
    await pointAndClick(window, row.getByTestId("mind-menu"));
    const menu = row.getByTestId("mind-actions");
    await expect(menu.getByTestId("mind.checkRecord.csv")).toHaveText(text.csv);
    await expect(menu.getByTestId("mind.checkRecord.pdf")).toHaveText(text.pdf);
    await pointAndClick(window, menu.getByTestId("mind.checkRecord.csv"));
    const csv = await readFile(await saved(window, text.csvFile), "utf8");

    const rows = parseCsv(csv);
    const at = rows.findIndex((each) => each[0] === text.columns[0] && each[1] === text.columns[1]);
    expect(at).toBeGreaterThan(0);
    const [header = [], ...items] = rows.slice(at);
    const column = (name: string) => header.indexOf(name);
    expect(items.map((each) => [each[1], each[2]])).toEqual([
      [SENTENCES[0], text.supported],
      [SENTENCES[1], text.unsupported],
    ]);
    // The supported sentence names the Document version it rests on: the file's own fingerprint.
    const sha = createHash("sha256").update(TIDES).digest("hex");
    expect(items[0]?.[column("SHA-256")]).toBe(sha);
    expect(items[0]?.[column(text.model)]).toBe("fake-model");
    expect(items[0]?.[column(text.service)]).toBe(text.thisComputer);
    expect(items[0]?.[column(text.checked)]).toMatch(/^\d{4}-\d{2}-\d{2}T[\d:.]+Z$/);
    expect(items[1]?.[column("SHA-256")]).toBe("");

    // From ⌘K, as a PDF.
    await window.keyboard.press("ControlOrMeta+k");
    await expect(window.getByTestId("palette-search")).toBeFocused();
    await window.keyboard.type(">");
    const item = window.getByTestId("palette").locator('[data-key="command:mind.checkRecord.pdf"]');
    await expect(item).toContainText(text.pdf);
    await pointAndClick(window, item);
    const pdf = await readFile(await saved(window, text.pdfFile));
    expect(pdf.subarray(0, 5).toString("latin1")).toBe("%PDF-");
    expect((await readdir(out)).sort()).toEqual([text.csvFile, text.pdfFile].sort());

    const keep = process.env.INCARNAMIND_SCREENSHOTS;
    if (keep) {
      await writeFile(join(keep, `check-record-${language}.pdf`), pdf);
      await writeFile(join(keep, `check-record-${language}.csv`), csv);
    }
    await app.close();
  });
}
