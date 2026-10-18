import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  addDocuments,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  nameMind,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
} from "./app";

/*
 * "Make a bar chart of revenue by quarter" (#1091), in English and Chinese: the
 * Mind gets the cited table and a chart drawn from it, right after it, the
 * facts line says both, and one ⌘Z takes both back. The model is the smoke
 * tests' fake.
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

const REPORT = [
  "# Annual report",
  "",
  "Revenue in Q1 was €120 million.",
  "",
  "Revenue in Q2 was €135 million.",
  "",
  "Revenue in Q3 was €150 million.",
  "",
  "Revenue in Q4 was €160 million.",
].join("\n");

for (const [language, request, kind, title, fact] of [
  [
    "en",
    "Make a bar chart of revenue by quarter",
    "bar",
    "Revenue by quarter",
    "Added a table and a chart",
  ],
  ["zh-CN", "画一个各季度收入的折线图", "line", "各季度收入", "添加了 1 个表格和 1 个图表"],
] as const) {
  test(`“${request}” adds the cited table and its ${kind} chart, and one Undo takes both back (${language})`, async () => {
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1440, 900);
    await writeFile(join(sources, "Annual report.md"), REPORT);
    await addDocuments(window, [join(sources, "Annual report.md")]);
    await newMind(window);
    await nameMind(window, "Year notes");
    await window.keyboard.type("Notes on the year.");

    await askInAssistant(window, request);

    const editor = window.getByTestId("mind-editor");
    const table = editor.locator("table");
    const chart = editor.getByTestId("chart");
    await expect(table).toHaveCount(1);
    await expect(chart).toHaveCount(1);
    const tableBox = await table.boundingBox();
    const chartBox = await chart.boundingBox();
    expect(tableBox && chartBox && tableBox.y + tableBox.height <= chartBox.y).toBe(true);
    await expect(chart.getByTestId("chart-title")).toContainText(title);
    await expect(chart.getByTestId("chart-plot")).toHaveAttribute("data-kind", kind);
    await expect(chart.getByTestId("chart-point")).toHaveCount(4);
    await expect(chart.getByTestId("chart-source")).toContainText("Annual report");
    await expect(editor.getByTestId("applied-line").getByTestId("applied-count")).toHaveText(fact);

    await editor.focus();
    await window.keyboard.press("ControlOrMeta+z");
    await expect(chart).toHaveCount(0);
    await expect(editor.locator("table")).toHaveCount(0);
    await expect(editor).toContainText("Notes on the year.");
  });
}
