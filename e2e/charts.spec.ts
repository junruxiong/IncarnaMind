import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { footnotesOf, paragraphsOf, part, unzip } from "../tests/helpers/docx";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  addToNote,
  askInAssistant,
  clickEditItem,
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
 * Charts (#436): a bar chart made from an Answer's cited table, from the
 * table's menu; hovering a value shows its card with its Citation, whose
 * Open shows the cited page with the quote washed; a column turned off and
 * on from the chart's menu, each undone with ⌘Z; and "Show as table", every
 * value with its check. Then all of it with the keyboard only. In English and
 * Chinese, at a narrow and a wide window. Set INCARNAMIND_SCREENSHOTS to a
 * folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(window: Page, name: string): Promise<void> {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** Acme's quarterly revenue, a year a page. */
const REPORT = buildPdf([
  { lines: ["Revenue by quarter, 2024", "Quarterly revenue was 289.4, 301.2, 309.8 and 330.6."] },
  { lines: ["Revenue by quarter, 2025", "Quarterly revenue was 341.5, 362.0 and 377.9."] },
]);

const WORDS = {
  en: {
    question: "Revenue by quarter?",
    chart: "Chart",
    chartFromTable: "Chart from table",
    bar: "Bar",
    line: "Line",
    showAsTable: "Show as table",
    showAsChart: "Show as chart",
    title: "Revenue by quarter ($ million)",
    menu: [
      "Show as table",
      "Show as line chart",
      "Show as pie chart",
      "Show as scatter chart",
      "Rename",
      "Delete",
    ],
    share: "of 2025",
    source: "values, each quoted from Acme",
    uncited: "1 with no Citation",
    noCitation: "No Citation",
  },
  "zh-CN": {
    question: "按季度列出营业收入 revenue quarter",
    chart: "图表",
    chartFromTable: "用表格生成图表",
    bar: "柱状图",
    line: "折线图",
    showAsTable: "以表格显示",
    showAsChart: "以图表显示",
    title: "分季度营业收入（百万美元）",
    menu: ["以表格显示", "显示为折线图", "显示为饼图", "显示为散点图", "重命名", "删除"],
    share: "占 2025 的",
    source: "个数值，均引自“Acme”",
    uncited: "1 个数值没有引用",
    noCitation: "没有引用",
  },
} as const;

type Language = keyof typeof WORDS;

let dataDir: string;
let sources: string;
let home: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
  home = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
  await removeDataFolder(home);
});

const setLanguage = (window: Page, language: Language) =>
  window.evaluate(
    (value) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: value },
      }),
    language,
  );

/** The app, in the language and at the width given, with the report to cite added once a model is set. */
async function openApp(language: Language, width: number) {
  await writeFile(join(sources, "Acme.pdf"), REPORT);
  const running = await launchApp(dataDir, { fakeChat: true, home });
  await dismissChatSetup(running.window);
  await setLanguage(running.window, language);
  await setWindowSize(running.app, running.window, width, 660);
  return running;
}

/** A Mind holding an Answer's table of quarterly figures, each cited but the last, added from the assistant. */
async function askForFigures(window: Page, language: Language) {
  await addDocuments(window, [join(sources, "Acme.pdf")]);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, editor);
  await addToNote(window, await askInAssistant(window, WORDS[language].question));
  const table = editor.locator("table");
  await expect(table.locator("tr")).toHaveCount(5);
  await expect(table.getByTestId("citation")).toHaveCount(7);
  return { editor, table };
}

/** The value of the chart at this row and column of its data. */
const point = (chart: Locator, row: number, series: number) =>
  chart.locator(`[data-testid="chart-point"][data-point="${row}:${series}"]`);

/**
 * Points at another value as a person does: off the card first, so it goes,
 * as it may lie over the value next to the one it is for.
 */
async function pointAt(window: Page, value: Locator): Promise<void> {
  await window.mouse.move(0, 0);
  await expect(window.getByTestId("chart-card")).toHaveCount(0);
  await value.hover();
}

/** Points inside a pie's slice, as a person does: the middle of its box can lie in a neighbour. */
async function pointAtSlice(window: Page, slice: Locator): Promise<void> {
  await window.mouse.move(0, 0);
  await expect(window.getByTestId("chart-card")).toHaveCount(0);
  const at = await slice.evaluate((element) => {
    const path = element as SVGPathElement;
    // The slice is drawn from the centre, out along a radius, round its arc and back.
    const centre = path.getPointAtLength(0);
    const arc = path.getPointAtLength(path.getTotalLength() / 2);
    const screen = path.getScreenCTM();
    if (!screen) return null;
    const inside = new DOMPoint(
      centre.x + (arc.x - centre.x) * 0.6,
      centre.y + (arc.y - centre.y) * 0.6,
    ).matrixTransform(screen);
    return { x: inside.x, y: inside.y };
  });
  if (!at) throw new Error("The slice isn't drawn.");
  await window.mouse.move(at.x, at.y, { steps: 5 });
}

/** Waits until the two boxes don't overlap: a card that grows in place settles first. */
async function expectApart(a: Locator, b: Locator): Promise<void> {
  await expect
    .poll(async () => {
      const [one, two] = [await a.boundingBox(), await b.boundingBox()];
      if (!one || !two) return "not on screen";
      const apart =
        one.x + one.width <= two.x ||
        two.x + two.width <= one.x ||
        one.y + one.height <= two.y ||
        two.y + two.height <= one.y;
      return apart ? "apart" : "overlapping";
    })
    .toBe("apart");
}

/** The bottom of a box, and its right edge. */
async function boxOf(locator: Locator) {
  const box = await locator.boundingBox();
  if (!box) throw new Error("It isn't on screen.");
  return box;
}

for (const [language, width] of [
  ["en", 1600],
  ["zh-CN", 1100],
] as const) {
  test(`a bar chart from a cited table: hover a value for its Citation, open its page, change a column, show it as a table (${language}, ${width}px)`, async () => {
    const words = WORDS[language];
    const { app, window } = await openApp(language, width);
    await useLocalChatModel(window);
    const { editor, table } = await askForFigures(window, language);

    // The table's menu makes the chart, inserted after the Answer.
    await pointAndClick(window, table.locator("td").filter({ hasText: /^Q2$/ }));
    await pointAndClick(window, window.getByTestId("table-menu"));
    const chartItem = window.getByTestId("table-chart");
    await expect(chartItem).toHaveText(words.chartFromTable);
    await pointAndClick(window, chartItem);
    const chart = editor.getByTestId("chart");
    await expect(chart).toHaveCount(1);
    await expect(chart.getByTestId("chart-title")).toHaveText(words.chart);
    await expect(chart.getByTestId("chart-kind")).toContainText(words.bar);
    const plot = chart.getByTestId("chart-plot");
    await expect(plot).toHaveAttribute("data-kind", "bar");
    await expect(chart.getByTestId("chart-point")).toHaveCount(8);
    // Drawn at the width it has: inside the Mind's column, its words 12 px.
    const column = await boxOf(editor);
    const drawn = await boxOf(plot);
    expect(drawn.x).toBeGreaterThanOrEqual(column.x);
    expect(drawn.x + drawn.width).toBeLessThanOrEqual(column.x + column.width + 1);
    await expect(chart.locator(".chart-axis").first()).toHaveCSS("font-size", "12px");
    // Renamed in place: a double-click, typing, Enter; F2 opens the same field, and Esc leaves it.
    await chart.getByTestId("chart-title").dblclick();
    const field = chart.getByTestId("chart-title-field");
    await expect(field).toBeFocused();
    await window.keyboard.type(words.title);
    await window.keyboard.press("Enter");
    await expect(chart.getByTestId("chart-title")).toHaveText(`${words.chart} · ${words.title}`);
    await chart.getByTestId("chart-show-table").focus();
    await window.keyboard.press("F2");
    await expect(field).toHaveValue(words.title);
    await window.keyboard.press("Escape");
    await expect(chart.getByTestId("chart-title")).toHaveText(`${words.chart} · ${words.title}`);
    await expect(window.locator("dialog[open]")).toHaveCount(0);

    // Its right-click menu: its own actions, Rename, then Delete last.
    await chart.locator(".chart-head").click({ button: "right", position: { x: 4, y: 4 } });
    const actions = window.getByTestId("chart-actions");
    await expect(actions).toBeVisible();
    // Each item, then its keys, if it has any.
    await expect(actions.getByRole("menuitem")).toHaveText(
      words.menu.map((label) => new RegExp(`^${label}`)),
    );
    await window.keyboard.press("Escape");
    await expect(actions).toBeHidden();

    // The key to its greys, and where its values came from, the one with none said quietly.
    await expect(chart.getByTestId("chart-legend")).toHaveText(/2024\s*2025/);
    await expect(chart.getByTestId("chart-source")).toContainText(words.source);
    await expect(chart.getByTestId("chart-source")).toContainText(words.uncited);

    // Hovering a value shows its card: as written, with its Citation and Open.
    const q3 = point(chart, 2, 1);
    await q3.hover();
    const card = window.getByTestId("chart-card");
    await expect(card).toBeVisible();
    await expect(card.getByTestId("chart-card-value")).toHaveText("377.9");
    await expect(card.getByTestId("chart-card-source")).toContainText("Acme");
    await expect(card.getByTestId("chart-check")).toHaveAttribute("data-check", "found");
    await expect(q3).toHaveAttribute("data-active", "true");
    // The card stays in the plot's frame: off the chart's head and its "Bar ⌄", and off its foot.
    await expectApart(card, chart.getByTestId("chart-kind"));
    await expectApart(card, chart.getByTestId("chart-foot"));
    await screenshot(window, `chart-point-${language}-${width}`);

    // So does the tallest value's, beside it under the head: its "1 with no Citation" foot shows.
    await pointAt(window, point(chart, 3, 1));
    await expect(card.getByTestId("chart-card-value")).toHaveText("401.2");
    await expectApart(card, chart.getByTestId("chart-kind"));
    await expectApart(card, chart.getByTestId("chart-show-table"));
    await expectApart(card, chart.getByTestId("chart-foot"));

    // A value with no Citation says so quietly, with nothing to open.
    await pointAt(window, point(chart, 3, 1));
    await expect(card.getByTestId("chart-card-value")).toHaveText("401.2");
    await expect(card.getByTestId("chart-card-source")).toHaveText(words.noCitation);
    await expect(card.getByTestId("chart-card-open")).toHaveCount(0);

    // Open shows the cited page, the quote washed, as a Citation's Open does.
    await pointAt(window, q3);
    await expect(card.getByTestId("chart-card-value")).toHaveText("377.9");
    await card.getByTestId("chart-card-open").click();
    await expect(card).toHaveCount(0);
    await expect(window.getByTestId("viewer-title")).toHaveText("Acme");
    await expect(
      window.getByTestId("viewer").locator("[data-quote-highlight]").first(),
    ).toBeVisible();
    // In a narrow window the page took the Mind's place, as a tab beside it: back to the Mind.
    if (!(await chart.isVisible())) await pointAndClick(window, window.getByTestId("mind-tab"));

    // A column off, from the chart's menu: only 2025 is drawn, in the secondary ink alone.
    await pointAndClick(window, chart.getByTestId("chart-kind"));
    const menu = window.getByTestId("chart-kind-menu");
    await expect(menu.getByTestId("chart-series")).toHaveText([/2024/, /2025/]);
    await pointAndClick(window, menu.getByTestId("chart-series").first());
    await expect(chart.getByTestId("chart-point")).toHaveCount(4);
    await expect(chart.getByTestId("chart-legend")).toHaveCount(0);
    // ⌘Z brings it back, from the chart's own button, and so does the Edit menu's Undo.
    await expect(chart.getByTestId("chart-kind")).toBeFocused();
    await window.keyboard.press("ControlOrMeta+z");
    await expect(chart.getByTestId("chart-point")).toHaveCount(8);
    await pointAndClick(window, chart.getByTestId("chart-kind"));
    await pointAndClick(window, menu.getByTestId("chart-series").first());
    await expect(chart.getByTestId("chart-point")).toHaveCount(4);
    await expect(chart.getByTestId("chart-kind")).toBeFocused();
    await clickEditItem(app, "edit.undo");
    await expect(chart.getByTestId("chart-point")).toHaveCount(8);

    // A line chart, from the same menu, undone too.
    await pointAndClick(window, chart.getByTestId("chart-kind"));
    await pointAndClick(window, menu.getByTestId("chart-kind-line"));
    await expect(plot).toHaveAttribute("data-kind", "line");
    await expect(chart.getByTestId("chart-kind")).toContainText(words.line);
    await window.keyboard.press("ControlOrMeta+z");
    await expect(plot).toHaveAttribute("data-kind", "bar");

    // A pie and a scatter chart: the foot counts the same cells as the bars did. A pie's card
    // says which column its slices are, and of what its share is.
    const foot = (await chart.getByTestId("chart-source").textContent()) ?? "";
    await pointAndClick(window, chart.getByTestId("chart-kind"));
    await pointAndClick(window, menu.getByTestId("chart-kind-pie"));
    await expect(plot).toHaveAttribute("data-kind", "pie");
    await expect(chart.getByTestId("chart-source")).toHaveText(foot);
    await pointAtSlice(window, point(chart, 2, 1));
    await expect(card.getByTestId("chart-card-body")).toContainText("Q3 · 2025");
    await expect(card.getByTestId("chart-card-body")).toContainText(words.share);
    await expectApart(card, chart.getByTestId("chart-kind"));
    await expectApart(card, chart.getByTestId("chart-foot"));
    await pointAndClick(window, chart.getByTestId("chart-kind"));
    await pointAndClick(window, menu.getByTestId("chart-kind-scatter"));
    await expect(plot).toHaveAttribute("data-kind", "scatter");
    await expect(chart.getByTestId("chart-source")).toHaveText(foot);
    await pointAt(window, point(chart, 3, 1));
    await expect(card.getByTestId("chart-card-value")).toContainText("401.2");
    await expectApart(card, chart.getByTestId("chart-kind"));
    await expectApart(card, chart.getByTestId("chart-foot"));
    await window.mouse.move(0, 0);
    await window.keyboard.press("ControlOrMeta+z");
    await window.keyboard.press("ControlOrMeta+z");
    await expect(plot).toHaveAttribute("data-kind", "bar");

    // Show as table: every value as written, with its check; the one with none marked so.
    const toggle = chart.getByTestId("chart-show-table");
    await expect(toggle).toHaveText(words.showAsTable);
    await pointAndClick(window, toggle);
    await expect(toggle).toHaveText(words.showAsChart);
    const values = chart.getByTestId("chart-table");
    await expect(values.locator("tbody tr")).toHaveCount(4);
    await expect(values.getByTestId("chart-value")).toHaveCount(8);
    await expect(values.locator('[data-check="found"]')).toHaveCount(7);
    await expect(values.locator('[data-check="none"]')).toHaveCount(1);
    await pointAt(window, values.getByTestId("chart-value").nth(5));
    await expect(card.getByTestId("chart-card-value")).toHaveText("377.9");
    await screenshot(window, `chart-table-${language}-${width}`);
    await app.close();
  });
}

/** Presses Tab until the focus is inside `within`, at most `tries` times. */
async function tabInto(window: Page, within: Locator, tries = 40): Promise<void> {
  for (let press = 0; press < tries; press++) {
    if (await within.evaluate((element) => element.contains(document.activeElement))) return;
    await window.keyboard.press("Tab");
  }
  throw new Error("Tab never reached it.");
}

for (const [language, width] of [
  ["en", 1100],
  ["zh-CN", 1600],
] as const) {
  test(`the same with the keyboard only (${language}, ${width}px)`, async () => {
    const words = WORDS[language];
    const { app, window } = await openApp(language, width);
    await useLocalChatModel(window);
    const { editor } = await askForFigures(window, language);

    // From the composer back to the Mind, and up into the table's last row.
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.press("Escape");
    await editor.focus();
    await expect(editor).toBeFocused();
    await window.keyboard.press("ControlOrMeta+End");
    for (let press = 0; press < 12; press++) {
      const inCell = await window.evaluate(() =>
        Boolean(document.getSelection()?.anchorNode?.parentElement?.closest("td, th")),
      );
      if (inCell) break;
      await window.keyboard.press("ArrowUp");
    }
    // ⌘/ opens the table's menu, where Chart from table is, once the editor has the cursor in
    // the cell (its ⋯ shows there).
    await expect(window.getByTestId("table-menu")).toBeVisible();
    await window.keyboard.press("ControlOrMeta+/");
    const chartItem = window.getByTestId("table-chart");
    await expect(chartItem).toBeVisible();
    for (let press = 0; press < 12; press++) {
      if (await chartItem.evaluate((element) => element === document.activeElement)) break;
      await window.keyboard.press("ArrowDown");
    }
    await window.keyboard.press("Enter");
    const chart = editor.getByTestId("chart");
    await expect(chart.getByTestId("chart-point")).toHaveCount(8);

    // Tab reaches the chart's controls, then its values as one stop.
    await tabInto(window, chart);
    await expect(chart.getByTestId("chart-kind")).toBeFocused();
    await window.keyboard.press("Tab");
    await expect(chart.getByTestId("chart-show-table")).toBeFocused();
    await window.keyboard.press("Tab");
    const first = point(chart, 0, 0);
    await expect(first).toBeFocused();
    const card = window.getByTestId("chart-card");
    await expect(card.getByTestId("chart-card-value")).toHaveText("289.4");
    // Each value has an accessible name: its label, value and source.
    await expect(first).toHaveAttribute("aria-label", /Q1 · 2024.*289\.4.*Acme/);
    // Right along its series, Down to the next series at the same row.
    await window.keyboard.press("ArrowRight");
    await window.keyboard.press("ArrowRight");
    await expect(point(chart, 2, 0)).toBeFocused();
    await window.keyboard.press("ArrowDown");
    await expect(point(chart, 2, 1)).toBeFocused();
    await expect(card.getByTestId("chart-card-value")).toHaveText("377.9");
    // Enter opens its page, the quote washed.
    await window.keyboard.press("Enter");
    await expect(window.getByTestId("viewer-title")).toHaveText("Acme");
    await expect(
      window.getByTestId("viewer").locator("[data-quote-highlight]").first(),
    ).toBeVisible();
    // Back to the chart: in a narrow window the page took the Mind's place, as a tab (⌘1 goes back).
    if (!(await chart.isVisible())) await window.keyboard.press("ControlOrMeta+1");
    if (await point(chart, 2, 1).evaluate((element) => element === document.activeElement)) {
      await window.keyboard.press("Escape");
      await window.keyboard.press("Shift+Tab");
      await window.keyboard.press("Shift+Tab");
    } else {
      await tabInto(window, chart);
    }
    await expect(chart.getByTestId("chart-kind")).toBeFocused();

    // The kind's menu, by keyboard: a column off, then back on.
    await window.keyboard.press("Enter");
    const series = window.getByTestId("chart-kind-menu").getByTestId("chart-series");
    await expect(series).toHaveCount(2);
    for (let press = 0; press < 12; press++) {
      if (await series.first().evaluate((element) => element === document.activeElement)) break;
      await window.keyboard.press("ArrowDown");
    }
    await window.keyboard.press("Enter");
    await expect(chart.getByTestId("chart-point")).toHaveCount(4);
    await window.keyboard.press("ControlOrMeta+z");
    await expect(chart.getByTestId("chart-point")).toHaveCount(8);

    // Show as table, by keyboard; its values one Tab stop, the arrows between them.
    await expect(chart.getByTestId("chart-kind")).toBeFocused();
    await window.keyboard.press("Tab");
    await expect(chart.getByTestId("chart-show-table")).toBeFocused();
    await window.keyboard.press("Enter");
    await expect(chart.getByTestId("chart-show-table")).toHaveText(words.showAsChart);
    await window.keyboard.press("Tab");
    const values = chart.getByTestId("chart-value");
    await expect(values.first()).toBeFocused();
    await window.keyboard.press("ArrowDown");
    await window.keyboard.press("ArrowRight");
    await expect(values.nth(3)).toBeFocused();
    await expect(card.getByTestId("chart-card-value")).toHaveText("362.0");
    await window.keyboard.press("Escape");
    await expect(card).toHaveCount(0);
    await app.close();
  });
}

test("a chart exports to Word as a picture the app draws, with its caption and its sources as footnotes", async () => {
  const { app, window } = await openApp("en", 1600);
  await useLocalChatModel(window);
  const { editor, table } = await askForFigures(window, "en");
  await pointAndClick(window, table.locator("td").filter({ hasText: /^Q1$/ }));
  await pointAndClick(window, window.getByTestId("table-menu"));
  await pointAndClick(window, window.getByTestId("table-chart"));
  await expect(editor.getByTestId("chart").getByTestId("chart-point")).toHaveCount(8);

  // The file's bytes, as Export writes them.
  const bytes = await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const [mind] = await bridge.listMinds();
    if (!mind) throw new Error("No Mind.");
    const exported = await bridge.exportMind(mind.id, { format: "docx" });
    return Array.from(exported.data);
  });
  const files = unzip(Uint8Array.from(bytes));
  // The chart's picture: a PNG twice the page's width, drawn in a hidden window, not blank.
  const picture = files.get("word/media/image1.png");
  if (!picture) throw new Error("The Word file has no picture.");
  expect(picture.readUInt32BE(0)).toBe(0x89504e47);
  expect(picture.readUInt32BE(16)).toBe(1200);
  expect(picture.length).toBeGreaterThan(5_000);
  if (SCREENSHOTS) await writeFile(join(SCREENSHOTS, "chart-word-picture.png"), picture);
  const document = part(files, "word/document.xml");
  expect(document).toContain('descr="Bar chart"');
  const captions = paragraphsOf(document).filter((paragraph) => paragraph.style === "Caption");
  expect(captions.map((paragraph) => paragraph.text)).toEqual([
    "Chart",
    expect.stringMatching(/^Source: \[\d+\] 1 value has no Citation\.$/),
  ]);
  expect(Object.values(footnotesOf(part(files, "word/footnotes.xml")))).toContain("Acme, p. 1");
  await app.close();
});
