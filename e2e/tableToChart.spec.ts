import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * A table in a Mind and its chart (#704): "Chart from table" puts the years of a table of years
 * by metrics on the x axis, with the metrics as its series; "Switch rows and columns" swaps them,
 * from the chart's menu and from its right-click menu, and ⌘Z puts them back; a calculation's
 * working table is offered no chart, in its menu or in ⌘K; and a plain-text copy of a Mind with
 * a table and its chart has the table once, tab-separated, while the HTML copy keeps both. In
 * English and Chinese, at a wide and a narrow window. Set INCARNAMIND_SCREENSHOTS to a folder to
 * also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(window: Page, name: string): Promise<void> {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const WORDS = {
  en: {
    years: {
      head: ["Metric", "2023", "2024", "2025"],
      rows: [
        ["Revenue", "289.4", "301.2", "330.6"],
        ["Net income", "41.2", "45.0", "52.3"],
      ],
    },
    working: {
      head: ["Step", "Formula", "Result"],
      rows: [
        ["Revenue growth", "(377.9 − 330.6) ÷ 330.6", "14.3%"],
        ["Net margin", "41.2 ÷ 377.9", "10.9%"],
      ],
    },
    firstMetric: "Revenue",
    metrics: ["Revenue", "Net income"],
    years2: ["2023", "2024", "2025"],
    chartFromTable: "Chart from table",
    switch: "Switch rows and columns",
    lead: "Revenue by year",
  },
  "zh-CN": {
    years: {
      head: ["指标", "2023年", "2024年", "2025年"],
      rows: [
        ["营业收入", "289.4", "301.2", "330.6"],
        ["净利润", "41.2", "45.0", "52.3"],
      ],
    },
    working: {
      head: ["步骤", "公式", "结果"],
      rows: [
        ["营业收入同比", "(377.9 − 330.6) ÷ 330.6", "14.3%"],
        ["净利率", "41.2 ÷ 377.9", "10.9%"],
      ],
    },
    firstMetric: "营业收入",
    metrics: ["营业收入", "净利润"],
    years2: ["2023年", "2024年", "2025年"],
    chartFromTable: "用表格生成图表",
    switch: "行列互换",
    lead: "按年度列出的营业收入",
  },
} as const;

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const tableHtml = (table: { head: readonly string[]; rows: readonly (readonly string[])[] }) =>
  `<table><tr>${table.head.map((each) => `<th>${each}</th>`).join("")}</tr>${table.rows
    .map((row) => `<tr>${row.map((each) => `<td>${each}</td>`).join("")}</tr>`)
    .join("")}</table>`;

/** A new Mind with a line of text and this table pasted in, as from a spreadsheet. */
async function mindWithTable(
  window: Page,
  lead: string,
  table: { head: readonly string[]; rows: readonly (readonly string[])[] },
) {
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, editor);
  await window.keyboard.type(lead);
  await window.keyboard.press("Enter");
  await window.evaluate((html) => {
    const data = new DataTransfer();
    data.setData("text/html", html);
    data.setData("text/plain", "table");
    (document.activeElement ?? document.body).dispatchEvent(
      new ClipboardEvent("paste", { clipboardData: data, bubbles: true }),
    );
  }, tableHtml(table));
  const grid = editor.locator("table");
  await expect(grid.locator("tr")).toHaveCount(table.rows.length + 1);
  return { editor, grid };
}

/** The labels along the chart's x axis and up its y axis, as drawn. */
const axisLabels = (chart: Locator) => chart.locator(".chart-axis").allTextContents();

/** The value of the chart at this row and series of its data. */
const point = (chart: Locator, row: number, series: number) =>
  chart.locator(`[data-testid="chart-point"][data-point="${row}:${series}"]`);

/** What the editor's copy puts on the clipboard for the selection, read from its own event. */
const copyPayload = (editor: Locator) =>
  editor.evaluate((dom) => {
    const data = new DataTransfer();
    dom.dispatchEvent(
      new ClipboardEvent("copy", { clipboardData: data, bubbles: true, cancelable: true }),
    );
    return { html: data.getData("text/html"), text: data.getData("text/plain") };
  });

for (const [language, width] of [
  ["en", 1600],
  ["zh-CN", 1100],
] as const) {
  test(`a table of years by metrics charts with the years on the x axis; Switch rows and columns swaps them and ⌘Z puts them back; copied as text it has the table once (${language}, ${width}px)`, async () => {
    const words = WORDS[language];
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, width, 660);
    const { editor, grid } = await mindWithTable(window, words.lead, words.years);

    // ⌘K finds Chart from table with the cursor in the table, and the table's menu makes the chart
    // after the table.
    const cell = grid.locator("td").filter({ hasText: words.firstMetric });
    await pointAndClick(window, cell);
    await window.keyboard.press("ControlOrMeta+k");
    await expect(window.getByTestId("palette")).toBeVisible();
    await window.keyboard.type(words.chartFromTable);
    await expect(window.getByTestId("palette-item").first()).toContainText(words.chartFromTable);
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("palette")).toBeHidden();
    await pointAndClick(window, cell);
    await pointAndClick(window, window.getByTestId("table-menu"));
    const chartItem = window.getByTestId("table-chart");
    await expect(chartItem).toHaveText(words.chartFromTable);
    await pointAndClick(window, chartItem);
    const chart = editor.getByTestId("chart");
    await expect(chart).toHaveCount(1);
    await expect(chart.getByTestId("chart-point")).toHaveCount(6);

    // The years are along the x axis and the metrics are the series, in the key.
    expect(await axisLabels(chart)).toEqual(expect.arrayContaining([...words.years2]));
    await expect(chart.getByTestId("chart-legend")).toContainText(words.metrics[0]);
    await expect(chart.getByTestId("chart-legend")).toContainText(words.metrics[1]);
    await expect(point(chart, 0, 0)).toHaveAttribute(
      "aria-label",
      new RegExp(`^${words.years2[0]} · ${words.metrics[0]}`),
    );
    await screenshot(window, `table-chart-years-${language}-${width}`);

    // Switch rows and columns, in the chart's menu: the metrics are along the axis, the years the series.
    await pointAndClick(window, chart.getByTestId("chart-kind"));
    const menu = window.getByTestId("chart-kind-menu");
    const switchItem = menu.getByTestId("chart-switch");
    await expect(switchItem).toHaveText(words.switch);
    await pointAndClick(window, switchItem);
    await expect(chart.getByTestId("chart-point")).toHaveCount(6);
    await expect.poll(() => axisLabels(chart)).toEqual(expect.arrayContaining([...words.metrics]));
    expect(await axisLabels(chart)).not.toContain(words.years2[0]);
    await expect(chart.getByTestId("chart-legend")).toContainText(words.years2[0]);
    await expect(point(chart, 0, 0)).toHaveAttribute(
      "aria-label",
      new RegExp(`^${words.metrics[0]} · ${words.years2[0]}`),
    );
    await expect(chart.getByTestId("chart-changed")).toHaveCount(0);
    await screenshot(window, `table-chart-switched-${language}-${width}`);

    // ⌘Z puts the years back along the axis.
    await expect(chart.getByTestId("chart-kind")).toBeFocused();
    await window.keyboard.press("ControlOrMeta+z");
    await expect.poll(() => axisLabels(chart)).toEqual(expect.arrayContaining([...words.years2]));
    await expect(point(chart, 0, 0)).toHaveAttribute(
      "aria-label",
      new RegExp(`^${words.years2[0]} · ${words.metrics[0]}`),
    );

    // The same from the right-click menu, among the chart's own actions, and again: a second one.
    await chart.locator(".chart-head").click({ button: "right", position: { x: 4, y: 4 } });
    const actions = window.getByTestId("chart-actions");
    await expect(actions).toBeVisible();
    const fromMenu = actions.getByRole("menuitem", { name: new RegExp(`^${words.switch}`) });
    await pointAndClick(window, fromMenu);
    await expect(actions).toBeHidden();
    await expect.poll(() => axisLabels(chart)).toEqual(expect.arrayContaining([...words.metrics]));
    await window.keyboard.press("ControlOrMeta+z");
    await expect.poll(() => axisLabels(chart)).toEqual(expect.arrayContaining([...words.years2]));
    await expect(chart.getByTestId("chart-changed")).toHaveCount(0);

    // Copied as text, the Mind has its table once, a row to a line with tabs; the HTML keeps both.
    await pointAndClick(window, grid.locator("td").filter({ hasText: words.firstMetric }));
    await window.keyboard.press("ControlOrMeta+a");
    const copied = await copyPayload(editor);
    const lines = copied.text.split("\n");
    expect(lines).toContain(words.years.head.join("\t"));
    expect(lines).toContain(words.years.rows[0]?.join("\t"));
    expect(lines).toContain(words.years.rows[1]?.join("\t"));
    expect(lines.filter((line) => line.includes(words.years.rows[0]?.[1] ?? ""))).toHaveLength(1);
    expect(copied.text).toContain(words.lead);
    expect(copied.html).toContain("<table");
    expect(copied.html).toContain('data-type="chart"');
    await app.close();
  });

  test(`a calculation's working table is offered no chart, in its menu or in ⌘K (${language}, ${width}px)`, async () => {
    const words = WORDS[language];
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, width, 660);
    const { grid } = await mindWithTable(window, words.lead, words.working);

    await pointAndClick(window, grid.locator("td").first());
    await pointAndClick(window, window.getByTestId("table-menu"));
    await expect(window.getByTestId("table-actions")).toBeVisible();
    await expect(window.getByTestId("table-rowBelow")).toBeVisible();
    await expect(window.getByTestId("table-chart")).toHaveCount(0);
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("table-actions")).toBeHidden();

    await window.keyboard.press("ControlOrMeta+k");
    await expect(window.getByTestId("palette")).toBeVisible();
    await window.keyboard.type(words.chartFromTable);
    await expect(window.getByTestId("palette-empty")).toBeVisible();
    await expect(window.getByTestId("palette-item")).toHaveCount(0);
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("palette")).toBeHidden();
    await app.close();
  });
}
