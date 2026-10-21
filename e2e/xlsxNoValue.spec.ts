import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { xlsxPackage } from "../tests/helpers/office";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  documentIdOf,
  launchApp,
  openDocumentAt,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * A workbook whose formulas have no saved result, as a script that writes formulas leaves them
 * (#1514), shows "No saved value" in those cells, never 0; a formula with a saved result, 0 too,
 * shows it. Set INCARNAMIND_SCREENSHOTS to a folder to also save a screenshot.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
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

for (const [language, said] of [
  ["en", "No saved value"],
  ["zh-CN", "无保存的值"],
] as const) {
  test(`a formula with no saved result says so, not 0 (${language})`, async () => {
    const text = (ref: string, value: string) =>
      `<c r="${ref}" t="inlineStr"><is><t>${value}</t></is></c>`;
    const sheet = [
      "<sheetData>",
      `<row r="1">${text("A1", "Shipments")}${text("B1", "March")}${text("C1", "April")}</row>`,
      `<row r="2">${text("A2", "Europe")}<c r="B2"><v>128400</v></c><c r="C2"><v>96200</v></c></row>`,
      `<row r="3">${text("A3", "Total")}<c r="B3"><f>SUM(B2:B2)</f><v></v></c><c r="C3"><f>SUM(C2:C2)</f></c><c r="D3"><f>SUM(B2:C2)</f><v>0</v></c></row>`,
      "</sheetData>",
    ].join("");
    const file = join(sources, "Shipments.xlsx");
    await writeFile(file, xlsxPackage({ sheets: [{ name: "Totals", xml: sheet }] }));
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1440, 900);
    await addDocuments(window, [file]);
    await openDocumentAt(window, { documentId: await documentIdOf(window, "Shipments") });

    const grid = window.getByTestId("viewer").getByTestId("viewer-grid");
    await expect(grid.locator('td[data-ref="B2"]')).toHaveText("128400");
    await expect(grid.locator('td[data-ref="B3"]')).toHaveText(said);
    await expect(grid.locator('td[data-ref="C3"]')).toHaveText(said);
    await expect(grid.locator('td[data-ref="D3"]')).toHaveText("0");
    await screenshot(window, `xlsx-no-value-${language === "en" ? "en" : "zh"}`);
    await app.close();
  });
}
