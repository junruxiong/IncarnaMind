import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { paragraph, run, wordFile } from "../tests/helpers/word";
import {
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  importWordFromMenu,
  interceptOpenDialog,
  launchApp,
  motionSettled,
  openOptions,
  removeDataFolder,
  savedPath,
  setLanguage,
  setWindowSize,
  useLocalChatModel,
  useOutputLocation,
} from "./app";
import { pdfText } from "./pdfText";

/*
 * Someone else's changes in an imported Word file, shown as changes (#734):
 * the assistant reads them marked, with who made each, and a PDF shows them
 * as Word prints them, or as if accepted, as the export sheet says. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

async function screenshot(window: Page, name: string): Promise<void> {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const PRIYA = 'w:author="Priya Nair" w:date="2026-10-02T09:00:00Z"';

/** A contract's termination clause, with the Customer's right to terminate deleted by Priya. */
const CONTRACT = wordFile({
  body: [
    paragraph("9. Termination", { style: "Heading1" }),
    paragraph(
      `${run("9.1 Either party may end this agreement on ")}<w:del w:id="1" ${PRIYA}><w:r><w:delText xml:space="preserve">30</w:delText></w:r></w:del><w:ins w:id="2" ${PRIYA}>${run("90")}</w:ins>${run(" days' notice.")}`,
    ),
    `<w:p><w:pPr><w:rPr><w:del w:id="3" ${PRIYA}/></w:rPr></w:pPr><w:del w:id="4" ${PRIYA}><w:r><w:delText xml:space="preserve">9.2 The Customer may terminate for convenience.</w:delText></w:r></w:del></w:p>`,
    paragraph("9.3 Fees paid are not refunded."),
  ].join(""),
});

let dataDir: string;
let home: string;
let files: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  home = await createDataFolder();
  files = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
  await removeDataFolder(files);
});

test("the assistant reads someone's changes as changes, and a PDF shows them struck and underlined with who made each, or as if accepted", async () => {
  const path = join(files, "MSA redline.docx");
  await writeFile(path, CONTRACT);
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  const output = join(files, "Exports");
  await mkdir(output);
  await useOutputLocation(window, output);
  await interceptOpenDialog(app, path);
  const dialog = await importWordFromMenu(window);
  await dialog.getByTestId("word-import-confirm").click();
  await expect(dialog).toBeHidden();
  await expect(window.getByTestId("their-changes-count")).toHaveText("2 changes from Priya Nair");

  // Asked beside the Mind, the model reads the deleted clause as Priya's deletion.
  const answer = await askInAssistant(window, "What did they change in clause 9?");
  await expect(answer).toContainText(
    "[deleted by Priya Nair, 2026-10-02: 9.2 The Customer may terminate for convenience.]",
  );
  await expect(answer).toContainText("[inserted by Priya Nair, 2026-10-02: 90]");

  // The export sheet offers how to show them, for a PDF: with changes shown, by default.
  await openOptions(window, "Export options");
  const sheet = window.getByTestId("export-dialog");
  await sheet.getByTestId("export-format").selectOption("pdf");
  const shown = sheet.getByTestId("export-their-changes");
  await expect(shown).toHaveValue("shown");
  await expect(sheet).toContainText(
    "2 changes from the Word file are waiting to be accepted or rejected.",
  );
  await motionSettled(window);
  await screenshot(window, "export-their-changes");
  await setLanguage(window, "zh-CN");
  await expect(sheet).toContainText("Word 文件中有 2 处修订尚未接受或拒绝。");
  await expect(shown.locator("option")).toHaveText(["显示修订", "按全部接受"]);
  await motionSettled(window);
  await screenshot(window, "export-their-changes-zh");
  await setLanguage(window, "en");
  await sheet.getByTestId("export-save").click();
  await expect(sheet).toBeHidden();

  // The PDF has the deleted clause, with a note of who deleted it and when.
  const result = window.getByTestId("export-result");
  await window.getByTestId("export-mind").click();
  await expect(result.getByTestId("export-done")).toHaveText("Saved MSA redline.pdf");
  const marked = (await pdfText(await readFile(await savedPath(window, "MSA redline.pdf")))).join(
    "\n",
  );
  expect(marked).toContain("9.2 The Customer may terminate for convenience.");
  expect(marked).toContain("Deleted by Priya Nair, 2026-10-02");
  expect(marked).toContain("Replaced by Priya Nair, 2026-10-02");

  // As if accepted: the clause is gone, the notice is 90 days, and no note is left.
  await openOptions(window, "Export options");
  await shown.selectOption("accepted");
  await sheet.getByTestId("export-save").click();
  await expect(sheet).toBeHidden();
  await window.getByTestId("export-mind").click();
  await expect(result.getByTestId("export-done")).toHaveText("Saved MSA redline 2.pdf");
  const accepted = (
    await pdfText(await readFile(await savedPath(window, "MSA redline 2.pdf")))
  ).join("\n");
  expect(accepted).not.toContain("9.2");
  expect(accepted).toContain("90 days' notice.");
  expect(accepted).not.toContain("Priya Nair");
  await app.close();
});
