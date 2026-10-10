import { readFile, realpath, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { expect, test } from "@playwright/test";
import { footnotesOf, paragraphsOf, part, unzip } from "../tests/helpers/docx";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  interceptOpenPath,
  interceptSaveDialog,
  interceptShowItemInFolder,
  launchApp,
  newMind,
  pathsOpened,
  pathsShown,
  removeDataFolder,
  saveDialogsAsked,
  useLocalChatModel,
} from "./app";

/** Spring tides are on page 2. */
const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
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

test("a Mind exports to .docx: the dialog counts the unverified Citation first, then the file is saved where the User chose", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);

  // A Note, then a Question whose Answer misquotes its Document.
  await newMind(window);
  await window.getByTestId("mind-title").fill("Tides");
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("Notes on tides.");
  await window.keyboard.press("Enter");
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("When do spring tides happen? Misquote it.");
  await window.keyboard.press("Enter");
  const answer = editor.getByTestId("answer");
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  await expect(answer.getByTestId("citation")).toHaveAttribute("data-check", "not-found");

  // Before anything is written, the dialog says how many Citations are unverified.
  await window.getByTestId("export-mind").click();
  const dialog = window.getByTestId("export-dialog");
  await expect(dialog.getByTestId("export-format-docx")).toBeChecked();
  await expect(dialog.getByTestId("export-include-questions")).not.toBeChecked();
  const summary = dialog.getByTestId("export-citations");
  await expect(summary).toHaveAttribute("data-unverified", "1");
  await expect(summary).toContainText("1 Citation is unverified");

  // The system save dialog is answered by the test hook.
  const target = join(sources, "Exported.docx");
  await interceptSaveDialog(app, target);
  await interceptShowItemInFolder(app);
  await dialog.getByTestId("export-save").click();
  // Saved: the dialog says where, and shows the file in the file manager.
  await expect(dialog.getByTestId("export-done")).toContainText("Saved Exported.docx");
  await expect(dialog.getByTestId("export-close")).toBeFocused();
  await dialog.getByTestId("export-show").click();
  await expect(dialog).toBeHidden();
  expect(await pathsShown(app)).toEqual([target]);
  const [asked] = await saveDialogsAsked(app);
  expect(basename(asked?.defaultPath ?? "")).toBe("Tides.docx");
  expect(asked?.filters).toEqual([{ name: "Word document", extensions: ["docx"] }]);

  // The file is there: the Note and the Answer, not the Question, and the Citation a footnote.
  const files = unzip(await readFile(target));
  const paragraphs = paragraphsOf(part(files, "word/document.xml")).map((each) => each.text);
  expect(paragraphs).toContain("Notes on tides.");
  expect(paragraphs).toContain("Your Documents answer this[1].");
  expect(paragraphs.join("\n")).not.toContain("When do spring tides happen?");
  expect(footnotesOf(part(files, "word/footnotes.xml"))).toEqual({
    "1": "Tides, p. 2 [unverified]",
  });
  await app.close();
});

test("maths exports to .docx as Word equations, which a Word viewer shows as equations", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window);
  await window.getByTestId("mind-title").fill("Gravity");
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  // A formula on its own line, then one in a sentence, from the slash menu.
  await window.keyboard.type("/math");
  await window.keyboard.press("Enter");
  await window.getByTestId("math-editor").fill("F = G\\frac{m_1 m_2}{r^2}");
  await window.getByTestId("math-editor").press("Enter");
  await expect(editor.locator('[data-type="block-math"] .katex')).toBeVisible();
  await window.keyboard.press("Enter");
  await window.keyboard.type("The tide-raising pull falls off as /inline");
  await expect(window.getByTestId("slash-item-inline-math")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await window.keyboard.press("Enter");
  await window.getByTestId("math-editor").fill("\\frac{1}{r^3}");
  await window.getByTestId("math-editor").press("Enter");
  await expect(editor.locator('[data-type="inline-math"] .katex')).toBeVisible();

  const target = join(sources, "Gravity.docx");
  await interceptSaveDialog(app, target);
  await window.getByTestId("export-mind").click();
  const dialog = window.getByTestId("export-dialog");
  await dialog.getByTestId("export-save").click();
  await expect(dialog.getByTestId("export-done")).toBeVisible();
  await dialog.getByTestId("export-close").click();

  // Word equations, not "$…$" text: one on its own line, one in the sentence.
  const document = part(unzip(await readFile(target)), "word/document.xml");
  expect(document.match(/<m:oMathPara>/g)).toHaveLength(1);
  expect(document.match(/<m:oMath>/g)).toHaveLength(2);
  expect(document).not.toContain("$");
  expect(paragraphsOf(document).map((each) => each.text)).toContain(
    "The tide-raising pull falls off as ",
  );

  // Opened as a Document, the Word viewer draws them as equations.
  await addDocuments(window, [target]);
  await window.getByTestId("open-document").click();
  const viewer = window.getByTestId("viewer");
  const docx = viewer.getByTestId("viewer-docx");
  await expect(docx).toHaveAttribute("data-rendered", "yes");
  const equations = docx.locator("math");
  await expect(equations).toHaveCount(2);
  await expect(equations.first().locator("mfrac")).toHaveCount(1);
  await expect(equations.nth(1).locator("mfrac")).toHaveCount(1);
  if (process.env.INCARNAMIND_SCREENSHOTS) {
    await viewer.screenshot({
      path: join(process.env.INCARNAMIND_SCREENSHOTS, "export-docx-equations.png"),
    });
  }
  await app.close();
});

test("Settings opens the data folder in the file manager", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await interceptOpenPath(app);

  await window.getByRole("button", { name: "Settings" }).click();
  await window.getByTestId("settings").getByTestId("open-data-folder").click();

  await expect.poll(() => pathsOpened(app)).toHaveLength(1);
  const [opened] = await pathsOpened(app);
  expect(await realpath(opened ?? "")).toBe(await realpath(dataDir));
  await app.close();
});
