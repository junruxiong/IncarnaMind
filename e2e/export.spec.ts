import { mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { footnotesOf, paragraphsOf, part, unzip } from "../tests/helpers/docx";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  interceptOpenPath,
  interceptShowItemInFolder,
  launchApp,
  newMind,
  openSettings,
  pathsOpened,
  pathsShown,
  removeDataFolder,
  useLocalChatModel,
  useOutputLocation,
} from "./app";

/** Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots of the questions and Settings. */
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  const folder = process.env.INCARNAMIND_SCREENSHOTS;
  if (folder) await target.screenshot({ path: join(folder, `${name}.png`) });
}

/** The core's bridge, as the window has it. */
const bridge = (window: Page) => ({
  link: (path: string) =>
    window.evaluate(
      (folder) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.addLinkedFolder(folder),
      path,
    ),
  language: (language: "en" | "zh-CN") =>
    window.evaluate(
      (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    ),
});

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

test("a Mind exports to .docx: the dialog counts the unverified Citation first; the first save asks where new files go, and the next goes there under a free name", async () => {
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

  // The first save asks where new files go, offering IncarnaMind in Documents (the test
  // build's own Documents folder). Esc cancels: nothing is saved, and the export waits.
  await interceptShowItemInFolder(app);
  await dialog.getByTestId("export-save").click();
  const ask = window.getByTestId("file-write-dialog");
  await expect(ask).toHaveAttribute("data-kind", "output-location");
  await expect(ask).toContainText("Where should new files go?");
  const suggested = join(dataDir, "Documents", "IncarnaMind");
  await expect(ask.getByTestId("file-write-suggested")).toHaveText(suggested);
  await screenshot(ask, "file-write-output-location-en");
  await window.keyboard.press("Escape");
  await expect(ask).toBeHidden();
  await expect(dialog.getByTestId("export-save")).toBeEnabled();
  await expect(dialog.getByTestId("export-done")).toHaveCount(0);

  // Save here: saved there, and the dialog says so and shows the file in the file manager.
  await dialog.getByTestId("export-save").click();
  await ask.getByTestId("file-write-save-here").click();
  await expect(ask).toBeHidden();
  await expect(dialog.getByTestId("export-done")).toContainText("Saved Tides.docx");
  await expect(dialog.getByTestId("export-close")).toBeFocused();
  await dialog.getByTestId("export-show").click();
  await expect(dialog).toBeHidden();
  const target = join(await realpath(suggested), "Tides.docx");
  expect(await pathsShown(app)).toEqual([target]);

  // The file is there: the Note and the Answer, not the Question, and the Citation a footnote.
  const files = unzip(await readFile(target));
  const paragraphs = paragraphsOf(part(files, "word/document.xml")).map((each) => each.text);
  expect(paragraphs).toContain("Notes on tides.");
  expect(paragraphs).toContain("Your Documents answer this[1].");
  expect(paragraphs.join("\n")).not.toContain("When do spring tides happen?");
  expect(footnotesOf(part(files, "word/footnotes.xml"))).toEqual({
    "1": "Tides, p. 2 [unverified]",
  });

  // The next export doesn't ask, and leaves the first file as it was.
  const first = await readFile(target);
  await window.getByTestId("export-mind").click();
  await dialog.getByTestId("export-save").click();
  await expect(dialog.getByTestId("export-done")).toContainText("Saved Tides 2.docx");
  await expect(ask).toBeHidden();
  expect((await readFile(target)).equals(first)).toBe(true);
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
  // The cursor comes back after the formula a frame later: type once the editor has it.
  await expect(editor).toBeFocused();
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
  await useOutputLocation(window, sources);
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

test("saving inside a Linked folder asks once; Settings › General shows where new files go and the answer, and turning it off makes the export say so, with Allow", async () => {
  const papers = join(sources, "Papers");
  await mkdir(join(papers, "Exports"), { recursive: true });
  await writeFile(join(papers, "Lease.txt"), "Rent is due monthly.");
  const original = await readFile(join(papers, "Lease.txt"));
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await bridge(window).link(papers);
  await newMind(window);
  await window.getByTestId("mind-title").fill("Tides");
  await window.getByTestId("mind-editor").click();
  await window.keyboard.type("Spring tides come twice a month.");

  // Settings › General: not chosen yet; Choose… picks a folder inside the Linked folder.
  const settings = await openSettings(window, "general");
  const output = settings.getByTestId("output-location-settings");
  await expect(output.getByTestId("output-location-path")).toHaveText(
    "Not chosen yet. The first save asks.",
  );
  await interceptOpenDialog(app, join(papers, "Exports"));
  await output.getByTestId("output-location-change").click();
  await expect(output.getByTestId("output-location-path")).toHaveText(
    await realpath(join(papers, "Exports")),
  );
  await expect(output.getByTestId("output-location-change")).toHaveText("Change…");
  await expect(output.getByTestId("linked-folder-writing")).toHaveCount(0);
  await closeSettings(window);

  // The first save there asks about the Linked folder; the next doesn't.
  const dialog = window.getByTestId("export-dialog");
  const ask = window.getByTestId("file-write-dialog");
  const exportOnce = async () => {
    await window.getByTestId("export-mind").click();
    await dialog.getByTestId("export-save").click();
  };
  await exportOnce();
  await expect(ask).toHaveAttribute("data-kind", "linked-folder");
  await expect(ask.getByRole("heading")).toHaveText("Add files to “Papers”?");
  await screenshot(ask, "file-write-linked-folder-en");
  await ask.getByTestId("file-write-allow").click();
  await expect(dialog.getByTestId("export-done")).toContainText("Saved Tides.docx");
  await dialog.getByTestId("export-close").click();
  await exportOnce();
  await expect(dialog.getByTestId("export-done")).toContainText("Saved Tides 2.docx");
  await expect(ask).toBeHidden();
  await dialog.getByTestId("export-close").click();

  // Settings shows the answer, and turns it off.
  await openSettings(window, "general");
  const row = output.getByTestId("linked-folder-writing-row");
  await expect(row).toContainText("Papers");
  const allowed = row.getByRole("switch", { name: "Allow new files in Papers" });
  await expect(allowed).toBeChecked();
  await screenshot(settings, "settings-general-output-location-en");
  await allowed.click();
  await expect(allowed).not.toBeChecked();
  await closeSettings(window);

  // The export says it isn't allowed, without asking; its Allow saves it.
  await exportOnce();
  const problem = dialog.getByTestId("export-problem");
  await expect(problem).toHaveText(
    "Couldn't save: you haven't allowed new files in “Papers”, a Linked folder · Allow",
  );
  await expect(ask).toBeHidden();
  await screenshot(dialog, "export-not-allowed-en");
  await problem.getByTestId("export-allow").click();
  await expect(dialog.getByTestId("export-done")).toContainText("Saved Tides 3.docx");
  await dialog.getByTestId("export-close").click();
  expect((await readFile(join(papers, "Lease.txt"))).equals(original)).toBe(true);

  // In Chinese, for another Linked folder: asked, and Don't allow is said where it happened.
  const notes = join(sources, "Notes");
  await mkdir(notes);
  await bridge(window).language("zh-CN");
  await bridge(window).link(notes);
  await useOutputLocation(window, notes);
  await exportOnce();
  await expect(ask.getByRole("heading")).toHaveText("在“Notes”中添加文件？");
  await screenshot(ask, "file-write-linked-folder-zh");
  await ask.getByTestId("file-write-deny").click();
  await expect(problem).toHaveText("无法保存：你没有允许在同步文件夹“Notes”中添加文件 · 允许");
  await screenshot(dialog, "export-not-allowed-zh");
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
