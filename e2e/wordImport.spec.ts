import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import { unzip } from "../tests/helpers/docx";
import {
  commented,
  field,
  NUMBERING,
  PNG,
  paragraph,
  picture,
  run,
  table,
  textBox,
  tracked,
  wordFile,
} from "../tests/helpers/word";
import {
  coreOf,
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  motionSettled,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  useOutputLocation,
} from "./app";

/*
 * Importing a Word file as a Mind (#104): from the "+" menu and by a drop on a
 * Mind, the import screen that asks whose document it is, the Mind with what
 * is kept as is shown in place, a section break that asks before it goes, and
 * a file refused with the reason. Set INCARNAMIND_SCREENSHOTS to a folder to
 * also save screenshots there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

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

async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** The memo on the board Final-30-Word-Import, with someone else's comment and change. */
function memo(): Buffer {
  const comment = commented(0, "Marcus Hale", "Is spring realistic?", run("spring 2027"));
  return wordFile({
    body: [
      paragraph(
        `${run("To: Velomark board")}<w:r><w:br/></w:r>${run("From: Lena Ortiz, Halden Advisory")}<w:r><w:br/></w:r>${run("Date: ")}${field('DATE \\@ "d MMMM yyyy"', "9 October 2026")}`,
      ),
      paragraph("1. Summary", { style: "Heading1" }),
      paragraph(
        `${run("Velomark should enter Germany in 2027 through a ")}${run("distribution partner", { b: true })}${run(" rather than its own sales team. A partner already sells to the specialist bike shops where most e-bikes are bought.")}`,
      ),
      paragraph(
        textBox([
          "At a glance",
          "Market: e-bike batteries and conversion kits, Germany",
          "Route: a distribution partner; Velo Haus and Radwerk shortlisted",
          `First sales: spring 2027`,
        ]),
      ),
      paragraph("2. Options", {
        style: "Heading1",
        sectPr: '<w:sectPr><w:type w:val="nextPage"/></w:sectPr>',
      }),
      paragraph("A direct sales team in Munich", { numId: 2 }),
      paragraph("A distribution partner", { numId: 2 }),
      paragraph(
        `${run("First sales in ")}${comment.inline}${tracked("ins", 4, "Marcus Hale", " at the earliest")}${run(".")}`,
      ),
      table([
        ["Route", "Cost", "Time"],
        ["Partner", "€0.4m", "6 months"],
      ]),
      paragraph(picture("rIdImage", "Velomark logo")),
    ].join(""),
    numbering: NUMBERING,
    comments: comment.comment,
    media: { rIdImage: { name: "media/image1.png", data: PNG } },
  });
}

/** "Import a Word file…" from the "+" menu; the test answers the system's open dialog. */
async function importFromMenu(window: Page): Promise<Locator> {
  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-import-word").click();
  const dialog = window.getByTestId("word-import-dialog");
  await expect(dialog).toBeVisible();
  return dialog;
}

test("Import a Word file… asks whose it is, then opens the Mind with what is kept as is in place", async () => {
  const path = join(files, "Market entry memo (Velomark).docx");
  await writeFile(path, memo());
  const { app, window } = await launchApp(dataDir, { home });
  await dismissChatSetup(window);
  await interceptOpenDialog(app, path);

  const dialog = await importFromMenu(window);
  await expect(dialog.getByTestId("word-import-file")).toHaveText(
    "Market entry memo (Velomark).docx",
  );
  // Someone else commented and changed it: their document, by default, and they are named.
  await expect(dialog.getByTestId("word-import-origin-someone-else")).toBeChecked();
  await expect(dialog.getByTestId("word-import-authors")).toHaveText(
    "It has changes or comments by Marcus Hale.",
  );
  await screenshot(dialog, "word-import-dialog");
  await dialog.getByTestId("word-import-confirm").click();
  await expect(dialog).toBeHidden();

  const pane = window.getByTestId("mind-pane");
  await expect(window.getByTestId("mind-title")).toHaveValue("Market entry memo (Velomark)");
  await expect(window.getByTestId("mind-word-origin")).toHaveText(
    "From Market entry memo (Velomark).docx · Export keeps its styles",
  );
  const editor = window.getByTestId("mind-editor");
  await expect(editor.locator("h1")).toHaveText(["1. Summary", "2. Options"]);
  await expect(editor.locator("ol li")).toHaveText([
    "A direct sales team in Munich",
    "A distribution partner",
  ]);
  await expect(editor.locator("table td")).toHaveCount(6);
  await expect(editor.locator("img.mind-image")).toHaveAttribute("alt", "Velomark logo");

  // A field is a quiet chip with its result; it says what it is, and that it is kept as is.
  const date = editor.locator('[data-testid="word-atom"][data-kind="field"]');
  await expect(date).toHaveText("9 October 2026");
  await expect(date).toHaveAttribute("title", /^Date field · kept as is/);
  // A text box is a box of its text, its label shown on hover.
  const box = editor.locator('[data-testid="word-atom"][data-kind="textBox"]');
  await expect(box).toContainText("At a glance");
  const label = box.locator(".word-atom-label");
  await expect(label).toHaveCSS("opacity", "0");
  await box.hover();
  await expect(label).toHaveText("Text box · kept as is");
  await expect(label).toHaveCSS("opacity", "1");
  // Someone else's change and comment are there, read-only for now.
  await expect(editor.locator('[data-kind="insertion"]')).toHaveText(" at the earliest");
  await expect(editor.locator('[data-kind="comment"]')).toHaveAttribute(
    "title",
    /Comment · Marcus Hale · .*\nIs spring realistic\?/,
  );
  await screenshot(pane, "word-import-mind");

  // Typing beside an atom goes beside it, never into it.
  await date.click();
  await window.keyboard.press("ArrowRight");
  await window.keyboard.type("!");
  await expect(date).toHaveText("9 October 2026");
  await expect(editor.locator("p").first()).toContainText("9 October 2026!");
  await app.close();
});

test("deleting a section break asks first: Keep it leaves it, Delete takes it", async () => {
  const path = join(files, "Two sections.docx");
  await writeFile(
    path,
    wordFile({
      body: [
        paragraph("Section one ends here.", {
          sectPr: '<w:sectPr><w:pgSz w:orient="landscape"/></w:sectPr>',
        }),
        paragraph("Section two."),
      ].join(""),
    }),
  );
  const { app, window } = await launchApp(dataDir, { home });
  await dismissChatSetup(window);
  await interceptOpenDialog(app, path);
  const dialog = await importFromMenu(window);
  await expect(dialog.getByTestId("word-import-origin-own")).toBeChecked();
  await dialog.getByTestId("word-import-confirm").click();

  const editor = window.getByTestId("mind-editor");
  const first = editor.locator("p").first();
  await expect(first).toHaveText("Section one ends here.");
  const sectionBreak = editor.locator('[data-kind="sectionBreak"]');
  await expect(sectionBreak).toHaveCount(1);
  // The whole paragraph, its break too, selected and deleted.
  await first.click({ clickCount: 3 });
  await expect
    .poll(() => window.evaluate(() => getSelection()?.toString()))
    .toContain("Section one ends here.");
  await window.keyboard.press("Backspace");
  const ask = window.getByTestId("section-break-dialog");
  await expect(ask).toBeVisible();
  await expect(ask).toContainText("Delete the section break?");
  await screenshot(ask, "section-break-dialog");
  await ask.getByRole("button", { name: "Keep it" }).click();
  await expect(ask).toBeHidden();
  await expect(sectionBreak).toHaveCount(1);
  await expect(first).toHaveText("Section one ends here.");

  await motionSettled(window);
  await first.click();
  await first.click({ clickCount: 3 });
  await expect
    .poll(() => window.evaluate(() => getSelection()?.toString()))
    .toContain("Section one ends here.");
  await window.keyboard.press("Backspace");
  await ask.getByTestId("section-break-delete").click();
  await expect(sectionBreak).toHaveCount(0);
  await expect(editor).not.toContainText("Section one ends here.");
  await expect(editor).toContainText("Section two.");
  await app.close();
});

test("a file that can't be imported says why, and no Mind is made", async () => {
  const path = join(files, "Old minutes.doc");
  await writeFile(path, "not really a Word 97 file");
  const { app, window } = await launchApp(dataDir, { home });
  await dismissChatSetup(window);
  await interceptOpenDialog(app, path);
  const minds = async () =>
    (await coreOf(window)).evaluate((bridge) => bridge.listMinds().then((list) => list.length));
  const before = await minds();

  const dialog = await importFromMenu(window);
  await expect(dialog.getByTestId("word-import-refusal")).toContainText(
    "“Old minutes.doc” is an older Word file (.doc). Save it as .docx in Word, then import that.",
  );
  await expect(dialog.getByTestId("word-import-confirm")).toHaveCount(0);
  await screenshot(dialog, "word-import-refused");
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toBeHidden();
  expect(await minds()).toBe(before);
  await app.close();
});

test("a Word file dropped on a Mind opens the import screen, which can add it as a Document instead", async () => {
  const path = join(files, "Dropped memo.docx");
  await writeFile(path, wordFile({ body: paragraph("Dropped on a Mind.") }));
  const { window } = await launchApp(dataDir, { home });
  await dismissChatSetup(window);
  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-new-mind").click();
  await expect(window.getByTestId("mind-pane")).toBeVisible();

  // A file from the disk, as a drag from Finder or Explorer carries one.
  await window.evaluate(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.dataset.testid = "drop-source";
    input.hidden = true;
    document.body.append(input);
  });
  await window.getByTestId("drop-source").setInputFiles(path);
  await window.evaluate(() => {
    const input = document.querySelector<HTMLInputElement>('[data-testid="drop-source"]');
    const file = input?.files?.[0];
    const target = document.querySelector('[data-testid="mind-pane"]');
    if (!file || !target) throw new Error("No file or no Mind to drop on.");
    const transfer = new DataTransfer();
    transfer.items.add(file);
    for (const type of ["dragenter", "dragover", "drop"]) {
      target.dispatchEvent(
        new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: transfer }),
      );
    }
  });

  const dialog = window.getByTestId("word-import-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByTestId("word-import-file")).toHaveText("Dropped memo.docx");
  await dialog.getByTestId("word-import-as-document").click();
  await expect(dialog).toBeHidden();
  await expect(window.getByTestId("sidebar")).toContainText("Dropped memo");
});

test("Export writes back into the Word file: what it can't keep is said first, Export anyway saves it, the rest as it was", async () => {
  const kept = paragraph("Keep this paragraph as it is.", { paraId: "0A0B0C0D" });
  const comment = commented(0, "Marcus Hale", "Is spring realistic?", run("spring 2027"));
  const path = join(files, "Memo.docx");
  await writeFile(
    path,
    wordFile({
      body: [
        kept,
        paragraph(`${run("First sales in ")}${comment.inline}${run(".")}`),
        paragraph("The end."),
      ].join(""),
      comments: comment.comment,
    }),
  );
  const { app, window } = await launchApp(dataDir, { home });
  await dismissChatSetup(window);
  await interceptOpenDialog(app, path);
  const dialog = await importFromMenu(window);
  await dialog.getByTestId("word-import-confirm").click();
  const output = join(files, "Exports");
  await mkdir(output);
  await useOutputLocation(window, output);

  // The paragraph with the comment goes.
  const editor = window.getByTestId("mind-editor");
  const commentedParagraph = editor.locator("p", { hasText: "First sales in" });
  await commentedParagraph.click({ clickCount: 3 });
  await expect
    .poll(() => window.evaluate(() => getSelection()?.toString()))
    .toContain("First sales in");
  await window.keyboard.press("Backspace");
  await window.keyboard.press("Backspace");
  await expect(editor).not.toContainText("First sales in");

  // Export says what the file can't keep before anything is saved.
  await window.getByTestId("export-mind").click();
  const result = window.getByTestId("export-result");
  await expect(result.getByTestId("export-review")).toHaveText(
    "1 thing in the Word file can't be kept: Marcus Hale's comment “Is spring realistic?” is on text you deleted · Export anyway",
  );
  await expect(result.getByTestId("export-review-items")).toHaveText(
    "Marcus Hale's comment “Is spring realistic?” is on text you deleted",
  );
  expect(await readdir(output)).toEqual([]);
  await motionSettled(window);
  await screenshot(window, "word-export-review");
  // A narrow window: the line and its list wrap by word, inside the tab.
  await setWindowSize(app, window, 900, 660);
  await motionSettled(window);
  const line = await result.boundingBox();
  const pane = await window.getByTestId("mind-pane").boundingBox();
  expect(line && pane && line.x >= pane.x && line.x + line.width <= pane.x + pane.width).toBe(true);
  await screenshot(window, "word-export-review-narrow");
  await setWindowSize(app, window, 1600, 660);
  await setLanguage(window, "zh-CN");
  await expect(result.getByTestId("export-review")).toContainText("Word 文件中有 1 处无法保留");
  await expect(result.getByTestId("export-anyway")).toHaveText("仍然导出");
  await motionSettled(window);
  await screenshot(window, "word-export-review-zh");
  await setLanguage(window, "en");

  // Export anyway: saved, the paragraph kept byte for byte, the comment's marks gone with its text.
  await result.getByTestId("export-anyway").click();
  await expect(result.getByTestId("export-done")).toHaveText("Saved Memo.docx");
  await motionSettled(window);
  await screenshot(window, "word-export-saved");
  const document =
    unzip(await readFile(join(output, "Memo.docx")))
      .get("word/document.xml")
      ?.toString("utf8") ?? "";
  expect(document).toContain(kept);
  expect(document).toContain("The end.");
  expect(document).not.toContain("First sales in");
  expect(document).not.toContain("comment");
  await app.close();
});
