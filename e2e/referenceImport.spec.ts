/**
 * "Import references…" in the Library (#287): the Library's More menu opens
 * the file picker, the export is matched to the Documents there, and the
 * result shows in the Library: what matched, what has no Document yet (with
 * "Add the PDF…"), what couldn't be read, and Undo. In English and Chinese.
 * Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */
import { copyFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const EXPORTS = join(__dirname, "..", "tests", "fixtures", "references", "exports");

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

/** The exports beside the PDF one of them points to, and two papers the export matches. */
async function writeSources() {
  for (const name of ["library.bib", "library.ris"]) {
    await copyFile(join(EXPORTS, name), join(sources, name));
  }
  await writeFile(join(sources, "notes.txt"), "Buy milk.\nCall the bank.\n");
  await mkdir(join(sources, "files"));
  const rust = join(sources, "files", "rust-book.pdf");
  await writeFile(rust, buildPdf([{ lines: ["The Rust Programming Language"] }]));
  const bowler = join(sources, "bowler.pdf");
  await writeFile(
    bowler,
    buildPdf([{ lines: ["Background", "https://doi.org/10.1186/1471-2458-10-456"] }]),
  );
  const halpern = join(sources, "halpern.pdf");
  await writeFile(
    halpern,
    buildPdf([{ lines: ["Chapter 1"] }], {
      info: { Title: "Reasoning about Uncertainty", Author: "Joseph Halpern" },
    }),
  );
  return { bowler, halpern, rust };
}

async function shot(window: Page, name: string) {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

test("Import references: matched to Documents, the rest kept, Add the PDF, and Undo", async () => {
  const { bowler, halpern, rust } = await writeSources();
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await addDocuments(window, [bowler, halpern]);
    await window.getByTestId("open-library").click();
    await setWindowSize(app, window, 1000, 700);

    // The Library's More menu, then the file picker (answered with the export).
    await interceptOpenDialog(app, join(sources, "library.bib"));
    await window.getByTestId("library-add-documents").click();
    await expect(window.getByTestId("import-references")).toHaveText("Import references…");
    await window.getByTestId("import-references").click();

    const panel = window.getByTestId("reference-import");
    await expect(window.getByTestId("import-summary")).toHaveText("Imported from “library.bib”");
    await expect(window.getByTestId("import-matched").locator("summary")).toHaveText(
      "Matched to Documents: 2",
    );
    await expect(window.getByTestId("import-unmatched").locator("summary")).toContainText(
      "Without a Document: 3",
    );
    // Undo is offered at once, as a line.
    await expect(window.getByTestId("undo-message")).toHaveText(
      "Imported references from “library.bib”",
    );

    // What matched, and how.
    await window.getByTestId("import-matched").locator("summary").click();
    await expect(window.getByTestId("import-matched-row")).toHaveText([
      /A systematic review.* · by DOI/,
      /Reasoning about Uncertainty · by title and year/,
    ]);
    // What didn't, each with "Add the PDF…".
    await window.getByTestId("import-unmatched").locator("summary").click();
    await expect(window.getByTestId("import-unmatched-row")).toHaveCount(3);
    // The export attaches one PDF that is on this computer.
    await expect(window.getByTestId("import-files")).toHaveText(
      "Attached files found on this computer: 1",
    );
    await shot(window, "287-import-en");

    // The panel fits the Library at the width it is.
    const fits = await panel.evaluate((element) => element.scrollWidth <= element.clientWidth);
    expect(fits).toBe(true);

    // The help is there when it is wanted.
    await window.getByTestId("import-help").locator("summary").click();
    await expect(window.getByTestId("import-help")).toContainText("Zotero: File ▸ Export Library…");

    // "Add the PDF…" gives a reference its Document.
    await interceptOpenDialog(app, rust);
    const rustRow = window
      .getByTestId("import-unmatched-row")
      .filter({ hasText: "The Rust Programming Language" });
    await rustRow.getByTestId("import-add-pdf").click();
    await expect(window.getByTestId("import-unmatched-row")).toHaveCount(2);
    await expect(window.getByTestId("import-unmatched").locator("summary")).toContainText(
      "Without a Document: 2",
    );

    // Undo takes the import back: the panel goes, and so do the references it added.
    await window.getByTestId("undo-action").click();
    await expect(panel).toHaveCount(0);

    // Importing again, and once more: the second changes nothing.
    await interceptOpenDialog(app, join(sources, "library.bib"));
    await window.getByTestId("library-add-documents").click();
    await window.getByTestId("import-references").click();
    await expect(window.getByTestId("import-summary")).toHaveText("Imported from “library.bib”");
    await window.getByTestId("library-add-documents").click();
    await window.getByTestId("import-references").click();
    await expect(window.getByTestId("import-summary")).toHaveText(
      "Nothing new: “library.bib” is already imported.",
    );
  } finally {
    await app.close();
  }
});

test("Import references speaks Chinese, and says why a file can't be imported", async () => {
  const { bowler, halpern } = await writeSources();
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await addDocuments(window, [bowler, halpern]);
    await setLanguage(window, "zh-CN");
    await window.getByTestId("open-library").click();
    await setWindowSize(app, window, 900, 700);

    await window.getByTestId("library-add-documents").click();
    await expect(window.getByTestId("import-references")).toHaveText("导入引用信息…");
    await interceptOpenDialog(app, join(sources, "library.ris"));
    await window.getByTestId("import-references").click();

    await expect(window.getByTestId("import-summary")).toHaveText("已从“library.ris”导入");
    await expect(window.getByTestId("import-matched").locator("summary")).toHaveText(
      "已匹配到文档：2",
    );
    await expect(window.getByTestId("import-unmatched").locator("summary")).toContainText(
      "还没有文档：3",
    );
    await window.getByTestId("import-unmatched").locator("summary").click();
    await expect(window.getByTestId("import-add-pdf").first()).toHaveText("添加 PDF…");
    await expect(window.getByTestId("undo-message")).toHaveText("已从“library.ris”导入引用信息");
    await shot(window, "287-import-zh");
    const panel = window.getByTestId("reference-import");
    expect(await panel.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
      true,
    );

    // A file that isn't references says so, with the one thing to do next.
    await interceptOpenDialog(app, join(sources, "notes.txt"));
    await window.getByTestId("library-add-documents").click();
    await window.getByTestId("import-references").click();
    await expect(window.getByTestId("import-failed")).toHaveText(
      "“notes.txt”不是 CSL JSON、BibTeX 或 RIS 文件。 · 选择其他文件…",
    );
    await shot(window, "287-import-zh-failed");
  } finally {
    await app.close();
  }
});
