import { realpath, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  expectReadyDocuments,
  launchApp,
  linkFolderFromSidebar,
  newMind,
  openFolderInLibrary,
  removeDataFolder,
} from "./app";

/*
 * From the Library into writing (#59): filters by year, format and status,
 * with counts, and "Ask about this Folder" / "Start a Mind from this Folder",
 * which scope the composer's next Question to the Folder, or to exactly the
 * Documents a filter shows. Only one test asks, with the scripted model. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

/** Each file's year is the latest year written in its first Unit (#53). */
const FIXTURES = {
  "Tide report.txt": "Field report, 2021: spring tides along the coast.\n",
  "Harbour notes.md": "# Harbour notes\n\nObserved at the harbour in 2023.\n",
  "Estuary survey.txt": "Estuary survey, summer 2023. Salt water reaches the bridge.\n",
  "Undated memo.txt": "A memo about the quay, with no date in it.\n",
  "Tide table.csv": "day,height\nMonday,2.1\nTuesday,2.4\n",
} as const;

const FOLDER = "Coastal research";

let dataDir: string;
/** The folder linked: outside the data folder. */
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await realpath(await createDataFolder());
  for (const [name, text] of Object.entries(FIXTURES)) await writeFile(join(sources, name), text);
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/**
 * Links the fixtures from the sidebar, makes the Folder in the Library and
 * files every Document but the CSV in it, by hand, as the User would.
 * Returns the Library, showing all Documents.
 */
async function libraryWithFolder(app: Parameters<typeof linkFolderFromSidebar>[0], window: Page) {
  await linkFolderFromSidebar(app, window, sources);
  await expectReadyDocuments(window, 5);
  await window.getByTestId("open-library").click();
  const library = window.getByTestId("library");
  await library.getByRole("button", { name: "New folder", exact: true }).click();
  await library.getByLabel("Folder name", { exact: true }).fill(FOLDER);
  await library.getByLabel("Description", { exact: true }).fill("Tides, harbours and estuaries");
  await library.getByRole("button", { name: "Save folder" }).click();
  for (const name of ["Tide report", "Harbour notes", "Estuary survey", "Undated memo"]) {
    const assignment = library.getByRole("combobox", { name: `Folder for ${name}`, exact: true });
    await assignment.selectOption({ label: FOLDER });
    await expect(assignment.locator("option:checked")).toHaveText(FOLDER);
  }
  return library;
}

/** Opens a filter's menu and returns its options, as [value, count, checked]. */
async function openFilter(library: Locator, facet: string) {
  await library.locator(`[data-testid="library-filter"][data-facet="${facet}"]`).click();
  const menu = library.locator(`[data-testid="library-filter-menu"][data-facet="${facet}"]`);
  await expect(menu).toBeVisible();
  return menu;
}

const optionsOf = (menu: Locator) =>
  menu
    .getByTestId("library-filter-option")
    .evaluateAll((items) =>
      items.map((item) => [
        item.querySelector(".truncate")?.textContent,
        Number(item.getAttribute("data-count")),
        item.getAttribute("aria-checked") === "true",
      ]),
    );

/** The open Mind's composer, where the Question is written, its Search scope as chips. */
const composerOf = (window: Page) => window.getByTestId("composer");

test("filters by year, format and status with counts, and asks about the Documents shown or the Folder", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    // A Mind written in already: the most recent one, where the Questions go.
    await newMind(window);
    await window.keyboard.type("Tide notes");
    await window.keyboard.press("Enter");
    await window.keyboard.type("Notes on the tides.");
    const library = await libraryWithFolder(app, window);
    // A file removed from the folder: its Document is missing, and stays in its Folder.
    await rm(join(sources, "Undated memo.txt"));
    await expect
      .poll(
        () =>
          window.evaluate(
            async () =>
              (
                await (
                  globalThis as unknown as { incarnamind: CoreBridge }
                ).incarnamind.listDocuments()
              ).filter((each) => each.fileStatus === "missing").length,
          ),
        { timeout: 15_000 },
      )
      .toBe(1);

    await openFolderInLibrary(window, FOLDER);
    await expect(library.getByRole("heading", { name: FOLDER, exact: true })).toBeVisible();
    const rows = library.getByTestId("library-document");
    await expect(rows).toHaveCount(4);
    await expect(library.getByTestId("library-ask")).toHaveText("Ask about this Folder");

    // Each filter counts its options among the Folder's Documents: newest year first, then no date.
    let menu = await openFilter(library, "year");
    expect(await optionsOf(menu)).toEqual([
      ["2023", 2, false],
      ["2021", 1, false],
      ["No date", 1, false],
    ]);
    await menu.getByRole("menuitemcheckbox", { name: "2023, 2 Documents" }).click();
    await window.keyboard.press("Escape");
    await expect(rows).toHaveCount(2);
    await expect(library.getByTestId("library-count")).toHaveText("2 of 4 Documents");
    await expect(library.locator('[data-testid="library-filter"][data-facet="year"]')).toHaveText(
      "Year: 2023",
    );

    // Other filters count among what the year keeps; options of one filter add up.
    menu = await openFilter(library, "format");
    expect(await optionsOf(menu)).toEqual([
      ["Markdown", 1, false],
      ["Plain text", 1, false],
    ]);
    await window.keyboard.press("Escape");
    menu = await openFilter(library, "status");
    expect(await optionsOf(menu)).toEqual([
      ["Available", 2, false],
      ["Missing", 0, false],
    ]);
    await window.keyboard.press("Escape");
    menu = await openFilter(library, "year");
    await menu.getByRole("menuitemcheckbox", { name: "No date, 1 Document" }).click();
    await window.keyboard.press("Escape");
    await expect(rows).toHaveCount(3);
    menu = await openFilter(library, "status");
    await menu.getByRole("menuitemcheckbox", { name: "Missing, 1 Document" }).click();
    await expect(rows).toHaveCount(1);
    await expect(rows).toContainText("Undated memo");
    await screenshot(window, "library-filters");
    await window.keyboard.press("Escape");

    // One click clears them all.
    await library.getByTestId("library-filters-clear").click();
    await expect(rows).toHaveCount(4);
    await expect(library.getByTestId("library-count")).toHaveText("4 Documents");
    await expect(library.getByTestId("library-filters-clear")).toHaveCount(0);

    // Filtered, the Question searches exactly the Documents shown, and the button says how many.
    menu = await openFilter(library, "year");
    await menu.getByRole("menuitemcheckbox", { name: "2023, 2 Documents" }).click();
    await window.keyboard.press("Escape");
    await expect(rows).toHaveCount(2);
    const shownIds = await rows.evaluateAll((items) =>
      items.map((item) => item.getAttribute("data-document-id")),
    );
    const ask = library.getByTestId("library-ask");
    await expect(ask).toHaveText("Ask about these 2 Documents");
    await expect(ask).toHaveAttribute("data-scope", "documents");
    await ask.click();

    await expect(window.getByTestId("library")).toHaveCount(0);
    await expect(window.getByTestId("mind-title")).toHaveValue("Tide notes");
    const editor = window.getByTestId("mind-editor");
    await expect(editor).toContainText("Notes on the tides.");
    // The composer takes the focus, its Search scope the Documents shown.
    const composer = composerOf(window);
    const input = composer.getByTestId("composer-input");
    const chips = composer.getByTestId("scope-chip");
    await expect(chips).toHaveCount(2);
    expect(
      (
        await chips.evaluateAll((items) => items.map((item) => item.getAttribute("data-id")))
      ).sort(),
    ).toEqual([...shownIds].sort());
    await expect(chips.first()).toHaveAttribute("data-kind", "document");
    await expect(input).toBeFocused();
    await window.keyboard.type("What was observed in 2023?");
    await expect(input).toHaveValue("What was observed in 2023?");
    await screenshot(window, "library-ask-documents");

    // Unfiltered, it searches the Folder itself, in the same Mind, after the first.
    await openFolderInLibrary(window, FOLDER);
    await expect(library.getByTestId("library-filters-clear")).toHaveCount(0);
    await expect(library.getByTestId("library-ask")).toHaveAttribute("data-scope", "folder");
    await library.getByRole("button", { name: "Ask about this Folder", exact: true }).click();
    await expect(window.getByTestId("mind-title")).toHaveValue("Tide notes");
    const folderChip = chips;
    await expect(folderChip).toHaveCount(1);
    await expect(folderChip).toHaveText(FOLDER);
    await expect(folderChip).toHaveAttribute("data-kind", "folder");
    await expect(input).toBeFocused();
    await window.keyboard.type(" Which harbour?");
    await expect(input).toHaveValue("What was observed in 2023? Which harbour?");
    // Only the one Mind: nothing new was made.
    await expect(window.getByTestId("mind-list-item")).toHaveCount(1);
  } finally {
    await app.close();
  }
});

test("starts a Mind named after the Folder with its scoped Question first, in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    const library = await libraryWithFolder(app, window);
    const folderId = await window.evaluate(
      async (name) =>
        (
          await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.getLibrary()
        ).groups.find((group) => group.name === name)?.id,
      FOLDER,
    );
    await openFolderInLibrary(window, FOLDER);
    // The bridges and filters read in Chinese too.
    await window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "zh-CN" },
      }),
    );
    await expect(library.getByTestId("library-ask")).toHaveText("就此文件夹提问");
    await expect(library.getByTestId("library-start-mind")).toHaveText("从此文件夹新建 Mind");
    await expect(library.locator('[data-testid="library-filter"][data-facet="status"]')).toHaveText(
      "状态",
    );
    await window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "en" },
      }),
    );

    await library.getByRole("button", { name: "Start a Mind from this Folder" }).click();
    await expect(window.getByTestId("library")).toHaveCount(0);
    await expect(window.getByTestId("mind-title")).toHaveValue(FOLDER);
    await expect(
      window.getByTestId("mind-list-item").getByTestId("row-text").getByText(FOLDER),
    ).toBeVisible();
    const editor = window.getByTestId("mind-editor");
    // The composer takes the focus, its Search scope the Folder.
    const composer = composerOf(window);
    const composerChip = composer.getByTestId("scope-chip");
    await expect(composerChip).toHaveText(FOLDER);
    await expect(composerChip).toHaveAttribute("data-kind", "folder");
    await expect(composerChip).toHaveAttribute("data-id", folderId ?? "");
    await expect(composer.getByTestId("composer-input")).toBeFocused();
    await window.keyboard.type("What do these say about tides?");
    await screenshot(window, "library-start-mind");

    // Asked, the Question is the Mind's first Block, with the Folder as its Search scope.
    await connectLocalChatModel(window);
    await window.keyboard.press("Enter");
    await expect
      .poll(() =>
        editor.evaluate(
          (element) =>
            element.firstElementChild?.matches(
              '[data-testid="question"], :has(> [data-testid="question"])',
            ) === true,
        ),
      )
      .toBe(true);
    const chip = editor.getByTestId("question").getByTestId("scope-chip");
    await expect(chip).toHaveText(FOLDER);
    await expect(chip).toHaveAttribute("data-id", folderId ?? "");
    await expect(editor.getByTestId("question").locator(".question-text")).toHaveText(
      "What do these say about tides?",
    );
    await expect(editor.getByTestId("answer")).toHaveAttribute("data-status", "done", {
      timeout: 15_000,
    });
  } finally {
    await app.close();
  }
});

test("with no Mind yet, asking about many Documents shown makes one, and their chips fold", async () => {
  for (let index = 1; index <= 10; index++) {
    await writeFile(join(sources, `Survey ${index}.txt`), `Tide survey number ${index}, 2020.\n`);
  }
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    await linkFolderFromSidebar(app, window, sources);
    await expectReadyDocuments(window, 15);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    // All Documents, unfiltered: every Question searches them already, so nothing is offered.
    await expect(library.getByTestId("library-ask")).toHaveCount(0);
    const menu = await openFilter(library, "year");
    await menu.getByRole("menuitemcheckbox", { name: "2020, 10 Documents" }).click();
    await window.keyboard.press("Escape");
    await expect(library.getByTestId("library-count")).toHaveText("10 of 15 Documents");
    await library.getByRole("button", { name: "Ask about these 10 Documents" }).click();

    await expect(window.getByTestId("mind-list-item")).toHaveCount(1);
    await expect(window.getByTestId("mind-title")).toHaveValue("");
    // The composer's Search scope: its first chips, then "+N more".
    const composer = composerOf(window);
    const input = composer.getByTestId("composer-input");
    const chips = composer.getByTestId("scope-chip");
    const more = composer.getByTestId("scope-chips-more");
    await expect(chips).toHaveCount(2);
    await expect(more).toHaveText("+8 more");
    await expect(input).toBeFocused();
    await more.click();
    await expect(chips).toHaveCount(10);
    await expect(more).toHaveText("Show fewer");
    // Unfolding keeps the focus where the Question is written.
    await expect(input).toBeFocused();
    await window.keyboard.type("How high were the tides?");
    await expect(input).toHaveValue("How high were the tides?");
  } finally {
    await app.close();
  }
});
