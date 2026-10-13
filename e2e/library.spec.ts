import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { totalmem } from "node:os";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { AUTO_CLASSIFICATION } from "../src/core/library/automatic";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  closeSettings,
  useLocalChatModel as connectLocalChatModel,
  coreOf,
  createDataFolder,
  dismissChatSetup,
  editFolderInLibrary,
  expectReadyDocuments,
  launchApp,
  openFolderInLibrary,
  openSettings,
  placeAssistant,
  pointAndClick,
  pointAt,
  removeDataFolder,
  sidebarFolder,
  tabTo,
} from "./app";

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

async function organizationSettings(window: Page) {
  const settings = await openSettings(window, "models");
  await expect(settings.getByTestId("organization-settings")).toBeVisible();
  return settings;
}
async function saveOrganization(window: Page) {
  await window
    .getByTestId("settings")
    .getByRole("button", { name: "Save settings", exact: true })
    .click();
  await expect(window.getByTestId("settings").getByRole("status")).toHaveText("Settings saved.");
  await closeSettings(window);
}

const makeFolders = async (
  window: Page,
  folders: readonly { name: string; description: string }[],
) =>
  (await coreOf(window)).evaluate(async (core, input) => {
    for (const folder of input) await core.createLibraryGroup(folder);
  }, folders);

/** A section of the Library page, by its name. */
const section = (library: Locator, name: string) =>
  library.getByTestId("library-section").filter({
    has: library.page().locator('[data-testid="library-section-name"]', {
      hasText: new RegExp(`^${name}$`),
    }),
  });

/** The row of the Document with this name. */
const rowOf = (scope: Locator, name: string) =>
  scope
    .getByTestId("library-document")
    .filter({ has: scope.page().getByRole("button", { name, exact: true }) });

/** Chooses a Folder in a row's Move to…, from its ⋯. */
async function moveRow(window: Page, row: Locator, folder: string) {
  // Hovered once it holds still: Organize may still be moving rows between sections.
  await row.hover();
  await pointAndClick(window, row.getByTestId("document-file-menu"));
  await pointAndClick(
    window,
    window
      .getByTestId("document-file-actions")
      .filter({ visible: true })
      .getByTestId("document.moveTo"),
  );
  const picker = window.getByTestId("move-to-picker");
  await expect(picker.getByTestId("move-to-search")).toBeFocused();
  await pointAndClick(window, picker.getByTestId("move-to-option").filter({ hasText: folder }));
  await expect(picker).toBeHidden();
}

/** Get info on a row, as ⌘I: its panel. Esc closes it. */
async function infoOf(window: Page, row: Locator): Promise<Locator> {
  await row.locator("[data-row-main]").focus();
  await window.keyboard.press("ControlOrMeta+i");
  const panel = window.getByTestId("info-panel");
  await expect(panel).toBeVisible();
  return panel;
}

/** Organize, beside Not in a Folder's count. */
const organize = (library: Locator) => library.getByTestId("library-organize");

test("folders, tags, search, edits and removal update the index while originals stay in place", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    const path = join(sources, "Quarterly report.txt");
    const text = "Quarterly financial report. Revenue increased this year.";
    await writeFile(path, text);
    await addDocuments(window, [path]);
    await makeFolders(window, [
      { name: "Client work", description: "Reports for my clients" },
      { name: "Research", description: "Papers" },
    ]);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    const folders = window.getByTestId("library-folders");
    await expect(folders.getByTestId("library-folder")).toHaveCount(2);
    await expect(folders.getByTestId("not-in-a-folder")).toBeVisible();

    // Into a Folder with Move to…, then a Tag from the row.
    const row = library.getByTestId("library-document");
    await moveRow(window, row, "Client work");
    await expect(rowOf(section(library, "Client work"), "Quarterly report")).toHaveCount(1);
    await pointAt(window, row);
    await pointAndClick(window, row.getByTestId("document-tags-menu"));
    await expect
      .poll(() =>
        row.getByTestId("document-tags-popover").evaluate((element) => {
          const rect = element.getBoundingClientRect();
          return rect.left >= 0 && rect.right <= globalThis.innerWidth;
        }),
      )
      .toBe(true);
    await row.getByRole("option", { name: "Report", exact: true }).click();
    await window.keyboard.press("Escape");
    await expect(row.getByTestId("document-tags")).toContainText("Report");

    // The Folder's name and description edit under its section, from its menu.
    const form = await editFolderInLibrary(window, "Client work");
    await form.getByLabel("Folder name", { exact: true }).fill("Client reports");
    await form.getByRole("button", { name: "Save folder" }).click();
    await expect(section(library, "Client reports")).toHaveCount(1);
    await expect(form).toHaveCount(0);

    // The Folder opens and closes in place, in the sidebar; the Library stays as it is.
    const clientRow = sidebarFolder(window, "Client reports").getByTestId("folder-row");
    await clientRow.click();
    await expect(folders.getByTestId("document-list-item")).toHaveCount(1);
    await clientRow.click();
    await expect(folders.getByTestId("document-list-item")).toHaveCount(0);
    await expect(section(library, "Client reports")).toHaveCount(1);
    await library.getByLabel("Search the Library").fill("nonsense");
    await expect(row).toHaveCount(0);
    await library.getByLabel("Search the Library").fill("Report");
    await expect(row).toHaveCount(1);
    await library.getByLabel("Search the Library").fill("");

    // Deleting asks nothing: the Folder goes to Recently deleted, its Document to Not in a Folder.
    const head = section(library, "Client reports").getByTestId("library-section-head");
    await pointAt(window, head);
    await pointAndClick(window, head.getByTestId("folder-menu"));
    await pointAndClick(
      window,
      window.getByTestId("folder-actions").filter({ visible: true }).getByTestId("folder.delete"),
    );
    await expect(section(library, "Client reports")).toHaveCount(0);
    await expect(rowOf(section(library, "Not in a Folder"), "Quarterly report")).toHaveCount(1);
    await expect(row.getByTestId("document-tags")).toContainText("Report");
    expect(await readFile(path, "utf8")).toBe(text);
    await window.setViewportSize({ width: 1000, height: 760 });
    // The viewer opens beside the Library: at 1,000px, with the assistant at the bottom (#253).
    await placeAssistant(window, { dock: "bottom" });
    await library.getByRole("button", { name: "Quarterly report", exact: true }).click();
    await expect(window.getByTestId("viewer")).toBeVisible();
    await expect
      .poll(() => library.evaluate((element) => element.scrollWidth <= element.clientWidth))
      .toBe(true);
  } finally {
    await app.close();
  }
});

test("Organize suggests a Folder and Tags, keeps corrections and survives reopening", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    await connectLocalChatModel(window, "main-chat");
    await makeFolders(window, [{ name: "Research", description: "Research papers and reports" }]);
    const settings = await organizationSettings(window);
    const providerId = await window.evaluate(
      async () =>
        (
          await (
            globalThis as unknown as { incarnamind: CoreBridge }
          ).incarnamind.listChatProviders()
        )[0]?.id,
    );
    if (!providerId) throw new Error("No provider");
    await settings.getByLabel("Connection", { exact: true }).selectOption(providerId);
    await settings.getByLabel("Model name", { exact: true }).fill("small-classifier");
    // On by default; turned off here, so the Document waits for Organize.
    const automatic = settings.getByRole("checkbox", {
      name: /^Organize new and changed Documents automatically/,
    });
    await expect(automatic).toBeChecked();
    await automatic.uncheck();
    await saveOrganization(window);
    const path = join(sources, "Research notes.txt");
    await writeFile(path, "Research report about membrane separation and water filtration.");
    await addDocuments(window, [path]);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await pointAndClick(window, organize(library));

    // Suggested: it lists and counts in the Folder, marked, with Organize's Tags.
    const research = section(library, "Research");
    const row = rowOf(research, "Research notes");
    await expect(row.getByTestId("library-suggested")).toHaveText("Suggested");
    await expect(row.getByTestId("document-tags")).toContainText("Report");
    let info = await infoOf(window, row);
    await expect(info.getByTestId("info-organize")).toContainText("Suggested");
    await expect(info.getByTestId("info-organize")).toContainText("small-classifier · Text");
    await window.keyboard.press("Escape");

    // Corrected by hand: its Tags, and out of the Folder.
    await pointAt(window, row);
    await pointAndClick(window, row.getByTestId("document-tags-menu"));
    await row.getByRole("option", { name: "Report", exact: true }).click();
    await row.getByRole("option", { name: "Invoice", exact: true }).click();
    await window.keyboard.press("Escape");
    await moveRow(window, row, "Not in a Folder");
    const loose = section(library, "Not in a Folder");
    const looseRow = rowOf(loose, "Research notes");
    await expect(looseRow).toHaveCount(1);

    // Organize again keeps both.
    await pointAndClick(window, organize(library));
    await expect(library.getByTestId("library-organizing")).toHaveCount(0);
    await expect(looseRow.getByTestId("document-tags")).toContainText("Invoice");
    await expect(looseRow.getByTestId("document-tags")).not.toContainText("Report");
    info = await infoOf(window, looseRow);
    await expect(info.getByTestId("info-organize")).toContainText("Your choice");
    await window.keyboard.press("Escape");

    // A new Tag, made in Manage Tags from the row's Tags, is one Organize can give.
    await pointAt(window, looseRow);
    await pointAndClick(window, looseRow.getByTestId("document-tags-menu"));
    await looseRow.getByTestId("open-tags-dialog").click();
    const dialog = window.getByTestId("tags-dialog");
    await dialog.getByTestId("tag-name-input").fill("Membrane");
    await dialog.getByTestId("tag-description-input").fill("Membrane science");
    await dialog.getByTestId("save-tag").click();
    await dialog.getByRole("button", { name: "Done" }).click();
    await pointAndClick(window, organize(library));
    await expect(looseRow.getByTestId("document-tags")).toContainText("Membrane");
    await library.getByLabel("Search the Library").fill("Membrane");
    await expect(library.getByTestId("library-document")).toHaveCount(1);
  } finally {
    await app.close();
  }
  const reopened = await launchApp(dataDir, { fakeChat: true });
  try {
    await reopened.window.getByTestId("open-library").click();
    const library = reopened.window.getByTestId("library");
    const row = rowOf(section(library, "Not in a Folder"), "Research notes");
    await expect(row.getByTestId("document-tags")).toContainText("Membrane");
    const settings = await organizationSettings(reopened.window);
    await expect(settings.getByLabel("Model name", { exact: true })).toHaveValue(
      "small-classifier",
    );
  } finally {
    await reopened.app.close();
  }
});

test("onboarding offers starter folders, and Organize with no model chosen opens its settings, in Chinese", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true });
  try {
    await window.getByTestId("onboarding-groups").click();
    await expect(window.getByTestId("library")).toBeVisible();
    await expect(window.getByTestId("chat-setup")).not.toBeVisible();
    await window.evaluate(async () =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: "zh-CN" },
      }),
    );
    await window.getByTestId("library").getByRole("button", { name: "添加所选文件夹" }).click();
    await expect(sidebarFolder(window, "研究论文").getByTestId("folder-row")).toBeVisible();
    await pointAndClick(window, organize(window.getByTestId("library")));
    await expect(window.getByTestId("settings-page-title")).toHaveText("模型");
    await expect(window.getByTestId("organization-settings")).toContainText("文档整理");
    await window.getByTestId("settings-close").click();
  } finally {
    await app.close();
  }
});

/** Real HTTP protocol and PDF rendering; only inference is deterministic here. */
test("Auto routes text and scans with tags, reports failures, retries and preserves overrides", async () => {
  const requests: { model: string; images?: string[] }[] = [];
  let models = ["tev1:0.8b", "clef-flash:latest"];
  let fail = false;
  const loaded = new Set<string>();
  const server = createServer(async (request, response) => {
    let raw = "";
    for await (const chunk of request) raw += chunk;
    const body = raw ? JSON.parse(raw) : {};
    response.setHeader("content-type", "application/json");
    if (request.url === "/api/tags")
      response.end(JSON.stringify({ models: models.map((name) => ({ name })) }));
    else if (request.url === "/api/ps")
      response.end(JSON.stringify({ models: [...loaded].map((name) => ({ name })) }));
    else if (request.url === "/api/generate") {
      loaded.delete(body.model);
      response.end(JSON.stringify({ done: true }));
    } else {
      requests.push(body);
      loaded.add(body.model);
      if (fail) {
        response.statusCode = 500;
        response.end(JSON.stringify({ detail: "Temporary test outage" }));
      } else
        response.end(
          JSON.stringify({
            answers: Object.fromEntries(
              Object.entries(body.questions).map(([id, question]) => [
                id,
                (question as { type: string }).type === "choice"
                  ? { type: "choice", choice: "__unsorted__" }
                  : { type: "noul", noul: 0.01 },
              ]),
            ),
          }),
        );
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    const scanPath = join(sources, "Scanned diagram.pdf");
    const textPath = join(sources, "Research notes.txt");
    await writeFile(scanPath, buildPdf([{ image: true }]));
    await writeFile(textPath, "Research about membrane separation and water filtration.");
    await window.evaluate(
      async (paths) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.addDocuments(paths),
      [scanPath, textPath],
    );
    await expect
      .poll(() =>
        window.evaluate(
          async () =>
            (
              await (
                globalThis as unknown as { incarnamind: CoreBridge }
              ).incarnamind.listDocuments()
            ).filter((doc) => doc.status === "ready" || doc.status === "no-text").length,
        ),
      )
      .toBe(2);
    await window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await core.addLibraryStarterGroups(["research", "reports"]);
    });
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    let settings = await organizationSettings(window);
    await settings.getByLabel("Connection", { exact: true }).selectOption("auto");
    await expect(settings.getByLabel("Model name", { exact: true })).toHaveCount(0);
    await settings
      .getByLabel("Local Ollama address", { exact: true })
      .fill(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
    await saveOrganization(window);
    await pointAndClick(window, organize(library));
    const notes = rowOf(library, "Research notes");
    const scan = rowOf(library, "Scanned diagram");
    const organized = async (row: Locator, model: string | RegExp) => {
      const info = await infoOf(window, row);
      await expect(info.getByTestId("info-organize")).toContainText("Organized");
      await expect(info.getByTestId("info-organize")).toContainText(model);
      await window.keyboard.press("Escape");
    };
    await expect(library.getByTestId("library-organizing")).toHaveCount(0);
    await organized(notes, "tev1:0.8b · Text");
    await organized(scan, "clef-flash · Text + PDF pages");
    expect(requests.map((item) => item.model)).toEqual(["tev1:0.8b", "clef-flash"]);
    expect(requests[0]?.images).toBeUndefined();
    expect(requests[1]?.images).toHaveLength(1);

    // A failure says so where it happened, with one way to try again.
    models = [...models, "tev1:4b"];
    fail = true;
    await pointAndClick(window, organize(library));
    await expect(library.getByTestId("library-organize-failed")).toHaveText(
      /^Organize couldn't sort (1 Document|2 Documents) · Try again$/,
    );
    await expect(notes.getByTestId("library-not-sorted")).toHaveAttribute(
      "title",
      /Temporary test outage/,
    );
    const info = await infoOf(window, scan);
    await expect(info.getByTestId("info-organize")).toContainText("Needs attention");
    await window.keyboard.press("Escape");
    fail = false;
    await pointAndClick(
      window,
      library.getByTestId("library-organize-failed").getByRole("button", { name: "Try again" }),
    );
    await expect(library.getByTestId("library-organize-failed")).toHaveCount(0);
    await expect(library.getByTestId("library-organizing")).toHaveCount(0);
    // Auto takes tev1:4b now it's there, unless this computer has too little memory for it
    // (the CI runner has 7 GB): then it keeps to the smaller model, and says why.
    await organized(
      notes,
      totalmem() < AUTO_CLASSIFICATION.smallMemoryBytes
        ? "tev1:0.8b · Text · Smaller model for memory limits"
        : "tev1:4b · Text",
    );
    await organized(scan, /clef-flash/);
    // Explicit Clef override can include or omit PDF images; the choice persists.
    settings = await organizationSettings(window);
    await expect(settings.getByLabel("Connection", { exact: true })).toHaveValue("auto");
    await settings.getByLabel("Connection", { exact: true }).selectOption("ollama");
    await settings.getByLabel("Model name", { exact: true }).fill("clef-flash");
    const images = settings.getByRole("checkbox", { name: /Include up to 3 PDF page images/ });
    await images.check();
    await saveOrganization(window);
    settings = await organizationSettings(window);
    await expect(images).toBeChecked();
    await images.uncheck();
    await saveOrganization(window);
    settings = await organizationSettings(window);
    await expect(images).not.toBeChecked();
    await closeSettings(window);
  } finally {
    await app.close();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("organized folders remain separate from source locations and follow changes in place", async () => {
  const fixtures = [
    [
      "Membrane separation.txt",
      "Research paper: polyamide membranes separate dissolved salts from seawater.",
    ],
    [
      "Seawater desalination.txt",
      "Research report: comparing pressure and salt rejection in filtration experiments.",
    ],
    [
      "October equipment invoice.txt",
      "Finance invoice: laboratory equipment, total GBP 1280 due 30 October.",
    ],
    [
      "Project planning.md",
      "Meetings notes: Alice updates the plan by Friday; Bob reserves the laboratory.",
    ],
    ["文献阅读笔记.txt", "Research notes: 聚酰胺膜的脱盐性能与水通量的研究记录。"],
    ["Quarterly budget.txt", "Finance report: quarterly budget and equipment expenditure."],
  ] as const;
  for (const [name, text] of fixtures) await writeFile(join(sources, name), text);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    await window.evaluate(async (sources) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const provider = await core.saveChatProvider({ kind: "ollama", modelId: "small-classifier" });
      for (const name of ["Research", "Finance", "Meetings"])
        await core.createLibraryGroup({ name, description: `${name} documents` });
      await core.saveLibrarySettings({
        classifier: {
          kind: "chat",
          choice: { providerId: provider.id, modelId: "small-classifier" },
        },
        automatic: true,
      });
      await core.addLinkedFolder(sources);
    }, sources);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await expect(library.getByTestId("library-document")).toHaveCount(6);
    await expect
      .poll(() =>
        window.evaluate(
          async () =>
            (
              await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.getLibrary()
            ).assignments.filter((item) => item.status === "classified").length,
        ),
      )
      .toBe(6);
    await expect(library.getByTestId("library-not-sorted")).toHaveCount(0);
    await expectReadyDocuments(window, 6);
    await window.setViewportSize({ width: 1280, height: 860 });
    const row = rowOf(library, "Membrane separation");
    await moveRow(window, row, "Finance");
    await expect(rowOf(section(library, "Finance"), "Membrane separation")).toHaveCount(1);
    await window.getByRole("button", { name: "Source locations", exact: true }).click();
    await expect(window.getByTestId("folder-item")).toHaveCount(1);
    await window.getByRole("button", { name: "Folders", exact: true }).click();
    await expect(window.getByTestId("library-folders")).toBeVisible();
    const path = join(sources, "Membrane separation.txt");
    const revised = "Research report: membrane invoice for laboratory services. Total GBP 250.";
    await writeFile(path, revised);
    await expect(row.getByTestId("document-tags")).toContainText("Invoice", {
      timeout: 15_000,
    });
    await expect(rowOf(section(library, "Finance"), "Membrane separation")).toHaveCount(1);
    expect(await readFile(path, "utf8")).toBe(revised);
    await expect(window.getByTestId("sidebar-status")).not.toContainText("Tags need a model");
    await window.setViewportSize({ width: 1000, height: 760 });
    // The viewer opens beside the Library: at 1,000px, with the assistant at the bottom (#253).
    await placeAssistant(window, { dock: "bottom" });
    await row.getByRole("button", { name: "Membrane separation", exact: true }).click();
    await expect(window.getByTestId("viewer")).toBeVisible();
    await pointAt(window, row);
    const tagsMenu = row.getByTestId("document-tags-menu");
    const picker = row.getByTestId("document-tags-popover");
    await tagsMenu.click();
    await expect(picker).toBeVisible();
    await window.keyboard.press("Escape");
    await expect(picker).toBeHidden();
    // Esc closes the picker first, even with the focus back on its button, the row still selected.
    await expect(row).toHaveAttribute("aria-selected", "true");
    await tagsMenu.click();
    await expect(picker.getByTestId("tag-picker-input")).toBeFocused();
    await tabTo(window, tagsMenu);
    await window.keyboard.press("Escape");
    await expect(picker).toBeHidden();
    await expect(row).toHaveAttribute("aria-selected", "true");
    await window.keyboard.press("Escape");
    await expect(row).toHaveAttribute("aria-selected", "false");
  } finally {
    await app.close();
  }
});

test("a bulk Organize leaves the Library's sections, filters and sidebar as the core has them", async () => {
  // Organize works through every Document, sending each one's progress (#156): the window
  // gathers those and must still end up showing exactly what the core saved.
  test.setTimeout(240_000);
  const kinds = [
    ["Research paper", "polyamide membranes separate dissolved salts from seawater."],
    ["Research report", "comparing pressure and salt rejection in filtration runs."],
    ["Finance invoice", "laboratory equipment, total GBP 1280 due in October."],
    ["Meetings notes", "Alice updates the plan by Friday; Bob reserves the room."],
    ["Holiday list", "pack sun cream, a hat and the camera."],
  ] as const;
  const count = 200;
  for (let i = 0; i < count; i++) {
    const [name, text] = kinds[i % kinds.length] ?? kinds[0];
    await writeFile(join(sources, `${name} ${i}.txt`), `${name} ${i}: ${text}`);
  }
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    await window.evaluate(async (sources) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const provider = await core.saveChatProvider({ kind: "ollama", modelId: "small-classifier" });
      for (const name of ["Research", "Finance", "Meetings"])
        await core.createLibraryGroup({ name, description: `${name} documents` });
      await core.saveLibrarySettings({
        classifier: {
          kind: "chat",
          choice: { providerId: provider.id, modelId: "small-classifier" },
        },
        automatic: false,
      });
      await core.addLinkedFolder(sources);
    }, sources);
    await expect
      .poll(
        () =>
          window.evaluate(
            async () =>
              (
                await (
                  globalThis as unknown as { incarnamind: CoreBridge }
                ).incarnamind.listDocuments()
              ).filter((doc) => doc.status === "ready").length,
          ),
        { timeout: 120_000 },
      )
      .toBe(count);

    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await expect(library.getByTestId("library-count")).toHaveText("200");
    const loose = section(library, "Not in a Folder");
    await expect(loose).toHaveAttribute("data-count", "200");
    await pointAndClick(window, organize(library));
    await expect(library.getByTestId("library-organizing")).toBeVisible();
    await expect(library.getByTestId("library-organizing")).toHaveCount(0, { timeout: 120_000 });
    await expect(organize(library)).toBeVisible();

    // What the core saved, Folder by Folder and Tag by Tag.
    const saved = await window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const [library, documents, tags] = await Promise.all([
        core.getLibrary(),
        core.listDocuments(),
        core.listTags(),
      ]);
      const folders: Record<string, number> = { "Not in a Folder": 0 };
      for (const group of library.groups) folders[group.name] = 0;
      for (const item of library.assignments) {
        const name =
          library.groups.find((group) => group.id === item.groupId)?.name ?? "Not in a Folder";
        folders[name] = (folders[name] ?? 0) + 1;
      }
      const byTag: Record<string, number> = {};
      for (const tag of tags)
        byTag[tag.name] = documents.filter((doc) =>
          doc.tags.some((link) => link.tagId === tag.id),
        ).length;
      return {
        folders,
        byTag,
        classified: library.assignments.filter((item) => item.status === "classified").length,
        tagged: documents.filter((doc) => doc.tagging === "tagged").length,
      };
    });
    expect(saved).toEqual({
      folders: { Research: 80, Finance: 40, Meetings: 40, "Not in a Folder": 40 },
      byTag: expect.objectContaining({ Paper: 40, Report: 40, Invoice: 40, Notes: 40, Book: 0 }),
      classified: count,
      tagged: count,
    });

    // The Library's sections and the sidebar's Folders: the same counts.
    const folders = window.getByTestId("library-folders");
    const folderRow = (name: string) =>
      folders.getByTestId("library-folder").filter({
        has: window.locator(':scope > div [data-testid="row-text"]', { hasText: name }),
      });
    const notInAFolder = folders.getByTestId("not-in-a-folder");
    for (const [name, total] of Object.entries(saved.folders)) {
      await expect(section(library, name)).toHaveAttribute("data-count", String(total));
      if (name === "Not in a Folder")
        await expect(notInAFolder).toHaveAttribute("data-count", String(total));
      else await expect(folderRow(name).getByTestId("browse-count")).toHaveText(String(total));
    }
    // Closed, a Folder lists nothing; Not in a Folder lists its first few, then "Show N more".
    await expect(folders.getByTestId("document-list-item")).toHaveCount(5);
    await expect(notInAFolder.getByTestId("show-more-documents")).toHaveText(
      "Show 35 more Documents",
    );
    // Opened, a Folder lists its first few too; "Show 50 more" lists more where they are.
    const research = folderRow("Research");
    await research.getByTestId("folder-row").click();
    await expect(research.getByTestId("document-list-item")).toHaveCount(5);
    await research.getByTestId("show-more-documents").click();
    await expect(research.getByTestId("document-list-item")).toHaveCount(55);
    await expect(
      folders.locator('[data-testid="document-list-item"]:not([data-tagging="tagged"])'),
    ).toHaveCount(0);
    // Each section lists its first 8; none couldn't be sorted.
    await expect(library.getByTestId("library-document")).toHaveCount(4 * 8);
    await expect(library.getByTestId("library-not-sorted")).toHaveCount(0);

    // A Folder from the sidebar: its section, all of it, and a Tags filter counting among it.
    await openFolderInLibrary(window, "Research");
    const inResearch = section(library, "Research");
    await expect(inResearch.getByTestId("library-section-name")).toBeFocused();
    await pointAndClick(window, inResearch.getByTestId("library-show-all"));
    await expect(inResearch.getByTestId("library-document")).toHaveCount(80);
    await pointAndClick(window, library.getByTestId("library-add-filter"));
    await pointAndClick(
      window,
      window.locator('[data-testid="library-filter-kind"][data-facet="folder"]'),
    );
    await pointAndClick(
      window,
      window.getByTestId("library-filter-option").filter({ hasText: "Research" }),
    );
    await window.keyboard.press("Escape");
    await pointAndClick(window, library.getByTestId("library-add-filter"));
    await pointAndClick(
      window,
      window.locator('[data-testid="library-filter-kind"][data-facet="tag"]'),
    );
    const tagOptions = window
      .getByTestId("library-add-filter-menu")
      .getByTestId("library-filter-option");
    await expect(tagOptions.filter({ hasText: "Paper" })).toHaveAttribute("data-count", "40");
    await expect(tagOptions.filter({ hasText: "Report" })).toHaveAttribute("data-count", "40");
    await expect(tagOptions.filter({ hasText: "Invoice" })).toHaveAttribute("data-count", "0");
    await tagOptions.filter({ hasText: "Paper" }).click();
    await window.keyboard.press("Escape");
    await expect(library.getByTestId("library-document")).toHaveCount(40);
    const remove = library.getByTestId("library-filter-remove");
    while ((await remove.count()) > 0) await pointAndClick(window, remove.first());

    // The sidebar's Tags: the same counts, and a Tag shows its Documents across Folders.
    const browse = window.getByTestId("browse-views");
    await browse.getByRole("button", { name: "Tags", exact: true }).click();
    const tagRow = (name: string) =>
      window
        .getByTestId("sidebar-tags")
        .getByTestId("sidebar-tag")
        .filter({
          has: window.locator(':scope > div [data-testid="row-text"]', { hasText: name }),
        });
    for (const name of ["Paper", "Report", "Invoice", "Notes", "Book"])
      await expect(tagRow(name).getByTestId("browse-count")).toHaveText(String(saved.byTag[name]));
    await tagRow("Invoice")
      .locator(":scope > div")
      .getByRole("button", { name: /^Invoice/ })
      .click();
    await expect(library.getByTestId("library-heading")).toContainText("Library");
    await expect(library.locator('[data-testid="library-filter"][data-facet="tag"]')).toHaveText(
      "Tags: Invoice",
    );
    await expect(
      library
        .getByTestId("library-document")
        .getByTestId("tag-chip")
        .filter({ hasText: "Invoice" }),
    ).toHaveCount(40);
    // Elsewhere on the row than the first click, which a double-click would be.
    await tagRow("Invoice")
      .locator(":scope > div")
      .getByRole("button", { name: /^Invoice/ })
      .click({ position: { x: 120, y: 12 } });
    await expect(library.getByTestId("library-filter")).toHaveCount(0);

    // Source locations: every Document's row, each done.
    await browse.getByRole("button", { name: "Source locations", exact: true }).click();
    const rows = window.getByTestId("sidebar").getByTestId("document-list-item");
    await expect(rows).toHaveCount(count);
    await expect(rows.and(window.locator('[data-tagging="tagged"]'))).toHaveCount(count);
  } finally {
    await app.close();
  }
});
