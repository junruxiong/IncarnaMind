import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { totalmem } from "node:os";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { AUTO_CLASSIFICATION } from "../src/core/library/automatic";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  closeSettings,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
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
  await window
    .getByTestId("library")
    .getByRole("button", { name: "Organization", exact: true })
    .click();
  return window.getByTestId("settings");
}
async function saveOrganization(window: Page) {
  await window
    .getByTestId("settings")
    .getByRole("button", { name: "Save settings", exact: true })
    .click();
  await expect(window.getByTestId("settings").getByRole("status")).toHaveText("Settings saved.");
  await closeSettings(window);
}

test("folders, tags, search, edits and removal update the index while originals stay in place", async () => {
  const { app, window } = await launchApp(dataDir);
  try {
    await dismissChatSetup(window);
    const path = join(sources, "Quarterly report.txt");
    const text = "Quarterly financial report. Revenue increased this year.";
    await writeFile(path, text);
    await addDocuments(window, [path]);
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await library.getByRole("checkbox", { name: /^Meeting notes/ }).uncheck();
    await library.getByRole("button", { name: "Add selected folders" }).click();
    const folders = window.getByTestId("library-folders");
    await expect(folders.getByTestId("library-folder")).toHaveCount(5);
    await library.getByRole("button", { name: "New folder", exact: true }).click();
    await library.getByLabel("Folder name", { exact: true }).fill("Client work");
    await library.getByLabel("Description", { exact: true }).fill("Reports for my clients");
    await library.getByRole("button", { name: "Save folder" }).click();
    const row = library.getByTestId("library-document");
    const assignment = library.getByRole("combobox", { name: "Folder for Quarterly report" });
    await assignment.selectOption({ label: "Client work" });
    await expect(row).toContainText("Your choice");
    await row.getByTestId("document-tags-menu").click();
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
    await folders.getByRole("button", { name: /^Client work/ }).click();
    await expect(library.getByRole("heading", { name: "Client work", exact: true })).toBeVisible();
    await library.getByRole("button", { name: "Edit folder" }).click();
    await library.getByLabel("Folder name", { exact: true }).fill("Client reports");
    await library.getByRole("button", { name: "Save folder" }).click();
    await expect(
      library.getByRole("heading", { name: "Client reports", exact: true }),
    ).toBeVisible();
    await folders.getByRole("button", { name: "Collapse Client reports" }).click();
    await expect(folders.getByTestId("document-list-item")).toHaveCount(0);
    await folders.getByRole("button", { name: "Expand Client reports" }).click();
    await expect(folders.getByTestId("document-list-item")).toHaveCount(1);
    await library.getByLabel("Find by name or tag").fill("nonsense");
    await expect(row).toHaveCount(0);
    await library.getByLabel("Find by name or tag").fill("Report");
    await expect(row).toHaveCount(1);
    await library.getByRole("button", { name: "Delete folder", exact: true }).click();
    await library.getByRole("button", { name: "Delete folder", exact: true }).last().click();
    await expect(assignment).toHaveValue("");
    await expect(row.getByTestId("document-tags")).toContainText("Report");
    expect(await readFile(path, "utf8")).toBe(text);
    await window.setViewportSize({ width: 1000, height: 760 });
    await window.screenshot({ path: "/tmp/incarnamind-organize-manual.png" });
    await library.getByRole("button", { name: "Quarterly report", exact: true }).click();
    await expect(window.getByTestId("viewer")).toBeVisible();
    await expect
      .poll(() => library.evaluate((element) => element.scrollWidth <= element.clientWidth))
      .toBe(true);
    await window.screenshot({ path: "/tmp/incarnamind-organize-viewer.png" });
  } finally {
    await app.close();
  }
});

test("one Organize action assigns folder and tags, preserves corrections and survives reopening", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  try {
    await dismissChatSetup(window);
    await connectLocalChatModel(window, "main-chat");
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await library.getByRole("button", { name: "New folder", exact: true }).click();
    await library.getByLabel("Folder name", { exact: true }).fill("Research");
    await library.getByLabel("Description", { exact: true }).fill("Research papers and reports");
    await library.getByRole("button", { name: "Save folder" }).click();
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
    await saveOrganization(window);
    const path = join(sources, "Research notes.txt");
    await writeFile(path, "Research report about membrane separation and water filtration.");
    await addDocuments(window, [path]);
    await library.getByRole("button", { name: "Organize", exact: true }).click();
    const row = library.getByTestId("library-document");
    await expect(row.locator("summary")).toHaveText("Organized");
    await expect(row.getByTestId("document-tags")).toContainText("Report");
    const assignment = row.getByRole("combobox", { name: /^Folder for/ });
    await expect(assignment.locator("option:checked")).toHaveText("Research");
    await expect(library.getByLabel("Model name", { exact: true })).toHaveCount(0);
    await row.getByTestId("document-tags-menu").click();
    await row.getByRole("option", { name: "Report", exact: true }).click();
    await row.getByRole("option", { name: "Invoice", exact: true }).click();
    await window.keyboard.press("Escape");
    await assignment.selectOption("");
    await library.getByRole("button", { name: "Organize", exact: true }).click();
    await expect(row.locator("summary")).toHaveText("Your choice");
    await expect(assignment).toHaveValue("");
    await expect(row.getByTestId("document-tags")).toContainText("Invoice");
    await expect(row.getByTestId("document-tags")).not.toContainText("Report");
    await library.getByRole("button", { name: "Manage tags" }).click();
    const dialog = window.getByTestId("tags-dialog");
    await dialog.getByTestId("tag-name-input").fill("Membrane");
    await dialog.getByTestId("tag-description-input").fill("Membrane science");
    await dialog.getByTestId("save-tag").click();
    await dialog.getByRole("button", { name: "Done" }).click();
    await library.getByRole("button", { name: "Organize", exact: true }).click();
    await expect(row.getByTestId("document-tags")).toContainText("Membrane");
    await library.getByLabel("Find by name or tag").fill("Membrane");
    await expect(row).toHaveCount(1);
    await window.screenshot({ path: "/tmp/incarnamind-organize-tags.png" });
  } finally {
    await app.close();
  }
  const reopened = await launchApp(dataDir, { fakeChat: true });
  try {
    await reopened.window.getByTestId("open-library").click();
    const row = reopened.window.getByTestId("library-document");
    await expect(row.getByRole("combobox", { name: /^Folder for/ })).toHaveValue("");
    await expect(row.getByTestId("document-tags")).toContainText("Membrane");
    const settings = await organizationSettings(reopened.window);
    await expect(settings.getByLabel("Model name", { exact: true })).toHaveValue(
      "small-classifier",
    );
  } finally {
    await reopened.app.close();
  }
});

test("onboarding offers starter folders and Chinese organization settings", async () => {
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
    await window.getByTestId("library").getByRole("button", { name: "添加选中的文件夹" }).click();
    await expect(
      window.getByTestId("library-folders").getByRole("button", { name: /^研究论文/ }),
    ).toBeVisible();
    await window
      .getByTestId("library")
      .getByRole("button", { name: "文档整理", exact: true })
      .click();
    await expect(window.getByTestId("settings-page-title")).toHaveText("模型");
    await expect(window.getByTestId("organization-settings")).toContainText("文档整理");
    await window.getByTestId("settings-close").click();
    await window.screenshot({ path: "/tmp/incarnamind-organize-chinese.png" });
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
    await window.getByTestId("open-library").click();
    const library = window.getByTestId("library");
    await library.getByRole("button", { name: "Add selected folders" }).click();
    let settings = await organizationSettings(window);
    await settings.getByLabel("Connection", { exact: true }).selectOption("auto");
    await expect(settings.getByLabel("Model name", { exact: true })).toHaveCount(0);
    await settings
      .getByLabel("Local Ollama address", { exact: true })
      .fill(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
    await saveOrganization(window);
    await library.getByRole("button", { name: "Organize", exact: true }).click();
    const notes = library
      .getByTestId("library-document")
      .filter({ has: window.getByRole("button", { name: "Research notes", exact: true }) });
    const scan = library
      .getByTestId("library-document")
      .filter({ has: window.getByRole("button", { name: "Scanned diagram", exact: true }) });
    await expect(notes.locator("summary")).toHaveText("Organized");
    await expect(scan.locator("summary")).toHaveText("Organized");
    await notes.locator("summary").click();
    await scan.locator("summary").click();
    await expect(notes).toContainText("tev1:0.8b · Text");
    await expect(scan).toContainText("clef-flash · Text + PDF pages");
    expect(requests.map((item) => item.model)).toEqual(["tev1:0.8b", "clef-flash"]);
    expect(requests[0]?.images).toBeUndefined();
    expect(requests[1]?.images).toHaveLength(1);
    models = [...models, "tev1:4b"];
    fail = true;
    await library.getByRole("button", { name: "Organize", exact: true }).click();
    await expect(notes).toContainText("Temporary test outage");
    await expect(scan.locator("summary")).toHaveText("Needs attention");
    fail = false;
    await library.getByRole("button", { name: "Organize", exact: true }).click();
    // Auto takes tev1:4b now it's there, unless this computer has too little memory for it
    // (the CI runner has 7 GB): then it keeps to the smaller model, and says why.
    await expect(notes).toContainText(
      totalmem() < AUTO_CLASSIFICATION.smallMemoryBytes
        ? "tev1:0.8b · Text · Smaller model for memory limits"
        : "tev1:4b · Text",
    );
    await expect(scan.locator("summary")).toHaveText("Organized");
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
    await window.screenshot({ path: "/tmp/incarnamind-organize-auto.png" });
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
    await expect(library.locator("summary").filter({ hasText: /^Organized$/ })).toHaveCount(6);
    await expect(
      window.locator('[data-testid="document-list-item"][data-status="ready"]'),
    ).toHaveCount(6);
    await window.setViewportSize({ width: 1280, height: 860 });
    await window.screenshot({ path: "/tmp/incarnamind-organize-overview.png" });
    const row = library
      .getByTestId("library-document")
      .filter({ has: window.getByRole("button", { name: "Membrane separation", exact: true }) });
    await row.getByRole("combobox", { name: /^Folder for/ }).selectOption({ label: "Finance" });
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
    await expect(
      row.getByRole("combobox", { name: /^Folder for/ }).locator("option:checked"),
    ).toHaveText("Finance");
    expect(await readFile(path, "utf8")).toBe(revised);
    await expect(window.getByTestId("sidebar-status")).not.toContainText("Tags need a model");
    await window.setViewportSize({ width: 1000, height: 760 });
    await row.getByRole("button", { name: "Membrane separation", exact: true }).click();
    await expect(window.getByTestId("viewer")).toBeVisible();
    await row.getByTestId("document-tags-menu").click();
    await expect(row.getByTestId("document-tags-popover")).toBeVisible();
    await window.keyboard.press("Escape");
    await expect(row.getByTestId("document-tags-popover")).toBeHidden();
    await window.screenshot({ path: "/tmp/incarnamind-organize-compact.png" });
  } finally {
    await app.close();
  }
});

test("a bulk Organize leaves the Library's counts, filters and sidebar as the core has them", async () => {
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
    const status = library.getByTestId("library-count");
    await expect(status).toHaveText("200 Documents");
    const organize = library.getByRole("button", { name: "Organize", exact: true });
    await organize.click();
    await expect(library.getByRole("button", { name: "Organizing…", exact: true })).toBeDisabled();
    await expect(status).toHaveText("200 Documents", { timeout: 120_000 });
    await expect(organize).toBeEnabled();

    // What the core saved, Folder by Folder and Tag by Tag.
    const saved = await window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const [library, documents, tags] = await Promise.all([
        core.getLibrary(),
        core.listDocuments(),
        core.listTags(),
      ]);
      const folders: Record<string, number> = { Unsorted: 0 };
      for (const group of library.groups) folders[group.name] = 0;
      for (const item of library.assignments) {
        const name = library.groups.find((group) => group.id === item.groupId)?.name ?? "Unsorted";
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
      folders: { Research: 80, Finance: 40, Meetings: 40, Unsorted: 40 },
      byTag: expect.objectContaining({ Paper: 40, Report: 40, Invoice: 40, Notes: 40, Book: 0 }),
      classified: count,
      tagged: count,
    });

    // The sidebar's Folders: the same counts, and every row listed done.
    const folders = window.getByTestId("library-folders");
    const folderRow = (name: string) =>
      folders.getByTestId("library-folder").filter({
        has: window.locator(':scope > div [data-testid="row-text"]', { hasText: name }),
      });
    for (const [name, total] of Object.entries(saved.folders))
      await expect(folderRow(name).getByTestId("browse-count")).toHaveText(String(total));
    // Each Folder lists its first 50 under it.
    await expect(folders.getByTestId("document-list-item")).toHaveCount(50 + 3 * 40);
    await expect(
      folders.locator('[data-testid="document-list-item"]:not([data-tagging="tagged"])'),
    ).toHaveCount(0);
    // The Library's rows, the first 100: all organized.
    await expect(library.getByTestId("library-document")).toHaveCount(100);
    await expect(library.locator("summary").filter({ hasText: /^Organized$/ })).toHaveCount(100);

    // A Folder: its count, its rows in it, and its Tags filter counting them.
    await folderRow("Research")
      .locator(":scope > div")
      .getByRole("button", { name: /^Research/ })
      .click();
    await expect(library.getByRole("heading", { name: "Research", exact: true })).toBeVisible();
    await expect(status).toHaveText("80 Documents");
    await expect(library.getByTestId("library-document")).toHaveCount(80);
    await expect(
      library.getByRole("combobox", { name: /^Folder for/ }).locator("option:checked"),
    ).toHaveText(Array.from({ length: 80 }, () => "Research"));
    await library.locator('[data-testid="library-filter"][data-facet="tag"]').click();
    const tagOptions = library.locator(
      '[data-testid="library-filter-menu"][data-facet="tag"] [data-testid="library-filter-option"]',
    );
    await expect(tagOptions.filter({ hasText: "Paper" })).toHaveAttribute("data-count", "40");
    await expect(tagOptions.filter({ hasText: "Report" })).toHaveAttribute("data-count", "40");
    await expect(tagOptions.filter({ hasText: "Invoice" })).toHaveCount(0);
    await tagOptions.filter({ hasText: "Paper" }).click();
    await window.keyboard.press("Escape");
    await expect(status).toHaveText("40 of 80 Documents");
    await expect(library.getByTestId("library-document")).toHaveCount(40);
    await library.getByTestId("library-filters-clear").click();
    await expect(status).toHaveText("80 Documents");

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
    await expect(library.getByRole("heading", { name: "All Documents" })).toBeVisible();
    await expect(status).toHaveText("40 of 200 Documents");
    await expect(
      library
        .getByTestId("library-document")
        .getByTestId("tag-chip")
        .filter({ hasText: "Invoice" }),
    ).toHaveCount(40);
    await tagRow("Invoice")
      .locator(":scope > div")
      .getByRole("button", { name: /^Invoice/ })
      .click();
    await expect(status).toHaveText("200 Documents");

    // Source locations: every Document's row, each done.
    await browse.getByRole("button", { name: "Source locations", exact: true }).click();
    const rows = window.getByTestId("sidebar").getByTestId("document-list-item");
    await expect(rows).toHaveCount(count);
    await expect(rows.and(window.locator('[data-tagging="tagged"]'))).toHaveCount(count);
  } finally {
    await app.close();
  }
});
