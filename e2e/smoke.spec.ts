import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  dragBy,
  launchApp,
  openDocumentMenu,
  openSettings,
  openViewer,
  removeDataFolder,
  showSettingsPage,
  widthOf,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("a Mind created before quitting is still there after reopening the app", async () => {
  // First run: create a Mind. It appears in the sidebar and opens in the centre.
  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  await first.window.getByTestId("new-mind").click();

  const created = first.window.getByTestId("mind-list-item");
  await expect(created).toHaveCount(1);
  const mindId = await created.getAttribute("data-mind-id");
  expect(mindId).toBeTruthy();
  await expect(first.window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${mindId}`);
  await expect(first.window.getByTestId("mind-title")).toBeVisible();
  await first.app.close();

  // Second run on the same data folder: the Mind is listed and opens.
  const second = await launchApp(dataDir);
  const restored = second.window.getByTestId("mind-list-item");
  await expect(restored).toHaveCount(1);
  await expect(restored).toHaveAttribute("data-mind-id", `${mindId}`);
  await restored.click();
  await expect(second.window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${mindId}`);
  await expect(second.window.getByTestId("mind-title")).toBeVisible();
  await second.app.close();
});

test("what was typed in a Mind, and its title, are still there after reopening the app", async () => {
  const paragraphs = ["Typed before quitting.", "A second paragraph."];

  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  await first.window.getByTestId("new-mind").click();
  await first.window.getByTestId("mind-title").fill("Field notes");
  const editor = first.window.getByTestId("mind-editor");
  await editor.click();
  await first.window.keyboard.type(paragraphs[0] ?? "");
  await first.window.keyboard.press("Enter");
  await first.window.keyboard.type(paragraphs[1] ?? "");
  await expect(editor.locator("p")).toHaveText(paragraphs);
  await expect(first.window.getByTestId("mind-list-item")).toHaveText("Field notes");
  await first.app.close();

  const second = await launchApp(dataDir);
  const item = second.window.getByTestId("mind-list-item");
  await expect(item).toHaveText("Field notes");
  await item.click();
  await expect(second.window.getByTestId("mind-title")).toHaveValue("Field notes");
  await expect(second.window.getByTestId("mind-editor").locator("p")).toHaveText(paragraphs);
  await second.app.close();
});

test("the sidebar lists the most recently edited Mind first, and a deleted Mind leaves it", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  await dismissChatSetup(window);
  const items = window.getByTestId("mind-list-item");
  for (const title of ["Older", "Newer"]) {
    await window.getByTestId("new-mind").click();
    await window.getByTestId("mind-title").fill(title);
    await expect(items.first()).toHaveText(title);
  }
  await expect(items).toHaveText(["Newer", "Older"]);

  // Editing the older Mind moves it to the top.
  await items.filter({ hasText: "Older" }).click();
  await window.getByTestId("mind-editor").click();
  await window.keyboard.type("An edit");
  await expect(items).toHaveText(["Older", "Newer"]);

  // Deleting it asks first, then removes it from the sidebar and closes it.
  const older = window.getByRole("listitem").filter({ hasText: "Older" });
  await older.hover();
  await older.getByTestId("delete-mind").click();
  await window.getByTestId("confirm-delete-mind").click();
  await expect(items).toHaveText(["Newer"]);
  await expect(window.getByTestId("mind-pane")).toHaveCount(0);
  await first.app.close();

  // It stays deleted after a restart.
  const second = await launchApp(dataDir);
  await expect(second.window.getByTestId("mind-list-item")).toHaveText(["Newer"]);
  await second.app.close();
});

test("the Document viewer is hidden until opened, resizes from its left edge and keeps its width", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  await dismissChatSetup(window);
  const viewer = window.getByTestId("viewer");
  const mindArea = window.getByTestId("mind-area");

  // Closed on launch: no panel at all, and the Mind area takes the rest of the width.
  await expect(viewer).toHaveCount(0);
  const fullWidth = await widthOf(mindArea);

  // Opening it narrows the Mind area.
  await openViewer(window);
  await expect(viewer).toBeVisible();
  expect(await widthOf(mindArea)).toBeLessThan(fullWidth);

  // Dragging the left edge 100px to the left widens the panel by 100px.
  const openedWidth = await widthOf(viewer);
  await dragBy(window, window.getByTestId("viewer-resize"), -100);
  await expect.poll(() => widthOf(viewer)).toBe(openedWidth + 100);

  // Esc closes it and the Mind area gets its full width back.
  await window.keyboard.press("Escape");
  await expect(viewer).toHaveCount(0);
  expect(await widthOf(mindArea)).toBe(fullWidth);
  await first.app.close();

  // The width is a per-device setting, so it survives a restart. The close button closes it.
  const second = await launchApp(dataDir);
  await expect(second.window.getByTestId("viewer")).toHaveCount(0);
  await openViewer(second.window);
  await expect.poll(() => widthOf(second.window.getByTestId("viewer"))).toBe(openedWidth + 100);
  await second.window.getByTestId("viewer-close").click();
  await expect(second.window.getByTestId("viewer")).toHaveCount(0);
  await second.app.close();
});

test("added files are processed, show as ready in the sidebar, and can be deleted", async () => {
  // The User's own files live outside the data folder.
  const sources = await createDataFolder();
  try {
    const notes = join(sources, "Reading notes.txt");
    const report = join(sources, "Report.pdf");
    await writeFile(notes, "Attention lets a model focus on the most relevant words.\n");
    await writeFile(
      report,
      buildPdf([{ lines: ["Quarterly report"] }, { lines: ["Revenue grew by ten percent."] }]),
    );

    const first = await launchApp(dataDir);
    const { window } = first;
    await dismissChatSetup(window);
    await window.getByTestId("add-documents-input").setInputFiles([notes, report]);

    const items = window.getByTestId("document-list-item");
    await expect(items).toHaveCount(2);
    const notesItem = items.filter({ hasText: "Reading notes" });
    const reportItem = items.filter({ hasText: "Report" });
    await expect(notesItem).toHaveAttribute("data-status", "ready");
    await expect(reportItem).toHaveAttribute("data-status", "ready");
    await expect(notesItem.getByTestId("document-status")).toHaveText("Ready");
    await first.app.close();

    // After a restart both are still there. Deleting one (from its menu) asks first, then removes it.
    const second = await launchApp(dataDir);
    const restored = second.window.getByTestId("document-list-item");
    await expect(restored).toHaveCount(2);
    const doomed = restored.filter({ hasText: "Reading notes" });
    await openDocumentMenu(doomed);
    await doomed.getByTestId("delete-document").click();
    await second.window.getByTestId("confirm-delete-document").click();
    await expect(restored).toHaveCount(1);
    await expect(restored).toHaveAttribute("data-status", "ready");
    await second.app.close();
  } finally {
    await removeDataFolder(sources);
  }
});

test("the experimental ChatGPT plan is off by default, and turning it on shows the warning and the sign-in", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await openSettings(window, "chat-model");

  const experimental = window.getByTestId("experimental-settings");
  const toggle = experimental.getByTestId("codex-switch");
  await expect(toggle).toBeVisible();
  await expect(toggle).not.toBeChecked();
  await expect(experimental.getByTestId("codex-warning")).toHaveCount(0);
  await expect(experimental.getByTestId("codex-sign-in")).toHaveCount(0);

  // Turning it on explains what it is before anything can be signed in. Nothing is signed in here.
  await toggle.check();
  const warning = experimental.getByTestId("codex-warning");
  await expect(warning).toBeVisible();
  await expect(warning).toContainText("isn't an official OpenAI integration");
  await expect(warning).toContainText("may block it");
  await expect(warning).toContainText("usage limits");
  await expect(experimental.getByTestId("codex-sign-in")).toBeVisible();
  await expect(experimental.getByTestId("codex-account")).toHaveText("Not signed in.");
  await app.close();
});

test("first-run chat setup appears on a fresh data folder and can be set up later", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;

  // No provider is preselected, and no key is needed to get past this screen. Where
  // Answers come from is one choice: local models, an API key, or the ChatGPT plan.
  const setup = window.getByTestId("chat-setup");
  await expect(setup).toBeVisible();
  await expect(setup.getByTestId("chat-provider-choices").getByRole("radio")).toHaveCount(3);
  await expect(setup.getByRole("radio", { checked: true })).toHaveCount(0);
  // Choosing an API key offers the four kinds of provider, none chosen yet.
  await setup.getByTestId("chat-choice-api-key").check();
  const providerChoices = setup.getByTestId("provider-form").getByRole("radio");
  await expect(providerChoices).toHaveCount(4);
  await expect(
    setup.getByTestId("provider-form").getByRole("radio", { checked: true }),
  ).toHaveCount(0);

  await setup.getByTestId("chat-setup-later").click();
  await expect(setup).toBeHidden();

  // Notes and Documents work without a provider; Questions explain what to configure.
  await window.getByTestId("new-mind").click();
  await expect(window.getByTestId("mind-pane")).toBeVisible();
  await expect(window.getByTestId("chat-readiness")).toBeVisible();
  await first.app.close();

  // "Set up later" is remembered.
  const second = await launchApp(dataDir);
  await second.window.getByTestId("mind-list-item").click();
  await expect(second.window.getByTestId("chat-readiness")).toBeVisible();
  await expect(second.window.getByTestId("chat-setup")).toBeHidden();

  // The notice opens Settings on its Chat model page, where a provider can be set up.
  await second.window.getByTestId("chat-readiness").getByRole("button").click();
  await expect(second.window.getByTestId("settings")).toHaveAttribute("data-page", "chat-model");
  await expect(second.window.getByTestId("chat-model-settings")).toBeVisible();
  // What is sent to other services lives on the Privacy page.
  await showSettingsPage(second.window, "privacy");
  await expect(second.window.getByTestId("consent-settings")).toBeVisible();
  await second.app.close();
});

test("Documents are filed in nested Folders, shown inside them, and kept when their Folder is deleted", async () => {
  const sources = await createDataFolder();
  try {
    const paper = join(sources, "Paper.txt");
    const notes = join(sources, "Loose notes.txt");
    await writeFile(paper, "A paper about attention.\n");
    await writeFile(notes, "Notes kept for later.\n");

    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    // Wait until both are ready, as the User would see. (The "tagging waits for a model"
    // notice that then appears sits in the sidebar's footer, so it no longer moves the tree.)
    await addDocuments(window, [paper, notes]);
    const documents = window.getByTestId("document-list-item");
    const paperItem = documents.filter({ hasText: "Paper" });
    const notesItem = documents.filter({ hasText: "Loose notes" });

    // Create a Folder, then a sub-Folder inside it.
    await window.getByTestId("new-folder").click();
    await window.getByTestId("folder-name-input").fill("Projects");
    await window.keyboard.press("Enter");
    const folders = window.getByTestId("folder-item");
    const projects = folders.filter({ hasText: "Projects" });
    await expect(projects).toHaveCount(1);
    await projects.getByTestId("new-subfolder").click();
    await window.getByTestId("folder-name-input").fill("2026");
    await window.keyboard.press("Enter");
    const year = folders.filter({ hasText: "2026" });
    await expect(year).toHaveCount(1);
    const projectsId = await projects.getAttribute("data-folder-id");
    const yearId = await year.getAttribute("data-folder-id");

    // Move the paper into the sub-Folder with "Move to…", and drag the notes onto the parent.
    await paperItem.getByTestId("move-document").click();
    await window.getByRole("menuitemradio", { name: "2026" }).click();
    await expect(paperItem).toHaveAttribute("data-folder-id", `${yearId}`);
    await notesItem.dragTo(projects.getByTestId("folder-toggle"));
    await expect(notesItem).toHaveAttribute("data-folder-id", `${projectsId}`);

    // Each shows inside its Folder, one step deeper: the notes in Projects, the paper in
    // 2026 inside Projects.
    await expect(projects).toHaveAttribute("data-depth", "0");
    await expect(year).toHaveAttribute("data-depth", "1");
    await expect(notesItem).toHaveAttribute("data-depth", "1");
    await expect(paperItem).toHaveAttribute("data-depth", "2");
    // Folding the parent hides what's inside it, its sub-Folder's too; unfolding shows it again.
    const toggle = projects.getByTestId("folder-toggle");
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "false");
    await expect(year).toHaveCount(0);
    await expect(documents).toHaveCount(0);
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
    await expect(documents).toHaveCount(2);

    // Deleting the parent asks first. Both Documents stay listed, unfiled, at the top level.
    await projects.getByTestId("delete-folder").click();
    await window.getByTestId("confirm-delete-folder").click();
    await expect(folders).toHaveCount(0);
    await expect(documents).toHaveCount(2);
    await expect(paperItem).toHaveAttribute("data-folder-id", "");
    await expect(notesItem).toHaveAttribute("data-folder-id", "");
    await expect(paperItem).toHaveAttribute("data-depth", "0");
    await app.close();
  } finally {
    await removeDataFolder(sources);
  }
});
