import { mkdir, realpath, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  dragBy,
  launchApp,
  linkFolderFromSidebar,
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

  // Deleting it asks first, then removes it from the sidebar and closes its tab: the
  // other open Mind shows instead.
  const olderId = await items.filter({ hasText: "Older" }).getAttribute("data-mind-id");
  const older = window.getByRole("listitem").filter({ hasText: "Older" });
  await older.hover();
  await older.getByTestId("delete-mind").click();
  await window.getByTestId("confirm-delete-mind").click();
  await expect(items).toHaveText(["Newer"]);
  await expect(window.getByTestId("mind-tab-title")).toHaveText(["Newer"]);
  await expect(window.getByTestId("mind-pane")).not.toHaveAttribute("data-mind-id", `${olderId}`);
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

  // Opening it narrows the Mind area: it opens at half the card (DESIGN.md: 8px from the
  // sidebar and from the window's right edge), which it shares with the Mind about evenly.
  await openViewer(window);
  await expect(viewer).toBeVisible();
  expect(await widthOf(mindArea)).toBeLessThan(fullWidth);
  const sidebarWidth = await widthOf(window.getByTestId("sidebar"));
  expect(sidebarWidth).toBe(248);
  const halfRoom = await window.evaluate(
    (sidebar) => Math.round((globalThis.innerWidth - sidebar - 16) / 2),
    sidebarWidth,
  );
  expect(await widthOf(viewer)).toBe(halfRoom);
  expect(Math.abs((await widthOf(mindArea)) - halfRoom)).toBeLessThanOrEqual(2);

  // Until it's resized, it keeps to half the room as the window changes: at 1280px, 508px.
  const resize = (width: number) =>
    first.app.evaluate(({ BrowserWindow }, size) => {
      BrowserWindow.getAllWindows()[0]?.setSize(size, 800);
    }, width);
  await resize(1000);
  await expect.poll(() => window.evaluate(() => globalThis.innerWidth)).toBe(1000);
  await expect.poll(() => widthOf(viewer)).toBe(368);
  await resize(1280);
  await expect.poll(() => window.evaluate(() => globalThis.innerWidth)).toBe(1280);
  await expect.poll(() => widthOf(viewer)).toBe(508);

  // Dragging the left edge 100px to the left widens the panel by 100px.
  const draggedWidth = (await widthOf(viewer)) + 100;
  await dragBy(window, window.getByTestId("viewer-resize"), -100);
  await expect.poll(() => widthOf(viewer)).toBe(draggedWidth);

  // In a window too narrow for that width it gives way to the Mind, which keeps its 300px;
  // the width comes back as the window grows.
  await resize(1000);
  await expect.poll(() => window.evaluate(() => globalThis.innerWidth)).toBe(1000);
  await expect.poll(() => widthOf(mindArea)).toBe(300);
  expect(await widthOf(viewer)).toBeLessThan(draggedWidth);
  await resize(1280);
  await expect.poll(() => window.evaluate(() => globalThis.innerWidth)).toBe(1280);
  await expect.poll(() => widthOf(viewer)).toBe(draggedWidth);

  // Esc closes it and the Mind area gets its full width back.
  await window.keyboard.press("Escape");
  await expect(viewer).toHaveCount(0);
  expect(await widthOf(mindArea)).toBe(fullWidth);
  await first.app.close();

  // The width is a per-device setting, so it survives a restart. The close button closes it.
  const second = await launchApp(dataDir);
  await expect(second.window.getByTestId("viewer")).toHaveCount(0);
  await openViewer(second.window);
  await expect.poll(() => widthOf(second.window.getByTestId("viewer"))).toBe(draggedWidth);
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

test('after "Don\'t allow", a connection test offers to ask again, and asks before sending anything', async () => {
  const { app, window } = await launchApp(dataDir);
  const form = window.getByTestId("chat-setup").getByTestId("provider-form");
  await window.getByTestId("chat-choice-api-key").check();
  await form.getByRole("radio", { name: "OpenAI", exact: true }).check();
  await form.getByLabel(/^API key/).fill("sk-not-a-real-key");
  const consent = window.getByTestId("consent-dialog");

  await form.getByRole("button", { name: "Test connection" }).click();
  await consent.getByTestId("consent-decline").click();
  const result = form.getByTestId("connection-test");
  await expect(result).toContainText("Nothing was sent");

  // Asking again forgets the "Don't allow": the test asks first, again.
  await result.getByTestId("connection-test-ask-again").click();
  await expect(consent).toBeVisible();
  await consent.getByTestId("consent-decline").click();
  await expect(result).toContainText("Nothing was sent");
  await app.close();
});

test("first run fits a small window; the not-ready notice and a long title keep the Mind's text edge", async () => {
  const { app, window } = await launchApp(dataDir);
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.setSize(1000, 600));
  const setup = window.getByTestId("chat-setup");
  await setup.getByTestId("chat-choice-api-key").check();
  await setup
    .getByTestId("provider-form")
    .getByRole("radio", { name: "OpenAI-compatible server" })
    .check();
  // The form makes the dialog scroll, and "Set up later" stays in view.
  await expect(setup.getByTestId("chat-setup-later")).toBeInViewport();
  await setup.getByTestId("chat-setup-later").click();

  await window.getByTestId("new-mind").click();
  const title = window.getByTestId("mind-title");
  await expect(title).toBeFocused();
  const long = "Reading notes on spring and neap tides, harbour tables and the Moon's pull";
  await window.keyboard.type(long);
  // Long, the title wraps instead of being cut off.
  await expect(title).toHaveValue(long);
  const titleBox = await title.boundingBox();
  if (!titleBox) throw new Error("The title isn't visible.");
  expect(titleBox.height).toBeGreaterThan(70);

  // The notice's box reaches into the margin; its text starts where the title's does.
  const notice = window.getByTestId("chat-readiness").locator("span").first();
  const noticeBox = await notice.boundingBox();
  if (!noticeBox) throw new Error("The notice isn't visible.");
  expect(Math.abs(noticeBox.x - titleBox.x)).toBeLessThan(1);
  await app.close();
});

test("the window opens at the size and place it was left at", async () => {
  const first = await launchApp(dataDir);
  await first.app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0]?.setBounds({ x: 120, y: 90, width: 1100, height: 760 }),
  );
  await first.app.close();

  const second = await launchApp(dataDir);
  const bounds = await second.app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0]?.getBounds(),
  );
  expect(bounds).toMatchObject({ x: 120, y: 90, width: 1100, height: 760 });
  await second.app.close();
});

test("an API key typed for one chat provider is cleared when another is chosen, or when the server changes", async () => {
  const { app, window } = await launchApp(dataDir);
  const form = window.getByTestId("chat-setup").getByTestId("provider-form");
  await window.getByTestId("chat-choice-api-key").check();
  const key = form.getByLabel(/^API key/);
  const test = form.getByRole("button", { name: "Test connection" });

  // Typed for OpenAI, the key isn't kept for Anthropic: there is nothing to test or use with it.
  await form.getByRole("radio", { name: "OpenAI", exact: true }).check();
  await key.fill("sk-typed-for-openai");
  await form.getByRole("radio", { name: "Anthropic" }).check();
  await expect(key).toHaveValue("");
  await expect(test).toBeDisabled();

  // For a server, the key stays while the address is fixed, and goes once it names another host.
  await form.getByRole("radio", { name: "OpenAI-compatible server" }).check();
  const server = form.getByLabel("Server URL");
  await server.fill("https://api.example.com/v1");
  await key.fill("sk-typed-for-example");
  await server.fill("https://api.example.com/v2");
  await server.blur();
  await expect(key).toHaveValue("sk-typed-for-example");
  await server.fill("https://other.example.net/v1");
  await server.blur();
  await expect(key).toHaveValue("");
  await app.close();
});

test("a Linked folder shows its Folders as on disk, and files added on their own are Other Documents", async () => {
  const sources = await createDataFolder();
  try {
    const library = join(await realpath(sources), "Library");
    await mkdir(join(library, "Projects", "2026"), { recursive: true });
    await writeFile(join(library, "Paper.txt"), "A paper about attention.\n");
    await writeFile(join(library, "Projects", "2026", "Plan.txt"), "The plan for the year.\n");
    const loose = join(sources, "Loose notes.txt");
    await writeFile(loose, "Notes kept for later.\n");

    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    const documents = window.getByTestId("document-list-item");
    const folders = window.getByTestId("folder-item");
    const others = window.getByTestId("other-documents");

    // A file added on its own, with no Linked folder yet: just its row, at the top level.
    await addDocuments(window, [loose]);
    const looseItem = documents.filter({ hasText: "Loose notes" });
    await expect(looseItem).toHaveAttribute("data-depth", "0");
    await expect(others).toHaveCount(0);
    // Folders come from disk: there is no making one, or moving a Document into one.
    await expect(window.getByTestId("new-folder")).toHaveCount(0);
    await expect(window.getByTestId("move-document")).toHaveCount(0);

    // "Add folder…" links a folder, picked with the system's picker (answered by the test),
    // once the User confirms in the link dialog.
    await linkFolderFromSidebar(app, window, library);
    const root = folders.filter({ hasText: "Library" });
    await expect(root).toHaveAttribute("data-depth", "0");
    await expect(root.getByTestId("folder-toggle")).toHaveAttribute("aria-expanded", "true");
    const paperItem = documents.filter({ hasText: "Paper" });
    await expect(paperItem).toHaveAttribute("data-depth", "1");
    // Its folders, nested as on disk, each level one step deeper.
    const projects = folders.filter({ hasText: "Projects" });
    await expect(projects).toHaveAttribute("data-depth", "1");
    const year = folders.filter({ hasText: "2026" });
    await expect(year).toHaveAttribute("data-depth", "2");
    const planItem = documents.filter({ hasText: "Plan" });
    await expect(planItem).toHaveAttribute("data-depth", "3");
    await expect(planItem).toHaveAttribute(
      "data-folder-id",
      `${await year.getAttribute("data-folder-id")}`,
    );
    // Folding a folder hides what's inside it, its sub-Folder's too; unfolding shows it again.
    const toggle = projects.getByTestId("folder-toggle");
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "false");
    await expect(year).toHaveCount(0);
    await expect(planItem).toHaveCount(0);
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
    await expect(planItem).toBeVisible();

    // The file added on its own is now in "Other Documents", below the Linked folder.
    await expect(others).toHaveAttribute("data-depth", "0");
    await expect(looseItem).toHaveAttribute("data-depth", "1");
    const planBox = await planItem.boundingBox();
    const othersBox = await others.boundingBox();
    expect(othersBox?.y).toBeGreaterThan(planBox?.y ?? Number.POSITIVE_INFINITY);

    // Folding the Linked folder hides everything in it; the Other Documents stay.
    await root.getByTestId("folder-toggle").click();
    await expect(documents).toHaveCount(1);
    await expect(folders).toHaveCount(1);
    await expect(looseItem).toBeVisible();
    await app.close();
  } finally {
    await removeDataFolder(sources);
  }
});
