import { mkdir, realpath, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  acceptPlainTextKeys,
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  frontMindName,
  interceptOpenExternal,
  launchApp,
  linkFolderFromSidebar,
  nameMind,
  newMind,
  openDocumentMenu,
  openSettings,
  removeDataFolder,
  setWindowSize,
  showSourceLocations,
  urlsOpened,
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
  await newMind(first.window);

  const created = first.window.getByTestId("mind-list-item");
  await expect(created).toHaveCount(1);
  const mindId = await created.getAttribute("data-mind-id");
  expect(mindId).toBeTruthy();
  await expect(first.window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${mindId}`);
  await expect(first.window.getByTestId("mind-editor")).toBeVisible();
  await first.app.close();

  // Second run on the same data folder: the Mind is listed and opens.
  const second = await launchApp(dataDir);
  const restored = second.window.getByTestId("mind-list-item");
  await expect(restored).toHaveCount(1);
  await expect(restored).toHaveAttribute("data-mind-id", `${mindId}`);
  await restored.click();
  await expect(second.window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${mindId}`);
  await expect(second.window.getByTestId("mind-editor")).toBeVisible();
  await second.app.close();
});

test("what was typed in a Mind, and its title, are still there after reopening the app", async () => {
  const paragraphs = ["Typed before quitting.", "A second paragraph."];

  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  await newMind(first.window);
  await nameMind(first.window, "Field notes");
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
  await expect(frontMindName(second.window)).toHaveText("Field notes");
  await expect(second.window.getByTestId("mind-editor").locator("p")).toHaveText(paragraphs);
  await second.app.close();
});

test("the sidebar lists the most recently edited Mind first, and a deleted Mind leaves it", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  await dismissChatSetup(window);
  const items = window.getByTestId("mind-list-item");
  for (const title of ["Older", "Newer"]) {
    await newMind(window);
    await nameMind(window, title);
    await expect(items.first()).toHaveText(title);
  }
  await expect(items).toHaveText(["Newer", "Older"]);

  // Editing the older Mind moves it to the top.
  await items.filter({ hasText: "Older" }).click();
  await window.getByTestId("mind-editor").click();
  await window.keyboard.type("An edit");
  await expect(items).toHaveText(["Older", "Newer"]);

  // Deleting it asks nothing and removes it from the sidebar and closes its tab: the
  // other open Mind shows instead.
  const olderId = await items.filter({ hasText: "Older" }).getAttribute("data-mind-id");
  const older = window.getByTestId("mind-row").filter({ hasText: "Older" });
  await older.hover();
  await older.getByTestId("mind-menu").click();
  await older.getByTestId("mind.delete").click();
  await expect(items).toHaveText(["Newer"]);
  await expect(window.getByTestId("mind-tab").getByTestId("tab-title")).toHaveText(["Newer"]);
  await expect(window.getByTestId("mind-pane")).not.toHaveAttribute("data-mind-id", `${olderId}`);
  await first.app.close();

  // It stays deleted after a restart.
  const second = await launchApp(dataDir);
  await expect(second.window.getByTestId("mind-list-item")).toHaveText(["Newer"]);
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

    // After a restart both are still there. Deleting one (from its menu) asks nothing and removes it.
    const second = await launchApp(dataDir);
    const restored = second.window.getByTestId("document-list-item");
    await expect(restored).toHaveCount(2);
    const doomed = restored.filter({ hasText: "Reading notes" });
    await openDocumentMenu(doomed);
    await doomed.getByTestId("document.delete").click();
    await expect(restored).toHaveCount(1);
    await expect(restored).toHaveAttribute("data-status", "ready");
    await second.app.close();
  } finally {
    await removeDataFolder(sources);
  }
});

test("Continue with ChatGPT opens OpenAI's sign-in for IncarnaMind's own registration, and can be cancelled", async () => {
  const { app, window } = await launchApp(dataDir);
  await interceptOpenExternal(app);

  // Settings → Models has it too. Nothing is signed in here: the browser is stood in for.
  await openSettings(window, "models");
  const section = window.getByTestId("chatgpt-settings");
  await expect(section.getByTestId("chatgpt-account")).toHaveText("Not signed in.");
  // Where the sign-in couldn't be kept safely (Linux without a keyring), that's said up front and
  // Continue waits, so OpenAI's page never opens for a sign-in that would have nowhere to go.
  const storage = await window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.getSecretStorage(),
  );
  if (!storage.canSave) {
    await expect(section.getByTestId("plain-text-secrets")).toBeVisible();
    await expect(section.getByTestId("chatgpt-continue")).toBeDisabled();
  }
  await acceptPlainTextKeys(window, section);
  await expect(section.getByTestId("chatgpt-continue")).toBeEnabled();
  await section.getByTestId("chatgpt-continue").click();
  await expect(section.getByTestId("chatgpt-cancel")).toBeVisible();
  await expect.poll(() => urlsOpened(app)).toHaveLength(1);
  const [opened] = await urlsOpened(app);
  const authorize = new URL(opened ?? "");
  expect(`${authorize.origin}${authorize.pathname}`).toBe(
    "https://auth.openai.com/api/accounts/authorize",
  );
  expect(authorize.searchParams.get("client_id")).toBe("dynamic_agent_client");
  expect(authorize.searchParams.get("agent_name_hint")).toBe("IncarnaMind");
  expect(authorize.searchParams.get("redirect_uri")).toMatch(
    /^http:\/\/127\.0\.0\.1:\d+\/auth\/callback$/,
  );

  // Cancelling is the User's own doing: the button comes back, with nothing to explain.
  await section.getByTestId("chatgpt-cancel").click();
  await expect(section.getByTestId("chatgpt-continue")).toBeVisible();
  await expect(section.getByTestId("chatgpt-error")).toHaveCount(0);
  await app.close();
});

test("first run asks for no model; the first own Question waits while the setup leads with the ChatGPT plan and a key, and Set up later gives the words back", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  await expect(window.getByTestId("chat-setup")).toHaveCount(0);
  await newMind(window);
  await expect(window.getByTestId("mind-pane")).toBeVisible();
  await expect(window.getByTestId("chat-readiness")).toBeVisible();

  // The first own Question waits in the conversation; the composer becomes the setup card.
  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("What is in my Documents?");
  await window.keyboard.press("Enter");
  const card = window.getByTestId("model-setup");
  await expect(card).toBeVisible();
  await expect(window.getByTestId("assistant-waiting-model")).toHaveText(
    "Waiting for you to choose a model",
  );
  await expect(card).toContainText("Choose your AI");
  await expect(card.getByTestId("setup-chatgpt")).toHaveText("Use my ChatGPT plan");
  await expect(card.getByTestId("setup-key")).toHaveText("Paste an API key");
  // Choosing it is the consent (ADR-0022): one line says what goes, and Details names it all.
  await expect(card.getByTestId("setup-chatgpt-line")).toHaveText("Plus or Pro · no key needed");
  await expect(card.getByTestId("setup-chatgpt-sends").locator("p")).toHaveText(
    "Sends your questions and document text to OpenAI. · Details",
  );
  // The card says what the Mind's "needs an AI model" line would, so that line steps aside.
  await expect(window.getByTestId("chat-readiness")).toHaveCount(0);
  // The other ways are folded under the two that lead.
  await expect(card.getByTestId("way-local")).toBeHidden();
  await card.getByTestId("setup-other").click();
  await expect(card.getByTestId("way-local")).toBeVisible();

  // In Chinese the key leads.
  await window.evaluate(async () =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await expect(card.getByTestId("setup-key")).toHaveText("粘贴 API 密钥");
  await expect(card.getByTestId("setup-chatgpt-line")).toHaveText(
    "Plus 或 Pro（中国大陆不可用）· 无需密钥",
  );
  await expect(card.getByTestId("setup-chatgpt-sends").locator("p")).toHaveText(
    "会把你的问题和文档内容发送给 OpenAI。 · 详情",
  );
  await window.evaluate(async () =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "en" },
    }),
  );

  // Set up later: the words are back in the composer.
  await card.getByTestId("setup-later").click();
  await expect(card).toBeHidden();
  await expect(window.getByTestId("composer-input")).toHaveValue("What is in my Documents?");
  await expect(window.getByTestId("chat-readiness")).toBeVisible();
  await first.app.close();
});

test("Settings → Models lists the key providers, and “Already pay for a plan?” with the ChatGPT plan", async () => {
  const { app, window } = await launchApp(dataDir);
  const settings = await openSettings(window, "models");
  await expect(settings.getByTestId("chat-choice-api-key")).toBeVisible();
  await expect(
    settings.locator("label").filter({ has: window.getByTestId("chat-choice-api-key") }),
  ).toContainText("OpenAI, Anthropic, Google, DeepSeek and more");
  const plans = settings.getByTestId("plans-settings");
  await expect(plans).toContainText("Already pay for a plan?");
  await expect(plans).toContainText("ChatGPT Plus or Pro");
  await window.evaluate(async () =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await expect(plans).toContainText("已经在付费订阅？");
  await app.close();
});

test("first run fits a small window; the not-ready notice and a long first line keep the Mind's text edge", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1000, 600);

  await newMind(window);
  await enterContent(window);
  const long = "Reading notes on spring and neap tides, harbour tables and the pull of the Moon";
  await window.keyboard.type(long);
  // Long, the first line wraps instead of being cut off.
  const line = window.getByTestId("mind-editor").locator("p").first();
  await expect(line).toHaveText(long);
  const lineBox = await line.boundingBox();
  if (!lineBox) throw new Error("The line isn't visible.");
  const lineHeight = await line.evaluate((p) => Number.parseFloat(getComputedStyle(p).lineHeight));
  expect(lineBox.height).toBeGreaterThan(lineHeight * 1.5);

  // The notice's box reaches into the margin; its text starts where the Mind's does.
  const notice = window.getByTestId("chat-readiness").locator("span").first();
  const noticeBox = await notice.boundingBox();
  if (!noticeBox) throw new Error("The notice isn't visible.");
  expect(Math.abs(noticeBox.x - lineBox.x)).toBeLessThan(1);
  await app.close();
});

test("the window opens at the size and place it was left at", async () => {
  const first = await launchApp(dataDir, { keepWindow: true });
  // Not the default centred 1280 × 800, and wholly on the screen, however small it is.
  const left = await first.app.evaluate(({ BrowserWindow, screen }) => {
    const { workArea } = screen.getPrimaryDisplay();
    const bounds = {
      x: workArea.x + 40,
      y: workArea.y + 30,
      width: Math.min(1100, workArea.width - 80),
      height: Math.min(760, workArea.height - 60),
    };
    BrowserWindow.getAllWindows()[0]?.setBounds(bounds);
    return bounds;
  });
  await first.app.close();

  const second = await launchApp(dataDir, { keepWindow: true });
  const bounds = await second.app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0]?.getBounds(),
  );
  expect(bounds).toEqual(left);
  await second.app.close();
});

test("an API key typed for one chat provider is cleared when another is chosen, or when the server changes", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  // A server the User gives is among the "More providers" in Settings.
  const settings = await openSettings(window, "models");
  const form = settings.getByTestId("provider-form");
  await settings.getByTestId("chat-choice-api-key").check();
  const key = form.getByLabel(/^API key/);
  const useKey = form.getByTestId("provider-use-key");

  // Typed for OpenAI, the key isn't kept for Anthropic: there is nothing to test or use with it.
  await form
    .locator("label")
    .filter({ hasText: /^OpenAI$/ })
    .click();
  await key.fill("sk-typed-for-openai");
  await form
    .locator("label")
    .filter({ hasText: /^Anthropic$/ })
    .click();
  await expect(key).toHaveValue("");
  await expect(useKey).toBeDisabled();

  // For a server, the key stays while the address is fixed, and goes once it names another host.
  await form.getByTestId("provider-more-toggle").click();
  await form.locator("label").filter({ hasText: "OpenAI-compatible server" }).click();
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

test("a Linked folder shows its folders as on disk in its section, folded at first, and files added on their own stay in Not in a Folder", async () => {
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
    const section = window.getByTestId("linked-folders-section");
    const folders = section.getByTestId("folder-item");
    const inSection = section.getByTestId("document-list-item");

    // A file added on its own, with no Linked folder yet: its row in Not in a Folder, at the top
    // level, and no Linked folders section.
    await addDocuments(window, [loose]);
    const looseItem = window
      .getByTestId("not-in-a-folder")
      .getByTestId("document-list-item")
      .filter({ hasText: "Loose notes" });
    await expect(looseItem).toHaveAttribute("data-depth", "0");
    await expect(section).toHaveCount(0);

    // "Add folder…" links a folder, picked with the system's picker (answered by the test),
    // once the User confirms in the link dialog. Its row comes folded, its state in sight.
    await linkFolderFromSidebar(app, window, library);
    const root = folders.filter({ hasText: "Library" });
    await expect(root).toHaveAttribute("data-depth", "0");
    await expect(root.getByTestId("folder-toggle")).toHaveAttribute("aria-expanded", "false");
    await expect(inSection).toHaveCount(0);
    await showSourceLocations(window);
    await expect(root.getByTestId("folder-toggle")).toHaveAttribute("aria-expanded", "true");
    const paperItem = inSection.filter({ hasText: "Paper" });
    await expect(paperItem).toHaveAttribute("data-depth", "1");
    // Its folders, nested as on disk, each level one step deeper.
    const projects = folders.filter({ hasText: "Projects" });
    await expect(projects).toHaveAttribute("data-depth", "1");
    const year = folders.filter({ hasText: "2026" });
    await expect(year).toHaveAttribute("data-depth", "2");
    const planItem = inSection.filter({ hasText: "Plan" });
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

    // The file added on its own is in no Linked folder: it stays in Not in a Folder alone.
    await expect(inSection.filter({ hasText: "Loose notes" })).toHaveCount(0);
    await expect(looseItem).toHaveAttribute("data-depth", "0");

    // Folding the Linked folder hides everything in it.
    await root.getByTestId("folder-toggle").click();
    await expect(inSection).toHaveCount(0);
    await expect(folders).toHaveCount(1);
    await app.close();
  } finally {
    await removeDataFolder(sources);
  }
});
