/**
 * Tagging says where it goes, and can be stopped per Folder: once a key
 * is saved, the sidebar's status names the company the Documents go to, with
 * Stop. Stop keeps that Folder's new Documents from being sent, across a
 * restart, until "Tag with AI" in the Folder's ⋯ menu turns it back on.
 * Set INCARNAMIND_SCREENSHOTS to a folder to keep screenshots of each step.
 */
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  acceptPlainTextKeys,
  closeSettings,
  createDataFolder,
  launchApp,
  motionSettled,
  openSettings,
  removeDataFolder,
  runFromPalette,
  setLanguage,
  setWindowSize,
  sidebarFolder,
} from "./app";

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

async function screenshot(window: Page, name: string): Promise<void> {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const FOLDER = "Corran Homes v. Brazos Steel";

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

async function slowFile(name: string): Promise<string> {
  const path = join(sources, `${name}.md`);
  await writeFile(path, `${name}. Read slowly: the facts of the case.\n`);
  return path;
}

async function createFolder(window: Page, name: string): Promise<string> {
  return window.evaluate(async (name) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await core.createLibraryGroup({ name, description: "The lawsuit's papers" })).id;
  }, name);
}

async function addToFolder(window: Page, paths: string[], groupId: string): Promise<void> {
  await window.evaluate(
    async ([paths, groupId]) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      await core.addDocuments(paths, { groupId });
    },
    [paths, groupId] as const,
  );
}

async function openFolder(window: Page, name: string): Promise<void> {
  const row = sidebarFolder(window, name).getByTestId("folder-row");
  if ((await row.getAttribute("aria-expanded")) !== "true") await row.click();
  await expect(row).toHaveAttribute("aria-expanded", "true");
}

const documentItem = (window: Page, name: string) =>
  window.getByTestId("document-list-item").filter({ hasText: name });

async function pasteKey(window: Page): Promise<void> {
  const settings = await openSettings(window, "models");
  await settings.getByTestId("chat-choice-api-key").check();
  const form = settings.getByTestId("provider-form");
  await form
    .locator("label")
    .filter({ hasText: /^Anthropic$/ })
    .click();
  await expect(form.getByTestId("provider-sends")).toContainText(
    "Tagging starts when you save the key.",
  );
  await acceptPlainTextKeys(window, form);
  await form.getByLabel(/^API key/).fill("sk-ant-not-a-real-key");
  await form.getByTestId("provider-use-key").click();
  await closeSettings(window);
}

test("the status names where Tagging goes; Stop keeps the Folder's new Documents from going, until Tag with AI", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1440, 900);
  const folderId = await createFolder(window, FOLDER);
  await addToFolder(
    window,
    await Promise.all(
      [
        "Supply contract",
        "Purchase order PO-2291",
        "Delivery log, March 2025",
        "Draft brief: motion for summary judgment",
      ].map(slowFile),
    ),
    folderId,
  );
  await openFolder(window, FOLDER);
  const footer = window.getByTestId("sidebar-status");
  await expect(footer.getByTestId("tagging-waiting")).toContainText("No AI model for Tags");

  await pasteKey(window);
  const status = footer.getByTestId("tagging-status");
  await expect(status.getByTestId("status-label")).toHaveText(
    /^Tagging [2-4] Documents with Anthropic: each one's name, outline and short excerpts are sent$/,
  );
  await expect(footer.getByTestId("tagging-waiting")).toHaveCount(0);
  const stop = status.getByTestId("tagging-stop");
  await expect(stop).toHaveText("Stop");
  await expect(stop).toHaveAttribute(
    "title",
    `Stop, and don't tag new Documents in “${FOLDER}” with AI until you turn it back on`,
  );
  await screenshot(window, "tagging-status-stop-en");

  await stop.click();
  await expect(window.getByTestId("undo-message")).toHaveText(`Stopped Tagging in “${FOLDER}”`);
  await expect(status).toHaveCount(0);
  await expect(documentItem(window, "Delivery log")).toHaveAttribute("data-tagging", "stopped");

  await addToFolder(window, [await slowFile("Expert report")], folderId);
  await expect(documentItem(window, "Expert report")).toHaveAttribute("data-tagging", "stopped");
  await expect(footer.getByTestId("tagging-status")).toHaveCount(0);
  await screenshot(window, "tagging-stopped-undo-en");
  await app.close();

  const again = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(again.app, again.window, 1440, 900);
  const folder = sidebarFolder(again.window, FOLDER);
  await openFolder(again.window, FOLDER);
  await expect(documentItem(again.window, "Expert report")).toHaveAttribute(
    "data-tagging",
    "stopped",
  );
  const settings = await openSettings(again.window, "models");
  await expect(settings.getByTestId("tagging-stopped")).toContainText(
    `Tagging with AI is stopped for Documents in “${FOLDER}”.`,
  );
  await expect(settings.getByTestId("tagging-restart")).toHaveText("Tag with AI");
  await settings.getByTestId("tagging-stopped").scrollIntoViewIfNeeded();
  await screenshot(again.window, "tagging-stopped-settings-en");
  await closeSettings(again.window);

  await folder.getByTestId("folder-row").hover();
  await folder.getByTestId("folder-menu").click();
  const tagWithAi = folder.getByTestId("folder.tagWithAi");
  await expect(tagWithAi).toHaveText("Tag with AI");
  await motionSettled(again.window);
  await screenshot(again.window, "tagging-stopped-folder-menu-en");
  await tagWithAi.click();
  await expect(again.window.getByTestId("undo-message")).toHaveText(
    `Tagging Documents in “${FOLDER}” with AI again`,
  );
  // The Folder has been told what is sent: from now on the short line.
  await expect(
    again.window
      .getByTestId("sidebar-status")
      .getByTestId("tagging-status")
      .getByTestId("status-label"),
  ).toHaveText(/^Tagging [1-6] Documents? with Anthropic…$/);
  await expect(documentItem(again.window, "Expert report")).not.toHaveAttribute(
    "data-tagging",
    "stopped",
  );
  await again.app.close();
});

test("in Chinese, the status names the company, and Stop names the Folder", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1100, 800);
  await setLanguage(window, "zh-CN");
  const folderId = await createFolder(window, FOLDER);
  await acceptPlainTextKeys(window);
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.saveChatProvider({
      kind: "anthropic",
      apiKey: "sk-ant-not-a-real-key",
      modelId: "claude-sonnet-5-5",
    });
  });
  await addToFolder(
    window,
    await Promise.all(
      ["Supply contract", "Purchase order", "Delivery log", "Draft brief"].map(slowFile),
    ),
    folderId,
  );
  await openFolder(window, FOLDER);
  const status = window.getByTestId("sidebar-status").getByTestId("tagging-status");
  await expect(status.getByTestId("status-label")).toHaveText(
    /^正在用 Anthropic 给 [2-4] 个文档打标签：会发送每个文档的名称、目录和简短摘录$/,
  );
  const stop = status.getByTestId("tagging-stop");
  await expect(stop).toHaveText("停止");
  await expect(stop).toHaveAttribute(
    "title",
    `停止。重新开启前，不再用 AI 给“${FOLDER}”里的新文档打标签`,
  );
  await screenshot(window, "tagging-status-stop-zh");
  await stop.click();
  await expect(window.getByTestId("undo-message")).toHaveText(`已停止给“${FOLDER}”里的文档打标签`);
  await app.close();
});

async function saveAnthropicKey(window: Page): Promise<string> {
  await acceptPlainTextKeys(window);
  return window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const provider = await core.saveChatProvider({
      kind: "anthropic",
      apiKey: "sk-ant-not-a-real-key",
      modelId: "claude-sonnet-5-5",
    });
    return provider.id;
  });
}

const stoppedItems = (window: Page) =>
  window.locator('[data-testid="document-list-item"][data-tagging="stopped"]');

test("⌘K stops Tagging, and turns it back on for Documents not in a Folder", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 800);
  await saveAnthropicKey(window);
  await window
    .getByTestId("add-documents-input")
    .setInputFiles(
      await Promise.all(["Field notes", "Site visit", "Interview", "Survey"].map(slowFile)),
    );
  const status = window.getByTestId("sidebar-status").getByTestId("tagging-status");
  await expect(status).toBeVisible();

  await runFromPalette(window, "Stop Tagging");
  await expect(window.getByTestId("undo-message")).toHaveText(
    "Stopped Tagging Documents not in a Folder",
  );
  await expect(status).toHaveCount(0);
  await expect(stoppedItems(window).first()).toBeVisible();

  await runFromPalette(window, "Tag Documents not in a Folder with AI");
  await expect(window.getByTestId("undo-message")).toHaveText(
    "Tagging Documents not in a Folder with AI again",
  );
  await expect(stoppedItems(window)).toHaveCount(0);
  await app.close();
});

test("while Organize runs, the status names the company too, with Stop", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1440, 900);
  const providerId = await saveAnthropicKey(window);
  await createFolder(window, FOLDER);
  await window.evaluate(async (providerId) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.saveLibrarySettings({
      classifier: { kind: "chat", choice: { providerId, modelId: "claude-sonnet-5-5" } },
      automatic: true,
    });
  }, providerId);
  const scan = join(sources, "Scanned receipt.pdf");
  await writeFile(scan, buildPdf([{ image: true }]));
  await window
    .getByTestId("add-documents-input")
    .setInputFiles([
      ...(await Promise.all(["Field notes", "Site visit", "Interview", "Survey"].map(slowFile))),
      scan,
    ]);
  const progress = window.getByTestId("sidebar-status").getByTestId("organization-progress");
  await expect(progress.getByTestId("status-label")).toHaveText(
    /^Organizing [2-5] Documents with Anthropic: each one's name, outline and short excerpts are sent \(page images for scans\)$/,
  );
  await screenshot(window, "organize-status-stop-en");
  await progress.getByTestId("tagging-stop").click();
  await expect(window.getByTestId("undo-message")).toHaveText(
    "Stopped Tagging Documents not in a Folder",
  );
  await expect(progress).toHaveCount(0);
  await app.close();
});
