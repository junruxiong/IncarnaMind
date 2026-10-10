import { chmod, mkdir, rename, rm, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  documentIdOf,
  interceptOpenDialog,
  interceptSaveDialog,
  interceptShowItemInFolder,
  launchApp,
  openDocumentAt,
  pathsShown,
  removeDataFolder,
} from "./app";

/**
 * Problems show where they happen (DESIGN.md, Problems): one line in the meta
 * tone with one action, never a modal and never a toast that goes away by
 * itself. Each test makes one kind of problem happen as a person would, then
 * checks its line is in place and that its action does what it says. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots of the lines.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(target: Page | Locator, name: string): Promise<void> {
  if (SCREENSHOTS) await target.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await chmod(join(sources, "readonly"), 0o755).catch(() => undefined);
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

const bridge = (window: Page) => ({
  settings: (language: "en" | "zh-CN") =>
    window.evaluate(
      (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    ),
});

/** A model server that refuses every request with `status`, counting the requests. */
async function refusingServer() {
  const state = { status: 401, requests: 0 };
  const server: Server = createServer((request, response) => {
    request.resume();
    request.on("end", () => {
      state.requests++;
      response.writeHead(state.status, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: { message: "refused for the test" } }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { state, server, url: `http://127.0.0.1:${port}` };
}

test("a failed Answer says so under itself with one action: the key's page in Settings, or trying again; and says it in Chinese", async () => {
  const model = await refusingServer();
  try {
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await window.evaluate(async (baseUrl) => {
      await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.saveChatProvider({
        kind: "ollama",
        baseUrl,
        modelId: "test-model",
      });
    }, model.url);

    await window.getByTestId("new-mind").click();
    await window.getByTestId("mind-editor").click();
    await window.keyboard.press("ControlOrMeta+j");
    await window.keyboard.type("What is in my Documents?");
    await window.keyboard.press("Enter");

    // The key was refused: the line names the model, in place under the Answer.
    const answer = window.getByTestId("answer");
    await expect(answer).toHaveAttribute("data-status", "failed", { timeout: 15_000 });
    const error = answer.getByTestId("answer-error");
    await expect(error).toContainText("The key for test-model was refused");
    // One line, one action; a line that screen readers hear as an alert, in the meta tone.
    const line = error.getByRole("alert");
    await expect(line).toHaveCSS("color", "rgb(101, 107, 116)");
    await expect(line.getByRole("button")).toHaveCount(1);
    await expect(line.getByRole("button")).toHaveText("Check the key");
    // It sits under the Answer's text, in the Mind, with no dialog or toast about it.
    expect(
      await answer.evaluate((node) => {
        const text = node.querySelector(".answer-content")?.getBoundingClientRect().bottom ?? 0;
        const line = node.querySelector('[data-testid="answer-error"]')?.getBoundingClientRect();
        return (line?.top ?? 0) >= text - 1;
      }),
    ).toBe(true);
    await expect(window.locator("dialog[open]")).toHaveCount(0);
    expect((await line.boundingBox())?.height).toBeLessThan(24);
    await screenshot(window, "answer-key-refused-en");

    // Its action opens the Models page, where the key is.
    await error.getByTestId("answer-error-action").click();
    await expect(window.getByTestId("settings")).toHaveAttribute("data-page", "models");
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("settings")).toBeHidden();

    // The provider has a problem now: Regenerate, and the line offers to try again, which asks again.
    model.state.status = 500;
    await answer.getByTestId("answer-regenerate").click();
    await expect(error).toContainText("The provider ran into a problem", { timeout: 60_000 });
    await expect(error.getByTestId("answer-error-action")).toHaveText("Try again");
    const asked = model.state.requests;
    // By keyboard: the focus ring shows on the action, and Enter asks again.
    await error.getByTestId("answer-error-action").focus();
    await window.keyboard.press("Tab");
    await window.keyboard.press("Shift+Tab");
    await expect(error.getByTestId("answer-error-action")).toBeFocused();
    await expect(error.getByTestId("answer-error-action")).toHaveCSS("outline-style", "solid");
    await window.keyboard.press("Enter");
    await expect.poll(() => model.state.requests).toBeGreaterThan(asked);
    await expect(error).toContainText("The provider ran into a problem", { timeout: 60_000 });

    // The same line in Chinese.
    await bridge(window).settings("zh-CN");
    await expect(error).toContainText("服务商出现了问题");
    await expect(error.getByTestId("answer-error-action")).toHaveText("重试");
    await screenshot(window, "answer-provider-problem-zh");
    await app.close();
  } finally {
    model.server.close();
  }
});

test("asking without a chat model says what to set up, in the composer's row and at the top of the Mind, and its action opens Models", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await window.getByTestId("new-mind").click();

  // At the top of the Mind: one line and one action, read as a status.
  const top = window.getByTestId("chat-readiness");
  await expect(top).toContainText("Asking Questions needs a chat model");
  await expect(top.getByRole("button")).toHaveText("Set up");
  await expect(window.getByRole("status").filter({ has: top })).toHaveCount(0);
  await screenshot(window, "needs-chat-model-en");

  // In the composer's row, where it was asked; the Question waits there, not in the note.
  await window.getByTestId("mind-editor").click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("Is anyone there?");
  await window.keyboard.press("Enter");
  const composer = window.getByTestId("composer");
  const notReady = composer.getByTestId("composer-not-ready");
  await expect(notReady).toContainText("Asking Questions needs a chat model");
  await expect(notReady.getByRole("button")).toHaveCount(1);
  await expect(notReady).toHaveCSS("color", "rgb(101, 107, 116)");
  await expect(composer.getByTestId("composer-input")).toHaveValue("Is anyone there?");
  await expect(window.getByTestId("mind-editor").getByTestId("question")).toHaveCount(0);
  await expect(window.locator("dialog[open]")).toHaveCount(0);
  await screenshot(window, "needs-chat-model-composer-en");

  // By keyboard: Tab reaches the action, Enter opens Settings on Models.
  await notReady.getByRole("button").focus();
  await expect(notReady.getByRole("button")).toHaveCSS("outline-style", "solid");
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("settings")).toHaveAttribute("data-page", "models");
  await window.keyboard.press("Escape");

  // In Chinese.
  await bridge(window).settings("zh-CN");
  await expect(top).toContainText("提问需要先设置对话模型");
  await expect(top.getByRole("button")).toHaveText("去设置");
  await expect(notReady).toContainText("提问需要先设置对话模型");
  await screenshot(window, "needs-chat-model-zh");
  await app.close();
});

test("a Document whose file moved says so in the viewer with Locate file…, which brings it back; a different file does not", async () => {
  const original = join(sources, "Amendment No. 2.md");
  const text = "# Amendment No. 2\n\nThe fee is due on 31 October 2026.\n";
  await writeFile(original, text);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [original]);
  const item = window.getByTestId("document-list-item");
  const documentId = await documentIdOf(window, "Amendment No. 2");
  await item.getByTestId("open-document").click();
  await expect(window.getByTestId("viewer-text")).toContainText("The fee is due");

  // The file is moved away: opened again, the viewer says so, in place.
  await mkdir(join(sources, "Archive"));
  const moved = join(sources, "Archive", "Amendment 2 (signed).md");
  await rename(original, moved);
  await expect(item).toHaveAttribute("data-file-status", "missing", { timeout: 15_000 });
  await window.getByTestId("viewer-close").click();
  await item.getByTestId("open-document").click();
  const gone = window.getByTestId("viewer-removed");
  await expect(gone).toHaveAttribute("data-reason", "missing");
  const line = gone.getByTestId("viewer-problem");
  await expect(line).toContainText("The file for Amendment No. 2 is missing from its folder");
  await expect(line).toHaveCSS("color", "rgb(101, 107, 116)");
  await expect(line.getByRole("button")).toHaveCount(1);
  await expect(line.getByRole("button")).toHaveText("Locate file…");
  await expect(window.locator("dialog[open]")).toHaveCount(0);
  await screenshot(window, "viewer-file-moved-en");

  // A file that isn't the Document's: told so, still offering to locate; the other file is a Document of its own.
  const other = join(sources, "Something else.md");
  await writeFile(other, "# Something else\n\nNothing to do with the amendment.\n");
  await interceptOpenDialog(app, other);
  await gone.getByTestId("viewer-locate-file").focus();
  await window.keyboard.press("Enter");
  const result = gone.getByTestId("viewer-locate-result");
  await expect(result).toContainText("That isn't the file for Amendment No. 2");
  await expect(item).toHaveCount(2);
  await expect(gone.getByTestId("viewer-locate-file")).toBeVisible();
  await screenshot(window, "viewer-file-not-this-en");

  // The right file, in its new place: the Document finds it again, and the viewer shows it.
  await interceptOpenDialog(app, moved);
  await gone.getByTestId("viewer-locate-file").click();
  await expect(window.getByTestId("viewer-removed")).toHaveCount(0);
  await expect(window.getByTestId("viewer-text")).toContainText("The fee is due");
  await expect(window.locator(`[data-document-id="${documentId}"]`).first()).toHaveAttribute(
    "data-file-status",
    "available",
  );
  // The same Document, not a new one: Citations to it still open.
  await expect(item).toHaveCount(2);
  await app.close();
});

test("a Document that is gone says so in the viewer and in Chinese; one that can't be reached offers nothing to do", async () => {
  const original = join(sources, "Lease.md");
  await writeFile(original, "# Lease\n\nRent is due monthly.\n");
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [original]);
  const item = window.getByTestId("document-list-item");
  const documentId = await documentIdOf(window, "Lease");
  await bridge(window).settings("zh-CN");

  await rm(original);
  await expect(item).toHaveAttribute("data-file-status", "missing", { timeout: 15_000 });
  await item.getByTestId("open-document").click();
  const line = window.getByTestId("viewer-removed").getByTestId("viewer-problem");
  await expect(line).toContainText("Lease 的文件已不在原来的文件夹中");
  await expect(line.getByRole("button")).toHaveText("定位文件…");
  await screenshot(window, "viewer-file-moved-zh");

  // Deleted from IncarnaMind: a line and the quote it kept, with no action to take.
  await window.getByTestId("viewer-close").click();
  await openDocumentAt(window, {
    documentId: `${documentId}-deleted`,
    pageFrom: 1,
    quote: "Rent is due monthly.",
  });
  const removed = window.getByTestId("viewer-removed");
  await expect(removed).toHaveAttribute("data-reason", "deleted");
  await expect(removed.getByTestId("viewer-problem")).toContainText("已从 IncarnaMind 中删除");
  await expect(removed.getByRole("button")).toHaveCount(0);
  await expect(removed).toContainText("Rent is due monthly.");
  await window.waitForTimeout(600);
  await screenshot(window, "viewer-deleted-zh");
  await app.close();
});

test("an export that can't be saved says why under the format, and Choose another saves it elsewhere; then Saved … shows it in the file manager", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await window.getByTestId("new-mind").click();
  await window.getByTestId("mind-title").fill("Tides");
  await window.getByTestId("mind-editor").click();
  await window.keyboard.type("Spring tides come twice a month.");

  const readOnly = join(sources, "readonly");
  await mkdir(readOnly);
  await chmod(readOnly, 0o555);
  await interceptSaveDialog(app, join(readOnly, "Tides.docx"));
  await interceptShowItemInFolder(app);
  await window.getByTestId("export-mind").click();
  const dialog = window.getByTestId("export-dialog");
  await dialog.getByTestId("export-save").click();

  // The folder is read-only: the line says so, with its one action, and nothing pops up over the dialog.
  const problem = dialog.getByTestId("export-problem");
  await expect(problem).toHaveText("Couldn't save: the folder is read-only · Choose another");
  await expect(problem).toHaveAttribute("role", "alert");
  await expect(problem.getByRole("button")).toHaveCount(1);
  await expect(problem).toHaveCSS("color", "rgb(101, 107, 116)");
  await expect(window.getByRole("alert")).toHaveCount(1);
  await screenshot(dialog, "export-cant-save-en");

  // Choose another, by keyboard: asks where again, and saves there.
  const target = join(sources, "Tides.docx");
  await interceptSaveDialog(app, target);
  await dialog.getByRole("button", { name: "Cancel" }).focus();
  await window.keyboard.press("Shift+Tab");
  await expect(dialog.getByTestId("export-choose-another")).toBeFocused();
  await expect(dialog.getByTestId("export-choose-another")).toHaveCSS("outline-style", "solid");
  await window.keyboard.press("Enter");
  const done = dialog.getByTestId("export-done");
  await expect(done).toHaveText(/^Saved Tides\.docx · Show in (Finder|folder)$/);
  await expect(done).toHaveAttribute("role", "status");
  await expect(problem).toHaveCount(0);
  await screenshot(dialog, "export-saved-en");

  // Show in Finder is the line's action.
  await done.getByTestId("export-show").click();
  await expect(dialog).toBeHidden();
  expect(await pathsShown(app)).toEqual([target]);

  // The same two lines in Chinese.
  await bridge(window).settings("zh-CN");
  await chmod(readOnly, 0o555);
  await interceptSaveDialog(app, join(readOnly, "Tides.docx"));
  await window.getByTestId("export-mind").click();
  await dialog.getByTestId("export-save").click();
  await expect(problem).toHaveText("无法保存：这个文件夹是只读的 · 另选位置");
  await screenshot(dialog, "export-cant-save-zh");
  await interceptSaveDialog(app, join(sources, "Tides 2.docx"));
  await dialog.getByTestId("export-choose-another").click();
  await expect(dialog.getByTestId("export-done")).toContainText("已保存 Tides 2.docx");
  await screenshot(dialog, "export-saved-zh");
  await app.close();
});
