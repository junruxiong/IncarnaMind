/**
 * When the AI model can't be reached, the sidebar's status says so in one line,
 * with Retry, also in ⌘K, and the Documents are tagged again once the computer
 * is back online. The scripted model can't be reached the first time it tags a
 * Document whose text says "unreachable".
 */
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  acceptPlainTextKeys,
  addDocuments,
  createDataFolder,
  launchApp,
  removeDataFolder,
  runFromPalette,
  setLanguage,
  setWindowSize,
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

async function unreachable(name: string): Promise<string> {
  const path = join(sources, `${name}.txt`);
  await writeFile(path, `${name}. The invoice server was unreachable.\n`);
  return path;
}

const item = (window: Page, name: string) =>
  window.getByTestId("document-list-item").filter({ hasText: name });

test("Documents that couldn't reach the model: one line with Retry, Retry in ⌘K, and tagged again once back online", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 800);
  await acceptPlainTextKeys(window);
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.saveChatProvider({ kind: "openai", apiKey: "sk-test", modelId: "gpt-6-luna" });
  });
  const footer = window.getByTestId("sidebar-status");

  await addDocuments(window, [await unreachable("Invoice 51")]);
  await expect(item(window, "Invoice 51")).toHaveAttribute("data-tagging", "failed");
  const failed = footer.getByTestId("tagging-failed");
  await expect(failed.getByTestId("status-label")).toHaveText("1 Document couldn't be tagged");
  await expect(failed).toHaveAttribute(
    "title",
    "The AI model couldn't be reached. Tagging tries again when you're back online, or now with Retry.",
  );
  await failed.getByTestId("tagging-failed-retry").click();
  await expect(item(window, "Invoice 51")).toHaveAttribute("data-tagging", "tagged");
  await expect(failed).toHaveCount(0);

  await window.getByTestId("add-documents-input").setInputFiles([await unreachable("Invoice 52")]);
  await expect(item(window, "Invoice 52")).toHaveAttribute("data-tagging", "failed");
  await runFromPalette(window, "Retry Tagging");
  await expect(item(window, "Invoice 52")).toHaveAttribute("data-tagging", "tagged");
  await expect(failed).toHaveCount(0);

  await window.getByTestId("add-documents-input").setInputFiles([await unreachable("Invoice 53")]);
  await expect(item(window, "Invoice 53")).toHaveAttribute("data-tagging", "failed");
  await setLanguage(window, "zh-CN");
  await expect(failed.getByTestId("status-label")).toHaveText("1 个文档未能打标签");
  await window.evaluate(() => globalThis.dispatchEvent(new Event("online")));
  await expect(item(window, "Invoice 53")).toHaveAttribute("data-tagging", "tagged");
  await expect(failed).toHaveCount(0);
  await app.close();
});
