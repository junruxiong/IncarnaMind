/**
 * When the AI model can't be reached, the sidebar's status says so in one line,
 * with Retry, and the Documents are tagged again once the computer is back
 * online. The scripted model can't be reached the first time it tags a
 * Document whose text says "unreachable", and meets its rate limit the first
 * time it tags one whose text says "rate-limited": Tagging then pauses, the
 * line says when it resumes, and Retry goes on at once.
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

test("Documents that couldn't reach the model: one line with Retry, and tagged again once back online", async () => {
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
    /^Can't reach the AI model\. Tagging resumes when you're online\./,
  );
  await failed.getByTestId("tagging-failed-retry").click();
  await expect(item(window, "Invoice 51")).toHaveAttribute("data-tagging", "tagged");
  await expect(failed).toHaveCount(0);

  await window.getByTestId("add-documents-input").setInputFiles([await unreachable("Invoice 52")]);
  await expect(item(window, "Invoice 52")).toHaveAttribute("data-tagging", "failed");
  await setLanguage(window, "zh-CN");
  await expect(failed.getByTestId("status-label")).toHaveText("1 个文档未能打标签");
  await window.evaluate(() => globalThis.dispatchEvent(new Event("online")));
  await expect(item(window, "Invoice 52")).toHaveAttribute("data-tagging", "tagged");
  await expect(failed).toHaveCount(0);
  await app.close();
});

test("A rate limit pauses Tagging: one line says when it resumes, and Retry goes on at once", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 800);
  await acceptPlainTextKeys(window);
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.saveChatProvider({ kind: "openai", apiKey: "sk-test", modelId: "gpt-6-luna" });
  });
  const footer = window.getByTestId("sidebar-status");
  const limited = join(sources, "Invoice 61.txt");
  await writeFile(limited, "Invoice 61. The invoice server was rate-limited.\n");
  const plain = join(sources, "Invoice 62.txt");
  await writeFile(plain, "Invoice 62. Nothing in the way.\n");

  await addDocuments(window, [limited, plain]);
  const paused = footer.getByTestId("tagging-rate-limited");
  await expect(item(window, "Invoice 61")).toHaveAttribute("data-tagging", "waiting-for-provider");
  await expect(item(window, "Invoice 62")).toHaveAttribute("data-tagging", "waiting-for-provider");
  const resumes = await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const documents = await core.listDocuments();
    return documents.find((each) => each.name.includes("Invoice 61"))?.taggingError?.resetsAt ?? "";
  });
  expect(resumes).not.toBe("");
  // The time is the pause's end, on the clock the interface is set to (a 12-hour one shows "am").
  const minutes = String(new Date(resumes).getMinutes()).padStart(2, "0");
  await expect(paused.getByTestId("status-label")).toHaveText(
    new RegExp(`^Rate limit reached · resumes at .*:${minutes}`),
  );

  await setLanguage(window, "zh-CN");
  await expect(paused.getByTestId("status-label")).toHaveText(
    new RegExp(`^已达到速率上限 · .*:${minutes}.* 继续$`),
  );

  await paused.getByTestId("tagging-rate-limited-retry").click();
  await expect(item(window, "Invoice 61")).toHaveAttribute("data-tagging", "tagged");
  await expect(item(window, "Invoice 62")).toHaveAttribute("data-tagging", "tagged");
  await expect(paused).toHaveCount(0);
  await app.close();
});
