import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge, EmbeddingSettings } from "../src/core/api";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
} from "./app";

type Page = typeof globalThis & { incarnamind: CoreBridge };

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

test("switching the embedding provider warns that every Document is processed again, and cancelling changes nothing", async () => {
  const path = join(sources, "Plant notes.txt");
  await writeFile(path, "Photosynthesis turns light into chemical energy.\n");
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await addDocuments(window, [path]);

  // Settings → Document search: the built-in model, on this computer.
  await window.getByRole("button", { name: "Settings" }).click();
  const section = window.getByTestId("embedding-settings");
  const current = section.getByTestId("embedding-current");
  await expect(current).toHaveText("Embedding model: Built-in (multilingual-e5-small)");
  await expect(section).toContainText("Nothing leaves this computer.");

  // Choose OpenAI, with a key; its suggested model is filled in.
  await section.getByTestId("embedding-change").click();
  const form = section.getByTestId("embedding-form");
  await form.getByLabel("OpenAI", { exact: true }).check();
  await form.getByLabel("API key").fill("sk-smoke-test-key");
  await expect(form.getByLabel("Embedding model name")).toHaveValue("text-embedding-3-small");
  await form.getByTestId("embedding-switch").click();

  // The warning: every Document is processed again, and OpenAI would receive all their text.
  const confirm = window.getByTestId("embedding-confirm");
  await expect(confirm).toBeVisible();
  await expect(confirm).toContainText("Switch to OpenAI · text-embedding-3-small?");
  await expect(confirm).toContainText(
    "Every Document (1 in all) will be processed again with the new model",
  );
  await expect(confirm.getByTestId("embedding-confirm-cloud")).toHaveText(
    "OpenAI will receive the full text of all your Documents, and every search you make.",
  );

  // Cancelling changes nothing: no consent is asked, nothing is saved, nothing is processed again.
  await confirm.getByTestId("embedding-confirm-cancel").click();
  await expect(confirm).toBeHidden();
  await expect(window.getByTestId("consent-dialog")).toBeHidden();
  await expect(current).toHaveText("Embedding model: Built-in (multilingual-e5-small)");
  const settings = await window.evaluate(
    (): Promise<EmbeddingSettings> => (globalThis as Page).incarnamind.getEmbeddingSettings(),
  );
  expect(settings).toEqual({
    provider: {
      kind: "built-in",
      baseUrl: null,
      modelId: "multilingual-e5-small",
      hasApiKey: false,
      dimensions: 384,
      service: null,
    },
    localOnly: false,
    rebuild: null,
    error: null,
  });
  await window.getByTestId("settings").getByRole("button", { name: "Done" }).click();
  await expect(window.getByTestId("document-list-item")).toHaveAttribute("data-status", "ready");
  await expect(window.getByTestId("embedding-rebuild-notice")).toHaveCount(0);
  await app.close();
});
