import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge, EmbeddingSettings } from "../src/core/api";
import {
  addDocuments,
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openSettings,
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
  await openSettings(window, "search");
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
  await closeSettings(window);
  await expect(window.getByTestId("document-list-item")).toHaveAttribute("data-status", "ready");
  await expect(window.getByTestId("embedding-rebuild-notice")).toHaveCount(0);
  await app.close();
});

test("in local mode, a server elsewhere and reranking are turned down as they're chosen, saying why", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await openSettings(window, "search");
  const section = window.getByTestId("embedding-settings");
  await section.getByTestId("local-only").check();

  await section.getByTestId("embedding-change").click();
  const form = section.getByTestId("embedding-form");
  await expect(form.getByLabel("OpenAI", { exact: true })).toBeDisabled();
  await form.getByLabel("OpenAI-compatible server").check();
  const server = form.getByLabel("Server URL");
  const why = form.getByTestId("embedding-local-only");
  await server.fill("https://api.deepseek.com/v1");
  await form.getByLabel("Embedding model name").fill("embed-small");
  await expect(why).toBeVisible();
  await expect(form.getByTestId("embedding-switch")).toBeDisabled();
  await expect(form.getByRole("button", { name: "Test connection" })).toBeDisabled();
  // A server on this computer is fine.
  await server.fill("http://127.0.0.1:1234/v1");
  await expect(why).toBeHidden();
  await expect(form.getByTestId("embedding-switch")).toBeEnabled();

  // Reranking: only the built-in model, on this computer, can be chosen.
  const rerank = window.getByTestId("rerank-settings");
  await rerank.getByTestId("rerank-set-up").click();
  const rerankForm = rerank.getByTestId("rerank-form");
  await expect(rerankForm.getByLabel("Cohere", { exact: true })).toBeDisabled();
  await expect(rerankForm.getByLabel("Voyage AI", { exact: true })).toBeDisabled();
  await expect(rerankForm.getByLabel("On this computer (built-in model)")).toBeChecked();
  await expect(rerankForm.getByTestId("rerank-local-only")).toBeVisible();
  await app.close();
});

test("the built-in reranking model is turned on in Settings, and stays on this computer", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await openSettings(window, "search");
  const rerank = window.getByTestId("rerank-settings");
  await expect(rerank).toContainText("Off");

  await rerank.getByTestId("rerank-set-up").click();
  const form = rerank.getByTestId("rerank-form");
  await form.getByLabel("On this computer (built-in model)").check();
  await expect(form.getByTestId("rerank-built-in-note")).toContainText(
    "Your searches and Documents stay on this computer.",
  );
  await expect(form.getByLabel("API key")).toHaveCount(0);
  await form.getByRole("button", { name: "Use for reranking" }).click();

  // The smoke tests' fake model has nothing to download: it is ready at once.
  await expect(rerank.getByTestId("rerank-current")).toHaveText(
    /^The built-in model reranks search results \(.+\)\.$/,
  );
  await expect(rerank.getByTestId("rerank-model-state")).toHaveCount(0);
  const settings = await window.evaluate(() =>
    (globalThis as Page).incarnamind.getRerankSettings(),
  );
  expect(settings).toMatchObject({ enabled: true, kind: "built-in", service: null });

  await rerank.getByTestId("rerank-remove").click();
  await expect(rerank).toContainText("Off");
  await app.close();
});
