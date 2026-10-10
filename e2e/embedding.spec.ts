import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type {
  CoreBridge,
  DocumentStatus,
  EmbeddingModelStatus,
  EmbeddingSettings,
  PassageSearchResult,
} from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
  turnOnEmbeddings,
} from "./app";

/** Long enough for a few dozen Passages, so "Embedding…" stays on screen for a moment. */
const NOTES = Array.from(
  { length: 600 },
  (_, index) => `Note ${index}: photosynthesis turns light into chemical energy.`,
).join("\n");

type Recorder = typeof globalThis & { incarnamind: CoreBridge; statuses?: DocumentStatus[] };

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder(); // the User's own files live outside the data folder
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

/** Records every status the window hears about. */
async function recordStatuses(window: Page): Promise<void> {
  await window.evaluate(() => {
    const page = globalThis as Recorder;
    page.statuses = [];
    page.incarnamind.on("document.status", (document) => {
      if (page.statuses?.at(-1) !== document.status) page.statuses?.push(document.status);
    });
  });
}

test("with embeddings off, the default, an added file is ready once read: nothing is downloaded or embedded", async () => {
  const path = join(sources, "Plant notes.txt");
  await writeFile(path, NOTES);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await recordStatuses(window);

  await window.getByTestId("add-documents-input").setInputFiles([path]);
  const item = window.getByTestId("document-list-item");
  await expect(item).toHaveAttribute("data-status", "ready");

  expect(await window.evaluate(() => (globalThis as Recorder).statuses)).toEqual([
    "queued",
    "extracting",
    "ready",
  ]);
  const { settings, model, found } = await window.evaluate(async () => {
    const bridge = (globalThis as Recorder).incarnamind;
    return {
      settings: await bridge.getEmbeddingSettings(),
      model: await bridge.getEmbeddingModel(),
      found: await bridge.searchPassages("photosynthesis", { limit: 3 }),
    };
  });
  expect((settings as EmbeddingSettings).provider.kind).toBe("off");
  expect((model as EmbeddingModelStatus).state).not.toBe("downloading");
  expect((found as PassageSearchResult[]).map((result) => result.documentName)).toContain(
    "Plant notes",
  );
  await expect(window.getByTestId("embedding-model-download")).toHaveCount(0);
  // No embedding process was started.
  const utilities = await app.evaluate(({ app }) =>
    app
      .getAppMetrics()
      .filter((metric) => metric.type === "Utility")
      .map((metric) => metric.name),
  );
  expect(utilities).not.toContain("IncarnaMind embedding");
  await app.close();
});

test("turned on, an added TXT file is embedded in the utility process, then ready and found by vector search", async () => {
  const path = join(sources, "Plant notes.txt");
  await writeFile(path, NOTES);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await turnOnEmbeddings(window);
  await recordStatuses(window);

  await window.getByTestId("add-documents-input").setInputFiles([path]);
  const item = window.getByTestId("document-list-item");
  await expect(item.getByTestId("document-status")).toContainText("Embedding…");
  await expect(item).toHaveAttribute("data-status", "ready");

  expect(await window.evaluate(() => (globalThis as Recorder).statuses)).toEqual([
    "queued",
    "extracting",
    "embedding",
    "ready",
  ]);
  const found = await window.evaluate(
    (): Promise<PassageSearchResult[]> =>
      (globalThis as Recorder).incarnamind.searchPassages("chemical energy from light", {
        mode: "vector",
        limit: 3,
      }),
  );
  expect(found.map((result) => result.documentName)).toEqual([
    "Plant notes",
    "Plant notes",
    "Plant notes",
  ]);
  // The model ran in its own utility process, not in the main process.
  const utilities = await app.evaluate(({ app }) =>
    app
      .getAppMetrics()
      .filter((metric) => metric.type === "Utility")
      .map((metric) => metric.name),
  );
  expect(utilities).toContain("IncarnaMind embedding");
  await app.close();
});
