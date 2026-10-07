import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { CoreBridge, DocumentStatus, PassageSearchResult } from "../src/core/api";
import { createDataFolder, dismissChatSetup, launchApp, removeDataFolder } from "./app";

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

test("an added TXT file is embedded in the utility process, then ready and found by vector search", async () => {
  const path = join(sources, "Plant notes.txt");
  await writeFile(path, NOTES);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  // Record every status the window hears about.
  await window.evaluate(() => {
    const page = globalThis as Recorder;
    page.statuses = [];
    page.incarnamind.on("document.status", (document) => {
      if (page.statuses?.at(-1) !== document.status) page.statuses?.push(document.status);
    });
  });

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
