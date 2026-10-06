import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  type ElectronApplication,
  _electron as electron,
  type Locator,
  type Page,
} from "@playwright/test";
import type { TestHooks } from "../src/shared/testHooks";

const appDir = resolve(__dirname, "..");

export interface RunningApp {
  app: ElectronApplication;
  window: Page;
}

/** Launches the built app (`out/`) on the given data folder, with test hooks on. */
export async function launchApp(dataDir: string): Promise<RunningApp> {
  const env: Record<string, string> = {};
  for (const [name, value] of Object.entries(process.env)) {
    if (value !== undefined) env[name] = value;
  }
  delete env.ELECTRON_RUN_AS_NODE;
  env.INCARNAMIND_DATA_DIR = dataDir;
  env.INCARNAMIND_TEST_HOOKS = "1";

  const app = await electron.launch({ args: [appDir], env });
  const window = await app.firstWindow();
  await window.getByTestId("new-mind").waitFor();
  return { app, window };
}

/** Opens the Document viewer through the test hook: nothing in the UI opens it yet. */
export async function openViewer(window: Page): Promise<void> {
  await window.evaluate(() => {
    const hooks = (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks;
    if (!hooks) throw new Error("Test hooks are off: launch with INCARNAMIND_TEST_HOOKS=1.");
    hooks.openViewer();
  });
}

/** The rendered width of an element, in CSS pixels. */
export async function widthOf(locator: Locator): Promise<number> {
  const box = await locator.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  return box.width;
}

/** Drags an element horizontally by `dx` CSS pixels. */
export async function dragBy(window: Page, handle: Locator, dx: number): Promise<void> {
  const box = await handle.boundingBox();
  if (!box) throw new Error("The drag handle isn't visible.");
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  await window.mouse.move(x, y);
  await window.mouse.down();
  await window.mouse.move(x + dx, y, { steps: 5 });
  await window.mouse.up();
}

/** A fresh, empty data folder. Remove it with `removeDataFolder`. */
export const createDataFolder = () => mkdtemp(join(tmpdir(), "incarnamind-smoke-"));

export const removeDataFolder = (dataDir: string) =>
  rm(dataDir, { recursive: true, force: true, maxRetries: 3 });
