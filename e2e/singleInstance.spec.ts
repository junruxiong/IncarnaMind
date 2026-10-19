import { type ChildProcess, spawn } from "node:child_process";
import { once } from "node:events";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { coreOf, createDataFolder, dismissChatSetup, launchApp, removeDataFolder } from "./app";

/*
 * One copy of the app to a data folder (#1352): a second start brings the first window forward,
 * hands it the files it was asked to open, and quits without making a window or touching the data;
 * a copy on another data folder is a copy of its own, and the data folder is free again once the
 * first copy has quit.
 */

let dataDir: string;
let otherDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  otherDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(otherDir);
  await removeDataFolder(sources);
});

const logOf = (folder: string) => readFile(join(folder, "logs", "incarnamind.log"), "utf8");

/** Starts the running app's own executable again, as a second shortcut or `open -n` would. */
async function startAgain(
  running: Awaited<ReturnType<typeof launchApp>>["app"],
  files: string[],
): Promise<ChildProcess> {
  const { execPath, argv, env } = await running.evaluate(() => ({
    execPath: process.execPath,
    argv: process.argv.slice(1),
    env: { ...process.env } as Record<string, string>,
  }));
  return spawn(execPath, [...argv, ...files], { env, stdio: "ignore" });
}

test("a second start on the same data folder shows the first window, hands it the file, and quits; another data folder gets a copy of its own", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const core = await coreOf(window);
  const file = join(sources, "Harbour tides.txt");
  await writeFile(file, "The harbour master publishes the tide tables every January.\n");

  // The first window is out of sight when the second start comes.
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.hide());
  const again = await startAgain(app, [file]);
  const [code] = (await Promise.race([
    once(again, "exit"),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("The second start didn't quit.")), 30_000),
    ),
  ])) as [number | null];
  expect(code).toBe(0);

  // The first copy shows its window, still the only one, and has the file as a Document.
  await expect
    .poll(() =>
      app.evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows().map((each) => each.isVisible()),
      ),
    )
    .toEqual([true]);
  expect(app.windows()).toHaveLength(1);
  await expect
    .poll(async () =>
      core.evaluate(async (api) => (await api.listDocuments()).map((each) => each.name)),
    )
    .toEqual([expect.stringContaining("Harbour tides")]);
  const log = await logOf(dataDir);
  expect(log).toContain("app.secondLaunch");
  expect(log).toContain("app.anotherCopyRunning");

  // A copy on another data folder runs beside it, and neither sees the other's data.
  const other = await launchApp(otherDir);
  await dismissChatSetup(other.window);
  const others = await (await coreOf(other.window)).evaluate(async (api) => api.listDocuments());
  expect(others).toEqual([]);
  await other.app.close();

  // Once the first copy has quit, the data folder is free: a copy starts on it, with its data.
  await app.close();
  const next = await launchApp(dataDir);
  const kept = await (await coreOf(next.window)).evaluate(async (api) => api.listDocuments());
  expect(kept.map((each) => each.name)).toEqual([expect.stringContaining("Harbour tides")]);
  await next.app.close();
});
