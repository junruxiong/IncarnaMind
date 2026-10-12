import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge, Task } from "../src/core/api";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  notificationsShown,
  pointAndClick,
  recordNotifications,
  removeDataFolder,
  setWindowSize,
  startScriptedTask,
  useLocalChatModel,
} from "./app";

/*
 * Tasks (#126): a comparison table keeps filling as a Task when its Mind is
 * closed and when IncarnaMind is force-quit, and goes on at the next start;
 * a Task that was waiting for approval comes back paused, "Needs you", with
 * the ask in the tray above the composer. Set INCARNAMIND_SCREENSHOTS to a
 * folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const SUPPLIERS: Record<string, string> = {
  "Corvane Agreement.txt":
    "Corvane Agreement.\nTermination notice: 90 days.\nLiability cap: 12 months of fees.\n",
  "Halvard MSA.txt":
    "Halvard MSA.\nTermination notice: 6 months.\nLiability cap: twice the annual fees.\n",
  "Lumeric Order Form.txt":
    "Lumeric Order Form.\nTermination notice: 30 days.\nGoverning law: New York.\n",
};

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

const tasksOf = (window: Page) =>
  window.evaluate(() =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listTasks(),
  );

const mindIdOf = (window: Page, title: string) =>
  window.evaluate(
    async (wanted) =>
      (await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listMinds()).find(
        (mind) => mind.title === wanted,
      )?.id ?? "",
    title,
  );

const setLanguage = (window: Page, language: "en" | "zh-CN") =>
  window.evaluate(
    (value) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: value },
      }),
    language,
  );

/** The tray sits joined above the composer, inside the assistant, and never over it. */
async function expectAboveComposer(window: Page, tray: Locator): Promise<void> {
  const composer = window.getByTestId("composer");
  const [trayBox, composerBox] = [await tray.boundingBox(), await composer.boundingBox()];
  if (!trayBox || !composerBox) throw new Error("The tray or the composer isn't on screen.");
  expect(trayBox.y + trayBox.height).toBeLessThanOrEqual(composerBox.y + 1);
  expect(Math.abs(trayBox.x - composerBox.x)).toBeLessThan(2);
  expect(Math.abs(trayBox.width - composerBox.width)).toBeLessThan(2);
}

const cellTexts = (table: Locator) =>
  table
    .getByTestId("comparison-row")
    .evaluateAll((rows) =>
      rows.map((row) =>
        [...row.querySelectorAll('[data-testid="comparison-cell"]')].map((cell) =>
          (cell as HTMLElement).innerText.replace(/\s+/g, " ").trim(),
        ),
      ),
    );

/** Force-quits the app as a crash would: its process killed, nothing saved on the way out. */
async function forceQuit(app: ElectronApplication): Promise<void> {
  const exited = new Promise<void>((resolve) => app.process().once("exit", () => resolve()));
  app.process().kill("SIGKILL");
  await exited;
}

test("a comparison table keeps filling as a Task: closing its Mind and force-quitting don't stop it, and it goes on at the next start", async () => {
  test.setTimeout(150_000);
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1440, 860);
  await useLocalChatModel(window);
  const paths: string[] = [];
  for (const [name, text] of Object.entries(SUPPLIERS)) {
    const path = join(sources, name);
    await writeFile(path, text);
    paths.push(path);
  }
  await addDocuments(window, paths);

  await newMind(window);
  await window.keyboard.type("Shortlist");
  await window.keyboard.press("Enter");
  await window.keyboard.type("/compare");
  await expect(window.getByTestId("slash-item-compare")).toBeVisible();
  await window.keyboard.press("Enter");
  const sheet = window.getByTestId("compare-sheet");
  await expect(sheet.getByTestId("compare-sheet-question").first()).toBeFocused();
  // "slowly": the scripted model takes its time over each cell, so the fill is halfway when quit.
  await window.keyboard.type("Termination notice, slowly");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Liability cap, slowly");
  await window.keyboard.press("ControlOrMeta+Enter");

  const table = window.getByTestId("comparison-table");
  await expect(table).toBeVisible();
  // The tray above the composer shows the Task, and the cells not yet answered wait.
  const tray = window.getByTestId("task-tray");
  const row = tray.getByTestId("task-row");
  await expect(row).toContainText("Compare Documents");
  await expect(row.getByTestId("task-step")).toHaveText(/^Filling cell \d of 6$/);
  const waiting = table.getByTestId("comparison-cell").last().locator(".comparison-cell-running");
  await expect(waiting).toBeVisible();
  await expect(waiting).toHaveText("Waiting…");
  await expectAboveComposer(window, tray);
  await expect(table.getByTestId("comparison-cell").first()).toContainText("90 days", {
    timeout: 30_000,
  });
  await screenshot(window, "compare-task-en-1440");

  // Closing the Mind doesn't stop it.
  const settledBefore = (await tasksOf(window))[0]?.progress?.done ?? 0;
  await window.keyboard.press("ControlOrMeta+w");
  await expect(table).toHaveCount(0);
  await expect
    .poll(async () => (await tasksOf(window))[0]?.progress?.done ?? 0, { timeout: 30_000 })
    .toBeGreaterThan(settledBefore);
  const [halfway] = await tasksOf(window);
  expect(halfway).toMatchObject({ kind: "comparison", status: "running" });
  expect(halfway?.progress?.done).toBeLessThan(6);

  // Force-quit halfway; at the next start it goes on by itself.
  await forceQuit(app);
  const again = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(again.app, again.window, 1440, 860);
  await expect
    .poll(async () => (await tasksOf(again.window)).find((each) => each.id === halfway?.id)?.status)
    .toBe("running");
  const mindId = await mindIdOf(again.window, "Shortlist");
  await again.window.evaluate(
    (id) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.changeLayout("main", {
        type: "open",
        item: { kind: "mind", mindId: id },
        where: { in: "place" },
      }),
    mindId,
  );
  const reopened = again.window.getByTestId("comparison-table");
  await expect(reopened).toBeVisible();
  const doneRow = again.window.getByTestId("task-tray").getByTestId("task-row");
  await expect(doneRow).toHaveAttribute("data-status", "done", { timeout: 90_000 });
  await expect(doneRow).toContainText("Compare Documents done · 6 cells");
  // Every cell answered once: one Citation each where its Document answers.
  expect(await cellTexts(reopened)).toEqual([
    ["90 days 1", "12 months of fees 2"],
    ["6 months 3", "twice the annual fees 4"],
    ["30 days 5", "! not found"],
  ]);
  await screenshot(again.window, "compare-task-done-en-1440");
  // The tray's rows and asks never go into the Mind.
  const text = await again.window.locator(".mind-editor").first().innerText();
  expect(text).not.toContain("Filling cell");
  expect(text).not.toContain("Compare Documents done");
});

test("a Task waiting for approval comes back paused after a quit: “Needs you”, the ask in the tray with its buttons pinned, answered by keyboard", async () => {
  test.setTimeout(120_000);
  const folder = await createDataFolder();
  try {
    const { app, window } = await launchApp(dataDir, { scriptedTasks: true });
    await dismissChatSetup(window);
    await setWindowSize(app, window, 1440, 860);
    await recordNotifications(app);
    await newMind(window);
    await window.keyboard.type("Credit memo");
    await window.keyboard.press("ControlOrMeta+t");
    await window.keyboard.type("Notes");
    const memoTab = window.getByTestId("mind-tab").filter({ hasText: "Credit memo" });
    const notesTab = window.getByTestId("mind-tab").filter({ hasText: "Notes" });
    await expect(notesTab).toHaveAttribute("aria-selected", "true");
    const memo = await mindIdOf(window, "Credit memo");

    // Started while its Mind isn't on screen, its first ask is a system notification.
    const taskId = await startScriptedTask(app, memo, { folder, files: ["a.txt", "b.txt"] });
    await expect
      .poll(() => notificationsShown(app))
      .toEqual([{ title: "Task needs you", body: "Credit memo" }]);
    await expect(window.getByTestId("task-tray")).toHaveCount(0);

    await pointAndClick(window, memoTab);
    const tray = window.getByTestId("task-tray");
    const ask = tray.getByTestId("task-ask");
    await expect(tray.getByTestId("task-step")).toHaveText("Waiting for your approval");
    await expect(ask.getByTestId("approval-card")).toBeVisible();
    await expect(ask.getByTestId("task-ask-count")).toHaveCount(0);
    await expectAboveComposer(window, tray);
    // ⌘↵ allows, with the focus on the ask.
    await ask.getByTestId("approval-allow-once").focus();
    await window.keyboard.press("ControlOrMeta+Enter");
    await expect(ask.getByTestId("task-ask-count")).toHaveText("2 of 2");
    await app.close();

    const again = await launchApp(dataDir, { scriptedTasks: true });
    await setWindowSize(again.app, again.window, 1440, 860);
    await pointAndClick(
      again.window,
      again.window.getByTestId("mind-tab").filter({ hasText: "Credit memo" }),
    );
    const paused = again.window.getByTestId("task-tray");
    const first = paused.getByTestId("task-row").first();
    await expect(first).toHaveAttribute("data-needs-you", "true");
    await expect(first.getByTestId("task-step")).toHaveText(
      "Needs you · paused while IncarnaMind was closed",
    );
    const pausedAsk = paused.getByTestId("task-ask");
    await expect(pausedAsk.getByTestId("task-ask-count")).toHaveText("2 of 2");
    await expectAboveComposer(again.window, paused);
    // The approval's buttons stay in view at the tray's foot.
    const buttons = pausedAsk.getByTestId("approval-allow-once");
    const [trayBox, buttonBox] = [await paused.boundingBox(), await buttons.boundingBox()];
    if (!trayBox || !buttonBox) throw new Error("The tray or its buttons aren't on screen.");
    expect(buttonBox.y + buttonBox.height).toBeLessThanOrEqual(trayBox.y + trayBox.height + 1);
    await screenshot(again.window, "tasks-paused-en-1440");

    await setWindowSize(again.app, again.window, 900, 640);
    await expectAboveComposer(again.window, paused);
    await screenshot(again.window, "tasks-paused-en-900");
    await setLanguage(again.window, "zh-CN");
    await expect(first.getByTestId("task-step")).toHaveText(
      "需要你处理 · IncarnaMind 关闭时已暂停",
    );
    await expect(pausedAsk.getByTestId("task-ask-count")).toHaveText("2/2");
    await screenshot(again.window, "tasks-paused-zh-900");
    await setLanguage(again.window, "en");
    await setWindowSize(again.app, again.window, 1440, 860);

    await buttons.focus();
    await again.window.keyboard.press("ControlOrMeta+Enter");
    const done = paused.getByTestId("task-row");
    await expect(done).toHaveAttribute("data-status", "done");
    const finished: Task | undefined = (await tasksOf(again.window)).find(
      (each) => each.id === taskId,
    );
    expect(finished?.status).toBe("done");
    // Each file written once.
    expect(await readFile(join(folder, "a.txt"), "utf8")).toBe("Written by the scripted Task.\n");
    expect(await readFile(join(folder, "b.txt"), "utf8")).toBe("Written by the scripted Task.\n");
    await pointAndClick(again.window, done.getByTestId("task-dismiss"));
    await expect(again.window.getByTestId("task-tray")).toHaveCount(0);
  } finally {
    await removeDataFolder(folder);
  }
});
