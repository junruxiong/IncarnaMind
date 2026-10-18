import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Page, test } from "@playwright/test";
import { openDatabase } from "../src/core/storage";
import {
  createDataFolder,
  launchApp,
  nameMind,
  newMind,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * Typing after a change storage refused (#1119), as a person meets it: the
 * window says what happened and what to do while the changes wait in memory,
 * Try again stores them, Save a copy… writes the Mind elsewhere, quitting asks
 * first, and a crash is said on the next start. In English and Chinese.
 */

let dataDir: string;
let elsewhere: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  elsewhere = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(elsewhere);
});

/** Makes storing Mind content fail as a full disk does, until the returned function is called. */
function failWrites(): () => void {
  const db = openDatabase(join(dataDir, "incarnamind.db"));
  db.exec(
    "CREATE TRIGGER fail_writes BEFORE INSERT ON mind_updates BEGIN SELECT RAISE(ABORT, 'database or disk is full'); END",
  );
  return () => {
    db.exec("DROP TRIGGER fail_writes");
    db.close();
  };
}

async function start(language: "en" | "zh-CN") {
  const { app, window } = await launchApp(dataDir);
  await setLanguage(window, language);
  await setWindowSize(app, window, 1440, 900);
  await newMind(window);
  await nameMind(window, "Notes");
  const editor = window.getByTestId("mind-editor");
  await window.keyboard.type("BASE");
  await expect(editor).toHaveText("BASE");
  return { app, window, editor };
}

const notice = (window: Page) => window.getByTestId("unsaved-notice");

for (const [language, line] of [
  [
    "en",
    "Couldn’t save your recent changes: the disk is full. Keep IncarnaMind open · Try again · Save a copy…",
  ],
  ["zh-CN", "最近的改动没能保存：磁盘已满。请先不要关闭 IncarnaMind · 重试 · 另存一份…"],
] as const) {
  test(`typing after a refused save is said, kept, and saved with Try again (${language})`, async () => {
    const { app, window, editor } = await start(language);

    const restore = failWrites();
    await window.keyboard.type(" LOST");
    await expect(notice(window).getByTestId("unsaved-notice-line")).toHaveText(line);
    await expect(window.getByTestId("error-notice")).toHaveCount(0);
    restore();
    await notice(window).getByTestId("unsaved-try-again").click();
    await expect(notice(window)).toHaveCount(0);
    await editor.focus();
    await window.keyboard.press("End");
    await window.keyboard.type(" AFTER");
    await expect(editor).toHaveText("BASE LOST AFTER");
    await app.close();

    const second = await launchApp(dataDir);
    await second.window.getByTestId("mind-list-item").filter({ hasText: "Notes" }).click();
    await expect(second.window.getByTestId("mind-editor")).toHaveText("BASE LOST AFTER");
  });
}

/** Answers the system's message box with `response`, and keeps what it was asked. */
async function answerSheet(app: ElectronApplication, response: number) {
  await app.evaluate(({ dialog }, answer) => {
    const asked: unknown[] = [];
    (globalThis as { sheetsAsked?: unknown[] }).sheetsAsked = asked;
    dialog.showMessageBox = (async (...args: unknown[]) => {
      asked.push(args.at(-1));
      return { response: answer, checkboxChecked: false };
    }) as unknown as typeof dialog.showMessageBox;
  }, response);
}

test("quitting while changes wait asks first, and Keep open keeps them", async () => {
  const { app, window, editor } = await start("en");
  const restore = failWrites();
  await window.keyboard.type(" LOST");
  await expect(notice(window)).toBeVisible();
  await answerSheet(app, 2);

  await app.evaluate(({ app: electron }) => electron.quit());

  await expect
    .poll(() =>
      app.evaluate(() => (globalThis as { sheetsAsked?: unknown[] }).sheetsAsked?.length ?? 0),
    )
    .toBe(1);
  const [asked] = await app.evaluate(
    () => (globalThis as { sheetsAsked?: unknown[] }).sheetsAsked ?? [],
  );
  expect(asked).toMatchObject({
    message: expect.stringMatching(/^\d+ changes aren’t saved$/),
    buttons: ["Save a copy…", "Quit anyway", "Keep open"],
    defaultId: 2,
    cancelId: 2,
  });
  await expect(editor).toHaveText("BASE LOST");
  restore();
  await notice(window).getByTestId("unsaved-try-again").click();
  await expect(notice(window)).toHaveCount(0);
});

test("Save a copy… writes the Mind as a Word file where the User chooses", async () => {
  const { app, window } = await start("en");
  const restore = failWrites();
  await window.keyboard.type(" LOST");
  const copy = join(elsewhere, "Notes.docx");
  await app.evaluate(({ dialog }, path) => {
    dialog.showSaveDialog = (async () => ({
      canceled: false,
      filePath: path,
    })) as unknown as typeof dialog.showSaveDialog;
  }, copy);

  await notice(window).getByTestId("unsaved-save-copy").click();

  await expect
    .poll(async () => (await readFile(copy).catch(() => null))?.byteLength ?? 0)
    .not.toBe(0);
  const zip = await readFile(copy);
  expect(zip.subarray(0, 2).toString()).toBe("PK");
  restore();
  await notice(window).getByTestId("unsaved-try-again").click();
  await expect(notice(window)).toHaveCount(0);
});

test("after a crash while changes wait, the next start says how many were lost", async () => {
  const { app, window } = await start("en");
  const restore = failWrites();
  await window.keyboard.type(" LOST");
  await expect(notice(window)).toBeVisible();

  const exited = new Promise<void>((done) => app.process().once("exit", () => done()));
  app.process().kill("SIGKILL");
  await exited;
  restore();
  const second = await launchApp(dataDir);

  await expect(second.window.getByTestId("unsaved-lost")).toHaveText(
    /^\d+ changes made before IncarnaMind closed couldn’t be saved/,
  );
});
