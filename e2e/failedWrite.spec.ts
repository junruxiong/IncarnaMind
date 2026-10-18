import { join } from "node:path";
import { expect, test } from "@playwright/test";
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
 * Typing after a change that couldn't be saved (#1119), as a person meets it: a
 * write fails, the window says so in plain words, storage works again, typing
 * goes on, and after a restart everything typed is there, the change that
 * failed included. In English and Chinese.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** Makes storing Mind content fail, as a full disk would, until the returned function is called. */
function failWrites(): () => void {
  const db = openDatabase(join(dataDir, "incarnamind.db"));
  db.exec(
    "CREATE TRIGGER fail_writes BEFORE INSERT ON mind_updates BEGIN SELECT RAISE(ABORT, 'disk full'); END",
  );
  return () => {
    db.exec("DROP TRIGGER fail_writes");
    db.close();
  };
}

for (const [language, notice] of [
  ["en", "Couldn’t save your latest changes (disk full)."],
  ["zh-CN", "没能保存最近的改动（disk full）。"],
] as const) {
  test(`typing after a failed save is all there after a restart (${language})`, async () => {
    const first = await launchApp(dataDir);
    await setLanguage(first.window, language);
    await setWindowSize(first.app, first.window, 1440, 900);
    await newMind(first.window);
    await nameMind(first.window, "Notes");
    const editor = first.window.getByTestId("mind-editor");
    await first.window.keyboard.type("BASE");
    await expect(editor).toHaveText("BASE");

    const restore = failWrites();
    await first.window.keyboard.type(" LOST");
    const error = first.window.getByTestId("error-notice-message");
    await expect(error).toContainText(notice);
    restore();
    await first.window.keyboard.type(" AFTER");
    await expect(editor).toHaveText("BASE LOST AFTER");
    await expect(error).toHaveCount(1);
    await first.app.close();

    const second = await launchApp(dataDir);
    await second.window.getByTestId("mind-list-item").filter({ hasText: "Notes" }).click();
    await expect(second.window.getByTestId("mind-editor")).toHaveText("BASE LOST AFTER");
  });
}
