import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenExternal,
  launchApp,
  removeDataFolder,
  setLanguage,
  urlsOpened,
} from "./app";

const { version } = JSON.parse(readFileSync(resolve(__dirname, "../package.json"), "utf8")) as {
  version: string;
};

const os =
  process.platform === "darwin" ? "mac" : process.platform === "win32" ? "windows" : "linux";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("Send feedback opens the feedback form with the app version and system, keeps the public GitHub page as a link, and the app makes no network request (#123)", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await interceptOpenExternal(app);
  const requests: string[] = [];
  window.on("request", (request) => {
    if (/^https?:/.test(request.url())) requests.push(request.url());
  });

  // The Help menu has it, and so does ⌘K.
  const item = await app.evaluate(({ Menu }) => {
    const help = Menu.getApplicationMenu()?.items.find((each) => each.role === "help");
    return help?.submenu?.items.map((each) => each.label) ?? [];
  });
  expect(item).toEqual(["Send feedback…"]);
  await window.keyboard.press("ControlOrMeta+k");
  await window.keyboard.type("Send feedback");
  await window.keyboard.press("Enter");
  const dialog = window.getByTestId("feedback-dialog");
  await expect(dialog).toBeVisible();

  await expect(dialog).toContainText(`version (${version})`);
  await expect(dialog).toContainText("GitHub, which is public");
  await dialog.getByTestId("feedback-open").click();
  await expect(dialog).toBeHidden();
  await expect
    .poll(() => urlsOpened(app))
    .toEqual([`https://tally.so/r/bZVZjo?version=${version}&os=${os}`]);
  expect(requests).toEqual([]);
  await app.close();
});

test("the feedback dialog speaks Chinese when the interface does", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setLanguage(window, "zh-CN");
  await window.keyboard.press("ControlOrMeta+k");
  await window.keyboard.type("发送反馈");
  await window.keyboard.press("Enter");
  const dialog = window.getByTestId("feedback-dialog");
  await expect(dialog).toContainText("公开");
  await expect(dialog.getByTestId("feedback-open")).toHaveText("打开表单");
  await interceptOpenExternal(app);
  await dialog.getByTestId("feedback-open").click();
  await expect
    .poll(() => urlsOpened(app))
    .toEqual([`https://tally.so/r/M1K1EX?version=${version}&os=${os}`]);
  await app.close();
});
