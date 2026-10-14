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

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("Send feedback opens a mail draft holding the User's text, and the app makes no network request (#123)", async () => {
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
  await expect(dialog.getByTestId("feedback-open-draft")).toBeDisabled();

  await dialog.getByTestId("feedback-text").fill("Tags are great & the search is fast.");
  await dialog.getByTestId("feedback-open-draft").click();
  await expect(dialog).toBeHidden();
  await expect.poll(() => urlsOpened(app)).toHaveLength(1);
  const [url] = await urlsOpened(app);
  const draft = new URL(url ?? "");
  expect(draft.protocol).toBe("mailto:");
  expect(draft.searchParams.get("body")).toBe("Tags are great & the search is fast.");
  expect(draft.searchParams.get("subject")).toBe("IncarnaMind feedback");
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
  await expect(dialog).toContainText("你的反馈");
  await expect(dialog.getByTestId("feedback-open-draft")).toHaveText("打开邮件草稿");
  await app.close();
});
