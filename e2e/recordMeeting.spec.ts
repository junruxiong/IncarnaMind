import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  frontMindName,
  launchApp,
  leaveText,
  newMind,
  openSettings,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * "Record a meeting" is offered from the first run, with nothing to turn on, and
 * after the recording notice has been read it records in one action: in the
 * Folder of the Mind in view; while a Meeting records it opens that Meeting.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const core = (window: Page) =>
  window.evaluateHandle(() => (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind);

const meetings = async (window: Page) =>
  (await core(window)).evaluate((bridge) => bridge.listMeetings());

const mindsOf = async (window: Page) =>
  (await core(window)).evaluate((bridge) => bridge.listMinds());

const readNotice = async (window: Page) =>
  (await core(window)).evaluate((bridge) =>
    bridge.updateSettings({ device: { meetingNoticeAccepted: true } }),
  );

const RECORD = { en: "Record a meeting", "zh-CN": "录制会议" } as const;

for (const language of ["en", "zh-CN"] as const) {
  test(`Record a meeting is in the + menu (under More) and ⌘K on a fresh data folder, and Settings › Meetings has no switch for it (${language})`, async () => {
    const { app, window } = await launchApp(dataDir);
    await dismissChatSetup(window);
    await setWindowSize(app, window, 1100, 720);
    await setLanguage(window, language);

    await pointAndClick(window, window.getByTestId("plus-menu"));
    await pointAndClick(window, window.getByTestId("plus-more"));
    await expect(window.getByTestId("plus-record-meeting")).toHaveText(RECORD[language]);
    await window.keyboard.press("Escape");

    await window.keyboard.press("ControlOrMeta+k");
    await expect(window.getByTestId("palette-search")).toBeFocused();
    await window.keyboard.type(RECORD[language]);
    await expect(
      window.getByTestId("palette-item").filter({ hasText: RECORD[language] }),
    ).toHaveCount(1);
    await window.keyboard.press("Escape");

    // Words people look for: "transcribe" and "call", 录音 and 转写.
    for (const word of language === "en" ? ["transcribe", "call"] : ["录音", "转写"]) {
      await window.keyboard.press("ControlOrMeta+k");
      await expect(window.getByTestId("palette-search")).toBeFocused();
      await window.keyboard.type(word);
      await expect(
        window.getByTestId("palette-item").filter({ hasText: RECORD[language] }),
      ).toHaveCount(1);
      await window.keyboard.press("Escape");
    }

    await setLanguage(window, "en");
    const settings = await openSettings(window, "meetings");
    await expect(settings.getByTestId("meetings-settings")).toContainText("What is recorded");
    await expect(settings.getByTestId("meetings-enabled")).toHaveCount(0);
    await closeSettings(window);
    await app.close();
  });
}

test("once the recording notice is read, Record records in one action, with the cursor in the notes", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await readNotice(window);

  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-more"));
  await pointAndClick(window, window.getByTestId("plus-record-meeting"));
  await expect(window.getByTestId("meeting-bar")).toHaveAttribute("data-status", "recording");
  await expect(window.getByTestId("meeting-start")).toHaveCount(0);
  expect((await meetings(window)).map((each) => each.status)).toEqual(["recording"]);
  expect(await mindsOf(window)).toHaveLength(1);

  await window.keyboard.type("First note");
  await expect(window.getByTestId("mind-editor")).toContainText("First note");
  await app.close();
});

test("the first time, Record makes a Mind with no line, and ⌘Z right after it doesn't take it back: Delete does", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);

  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-more"));
  await pointAndClick(window, window.getByTestId("plus-record-meeting"));
  await expect(window.getByTestId("meeting-start")).toBeVisible();
  expect(await mindsOf(window)).toHaveLength(1);

  await expect(window.getByTestId("undo-message")).toHaveCount(0);
  await leaveText(window);
  await window.keyboard.press("ControlOrMeta+z");
  expect(await mindsOf(window)).toHaveLength(1);
  await app.close();
});

test("once a recording has run and stopped, ⌘Z never takes its Meeting or audio back", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await readNotice(window);

  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-more"));
  await pointAndClick(window, window.getByTestId("plus-record-meeting"));
  await expect(window.getByTestId("meeting-bar")).toHaveAttribute("data-status", "recording");
  await pointAndClick(window, window.getByTestId("meeting-stop"));
  await expect
    .poll(async () => (await meetings(window))[0]?.status, { timeout: 30_000 })
    .toBe("stopped");
  const [stopped] = await meetings(window);

  await leaveText(window);
  await window.keyboard.press("ControlOrMeta+z");
  await window.keyboard.press("ControlOrMeta+z");
  // The core answers in the order it was asked: a deletion Undo had sent would show here.
  const after = await meetings(window);
  expect(after.map((each) => each.id)).toEqual([stopped?.id]);
  expect(after[0]?.status).toBe("stopped");
  expect(after[0]?.audio.kept).toBe(stopped?.audio.kept);
  expect(await mindsOf(window)).toHaveLength(1);
  await app.close();
});

test("Record from ⌘K records in the Folder of the Mind in view", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await readNotice(window);
  const folder = await (await core(window)).evaluate((bridge) =>
    bridge.createLibraryGroup({ name: "Clients", description: "Client calls." }),
  );
  await (await core(window)).evaluate(
    (bridge, folderId) => bridge.createMind({ title: "Brief", folderId }),
    folder.id,
  );
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type("Brief");
  await pointAndClick(
    window,
    window.getByTestId("palette-item").filter({ hasText: "Brief" }).first(),
  );
  await expect(frontMindName(window)).toHaveText("Brief");

  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type("Record a meeting");
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("meeting-bar")).toHaveAttribute("data-status", "recording");
  const [recording] = await meetings(window);
  const made = (await mindsOf(window)).find((mind) => mind.id === recording?.mindId);
  expect(made?.folderId).toBe(folder.id);
  await app.close();
});

test("Record while a Meeting records opens that Meeting and makes nothing", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await readNotice(window);
  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-more"));
  await pointAndClick(window, window.getByTestId("plus-record-meeting"));
  await expect(window.getByTestId("meeting-bar")).toHaveAttribute("data-status", "recording");
  const [recording] = await meetings(window);
  const recordingName = await frontMindName(window).textContent();

  await newMind(window);
  await expect(frontMindName(window)).not.toHaveText(recordingName ?? "");
  const before = (await mindsOf(window)).length;

  await pointAndClick(window, window.getByTestId("plus-menu"));
  await pointAndClick(window, window.getByTestId("plus-more"));
  await pointAndClick(window, window.getByTestId("plus-record-meeting"));
  await expect(frontMindName(window)).toHaveText(recordingName ?? "");
  await expect(window.getByTestId("meeting-bar")).toHaveAttribute("data-status", "recording");
  expect(await mindsOf(window)).toHaveLength(before);
  expect((await meetings(window)).map((each) => each.id)).toEqual([recording?.id]);
  await app.close();
});

for (const language of ["en", "zh-CN"] as const) {
  test(`on a copy that can't record audio, Record says so in one line, with the way on (${language})`, async () => {
    const { app, window } = await launchApp(dataDir, { noAudioEngine: true });
    await dismissChatSetup(window);
    await setLanguage(window, language);
    const line = {
      en: "This copy of IncarnaMind can’t record audio.",
      "zh-CN": "这个版本的 IncarnaMind 无法录制音频。",
    }[language];

    await pointAndClick(window, window.getByTestId("plus-menu"));
    await pointAndClick(window, window.getByTestId("plus-more"));
    await pointAndClick(window, window.getByTestId("plus-record-meeting"));
    const problem = window.getByTestId("cant-record");
    await expect(problem).toContainText(line);
    expect(await mindsOf(window)).toHaveLength(0);

    await pointAndClick(window, problem.getByTestId("cant-record-notes"));
    await expect(problem).toHaveCount(0);
    await expect.poll(async () => (await mindsOf(window)).length).toBe(1);
    expect(await meetings(window)).toEqual([]);
    await app.close();
  });
}
