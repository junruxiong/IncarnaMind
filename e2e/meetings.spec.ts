import { existsSync, readdirSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import type { FakeCapture } from "../src/core/meetings/fakeCapture";
import type { TestHooks } from "../src/shared/testHooks";
import {
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openSettings,
  removeDataFolder,
  setWindowSize,
} from "./app";

/**
 * Recording a Meeting (#135), with the fake capture every smoke test gets: a
 * tone for the microphone and for the computer's sound, never a device. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function shot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

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

const bridge = (window: Page) =>
  window.evaluate(async () =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listMeetings(),
  );

/** Settings → Meetings → Record meetings, by its switch. */
async function turnOnMeetings(window: Page): Promise<void> {
  const settings = await openSettings(window, "meetings");
  const recording = settings.getByTestId("meetings-enabled");
  await recording.hover();
  await recording.click();
  await expect(recording).toBeChecked();
  await closeSettings(window);
}

/** The + menu's "Record a meeting": a new Mind with the start block. */
async function recordMeeting(window: Page): Promise<Locator> {
  const plus = window.getByTestId("plus-menu");
  await plus.hover();
  await plus.click();
  await window.getByRole("menuitem", { name: "Record a meeting" }).click();
  const start = window.getByTestId("meeting-start");
  await expect(start).toBeVisible();
  return start;
}

/** Starts recording from the start block; the first time, through the one-time notice. */
async function startRecording(window: Page, start: Locator, { notice = false } = {}) {
  const record = start.getByTestId("meeting-start-record");
  await record.hover();
  await record.click();
  if (notice) {
    const dialog = window.getByTestId("meeting-notice");
    await expect(dialog).toBeVisible();
    await dialog.getByTestId("meeting-notice-start").click();
    await expect(dialog).toBeHidden();
  }
  const bar = window.getByTestId("meeting-bar");
  await expect(bar).toHaveAttribute("data-status", "recording");
  return bar;
}

/** The parts saved for a Meeting's source, by where they start, in ms. */
function partsOf(meetingId: string, source: "microphone" | "system"): number[] {
  const folder = join(dataDir, "meetings", meetingId);
  if (!existsSync(folder)) return [];
  return readdirSync(folder)
    .map((name) => new RegExp(`^${source}-(\\d{10})\\.wav$`).exec(name)?.[1])
    .filter((offset): offset is string => offset !== undefined)
    .map(Number)
    .sort((a, b) => a - b);
}

const fakeAudio = (app: ElectronApplication) =>
  app.evaluate(() => {
    const fake = (globalThis as { incarnamindFakeAudio?: FakeCapture }).incarnamindFakeAudio;
    if (!fake) throw new Error("The app isn't recording with the fake capture.");
    return { settingsOpened: [...fake.settingsOpened], running: fake.running };
  });

test("a meeting records from the + menu, after the one-time notice: the bar, ⌘⇧P, Stop and Delete audio now", async () => {
  const { window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  // Off until turned on: the + menu has no "Record a meeting".
  await window.getByTestId("plus-menu").click();
  await expect(window.getByRole("menuitem", { name: "Record a meeting" })).toHaveCount(0);
  await window.keyboard.press("Escape");
  await turnOnMeetings(window);

  const start = await recordMeeting(window);
  await expect(window.getByTestId("mind-title")).toHaveValue(/^Meeting, \d{1,2} [A-Z][a-z]+$/);
  await expect(window.getByTestId("end-hint")).toHaveText(
    "Write the agenda, or anything you want to cover…",
  );
  await expect(start.getByTestId("meeting-start-audio")).toHaveValue("both");
  await shot(window, "meeting-start-en");

  // The first recording on this device shows the notice once; Cancel records nothing.
  await start.getByTestId("meeting-start-record").click();
  const notice = window.getByTestId("meeting-notice");
  await expect(notice).toContainText("Tell everyone you’re recording before you start.");
  await shot(window, "meeting-notice-en");
  await notice.getByRole("button", { name: "Cancel" }).click();
  await expect(notice).toBeHidden();
  expect((await bridge(window))[0]?.status).toBe("ready");

  const bar = await startRecording(window, start, { notice: true });
  await expect(start).toBeHidden();
  await expect(bar).toContainText("Recording");
  await expect(window.getByTestId("meeting-time")).toHaveText(/^0:0[2-9]$/);
  // The bar is the composer's second row: its controls sit under the text, left of Ask.
  const input = await window.getByTestId("composer-input").boundingBox();
  const controls = await bar.boundingBox();
  const ask = await window.getByTestId("composer-ask").boundingBox();
  if (!input || !controls || !ask) throw new Error("The meeting bar isn't drawn.");
  expect(controls.y).toBeGreaterThanOrEqual(input.y + input.height - 1);
  expect(controls.x + controls.width).toBeLessThan(ask.x);
  await shot(window, "meeting-bar-en");

  // While it records, Documents wait to be read.
  const file = join(sources, "Agenda.md");
  await writeFile(file, "# Agenda\n\nThe rebates.\n");
  await window.evaluate(async (path) => {
    const hooks = (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks;
    if (!hooks) throw new Error("Test hooks are off.");
    await hooks.addPaths([path]);
  }, file);
  await expect(window.getByTestId("reading-paused")).toBeVisible();

  // ⌘⇧P (Ctrl+Shift+P) pauses and resumes, wherever the focus is.
  await window.getByTestId("mind-title").focus();
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(bar).toHaveAttribute("data-status", "paused");
  await expect(bar).toContainText("Paused");
  await expect(window.getByTestId("meeting-resume")).toHaveAccessibleName(/Resume recording/);
  await shot(window, "meeting-paused-en");
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(bar).toHaveAttribute("data-status", "recording");

  await bar.getByTestId("meeting-stop").click();
  await expect(bar).toBeHidden();
  const audio = window.getByTestId("meeting-audio");
  await expect(audio).toContainText(/^Recorded 0:\d\d · Audio kept on this computer until /);
  // Stopped, the Document is read.
  await expect(window.getByTestId("document-list-item")).toHaveAttribute("data-status", "ready");
  await shot(window, "meeting-stopped-en");

  await audio.getByTestId("meeting-audio-delete").click();
  await expect(audio).toContainText("Audio deleted");
  const [meeting] = await bridge(window);
  expect(meeting?.audio.kept).toBe(false);
  expect(existsSync(join(dataDir, "meetings", meeting?.id ?? ""))).toBe(false);

  // The notice comes once on a device: the next meeting records at once.
  await startRecording(window, await recordMeeting(window));
  await expect(window.getByTestId("meeting-notice")).toBeHidden();
});

test("with the computer's sound denied, a line says so and opens System Settings; the microphone records on", async () => {
  const { app, window } = await launchApp(dataDir, { computerSound: "denied" });
  await dismissChatSetup(window);
  await turnOnMeetings(window);
  const bar = await startRecording(window, await recordMeeting(window), { notice: true });

  const line = window.getByTestId("meeting-line-system");
  await expect(line).toContainText(
    "Only your microphone is recorded: IncarnaMind isn’t allowed to record the computer’s sound.",
  );
  await expect(bar).toHaveAttribute("data-status", "recording");
  await shot(window, "meeting-system-denied-en");
  const open = line.getByTestId("meeting-settings-system");
  await open.hover();
  await open.click();
  await expect.poll(async () => (await fakeAudio(app)).settingsOpened).toEqual(["system"]);

  // In Chinese, the same line in Chinese words.
  await window.evaluate(async () =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await expect(line).toContainText("只录到了你的麦克风：IncarnaMind 没有录制电脑声音的权限。");
  await expect(bar).toContainText("正在录制");
  await shot(window, "meeting-system-denied-zh");

  // At the narrowest window, the bar keeps its controls on one row under the text.
  await setWindowSize(app, window, 900, 660);
  await expect(bar).toBeVisible();
  const box = await bar.boundingBox();
  expect(box?.height).toBeLessThanOrEqual(28);
  await shot(window, "meeting-bar-narrow-zh");
});

test("a device change keeps recording, the new microphone's audio at the right time", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await turnOnMeetings(window);
  const bar = await startRecording(window, await recordMeeting(window), { notice: true });
  const time = window.getByTestId("meeting-time");
  await expect(time).toHaveText(/^0:0[2-9]$/);

  await app.evaluate(() =>
    (globalThis as { incarnamindFakeAudio?: FakeCapture }).incarnamindFakeAudio?.changeDevice(1500),
  );
  await expect(time).toHaveText(/^0:0[5-9]$/);
  await expect(bar).toHaveAttribute("data-status", "recording");
  await expect(window.getByTestId("meeting-line-microphone")).toHaveCount(0);

  await bar.getByTestId("meeting-stop").click();
  const [meeting] = await bridge(window);
  // Two runs: before the change, and from the new device on, after the moment with no audio.
  const parts = partsOf(meeting?.id ?? "", "microphone");
  expect(parts).toHaveLength(2);
  expect((parts[1] ?? 0) - (parts[0] ?? 0)).toBeGreaterThan(1500);
});

test("killed mid-recording, IncarnaMind recovers the audio up to the last part and marks the gap", async () => {
  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  await turnOnMeetings(first.window);
  await startRecording(first.window, await recordMeeting(first.window), { notice: true });
  const [meeting] = await bridge(first.window);
  const meetingId = meeting?.id ?? "";
  // The first 10-second part is on disk: then the app is killed, with no time to save more.
  await expect.poll(() => partsOf(meetingId, "microphone").length, { timeout: 30_000 }).toBe(1);
  first.app.process().kill("SIGKILL");
  await first.app.waitForEvent("close");

  const { window } = await launchApp(dataDir);
  const bar = window.getByTestId("meeting-bar");
  await expect(bar).toHaveAttribute("data-status", "paused");
  const interrupted = window.getByTestId("meeting-line-interrupted");
  await expect(interrupted).toContainText(
    "IncarnaMind closed while recording. Everything up to 0:10 is saved.",
  );
  await shot(window, "meeting-interrupted-en");

  await interrupted.getByTestId("meeting-line-resume").click();
  await expect(bar).toHaveAttribute("data-status", "recording");
  await expect(window.getByTestId("meeting-line-gap")).toContainText(
    / not recorded at 0:10: IncarnaMind was closed\. · Dismiss$/,
  );
  await shot(window, "meeting-gap-en");
  const [resumed] = await bridge(window);
  expect(resumed?.gaps).toEqual([
    { atMs: expect.any(Number), missedMs: expect.any(Number), reason: "interrupted" },
  ]);
  // The gap starts where the saved part ends, about 10 seconds in.
  expect(Math.floor((resumed?.gaps[0]?.atMs ?? 0) / 1000)).toBe(10);
});
