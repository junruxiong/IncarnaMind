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
  openDocumentAt,
  openSettings,
  removeDataFolder,
  setWindowSize,
  showSettingsPage,
  useOutputLocation,
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
async function turnOnMeetings(window: Page, name?: string): Promise<void> {
  const settings = await openSettings(window, "meetings");
  const recording = settings.getByTestId("meetings-enabled");
  await recording.hover();
  await recording.click();
  await expect(recording).toBeChecked();
  if (name) await shot(window, name);
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
  // The Transcript is saved at Stop without asking where.
  await useOutputLocation(window, sources);
  // Off until turned on: the + menu has no "Record a meeting".
  await window.getByTestId("plus-menu").click();
  await expect(window.getByRole("menuitem", { name: "Record a meeting" })).toHaveCount(0);
  await window.keyboard.press("Escape");
  await turnOnMeetings(window, "settings-meetings-en");
  if (SCREENSHOTS) {
    await window.getByTestId("plus-menu").click();
    await expect(window.getByRole("menuitem", { name: "Record a meeting" })).toBeVisible();
    await shot(window, "plus-menu-record-en");
    await window.keyboard.press("Escape");
  }

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
  // Live, its dot is red, as on the tab.
  await expect(bar.getByTestId("meeting-dot")).toHaveCSS("background-color", "rgb(180, 35, 24)");
  await expect(window.getByTestId("meeting-time")).toHaveText(/^0:0[2-9]$/);
  // The bar sits at the top of the Mind, under its title and above the notes; the composer
  // stays for asking.
  const title = await window.getByTestId("mind-title").boundingBox();
  const controls = await bar.boundingBox();
  const editor = await window.getByTestId("mind-editor").boundingBox();
  if (!title || !controls || !editor) throw new Error("The meeting bar isn't drawn.");
  expect(controls.y).toBeGreaterThanOrEqual(title.y + title.height - 1);
  expect(controls.y + controls.height).toBeLessThanOrEqual(editor.y + 1);
  await expect(window.getByTestId("composer")).not.toContainText("Recording");
  // The tab shows the recording: a red dot and the meeting's time.
  const tab = window.getByTestId("mind-tab").filter({ hasText: "Meeting" });
  await expect(tab.getByTestId("mind-tab-recording")).toHaveAttribute("data-status", "recording");
  await expect(tab.getByTestId("mind-tab-recording")).toHaveCSS(
    "background-color",
    "rgb(180, 35, 24)",
  );
  await expect(tab.getByTestId("mind-tab-time")).toHaveText(/^0:\d\d$/);
  await shot(window, "meeting-bar-en");

  // Notes written below scroll under the bar, which stays in view at the top.
  await window.getByTestId("mind-editor").click();
  for (let line = 0; line < 30; line++) await window.keyboard.press("Enter");
  await window.keyboard.type("Rebates stay as they are until January.");
  await expect(window.getByTestId("mind-title")).not.toBeInViewport();
  await expect(bar).toBeInViewport();
  await shot(window, "meeting-bar-scrolled-en");

  // While it records, Documents wait to be read; its Transcript is read at once.
  await expect(window.getByTestId("tagging-waiting")).toBeVisible({ timeout: 20_000 });
  const file = join(sources, "Agenda.md");
  await writeFile(file, "# Agenda\n\nThe rebates.\n");
  await window.evaluate(async (path) => {
    const hooks = (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks;
    if (!hooks) throw new Error("Test hooks are off.");
    await hooks.addPaths([path]);
  }, file);
  // The Transcript's tags waiting for a model is the more pressing status: reading waits behind "+1".
  const agenda = window.getByTestId("document-list-item").filter({ hasText: "Agenda" });
  await expect(agenda).toHaveAttribute("data-status", "queued");
  await window.getByTestId("status-more").click();
  await expect(window.getByTestId("status-others").getByTestId("reading-paused")).toBeVisible();
  await window.keyboard.press("Escape");

  // ⌘⇧P (Ctrl+Shift+P) is the "Pause recording" command, bound in the window's one keymap:
  // it works while the User types in the meeting's notes, and types nothing there.
  const notes = window.getByTestId("mind-editor");
  await notes.click();
  await window.keyboard.press("End");
  await window.keyboard.type(" Next item");
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(bar).toHaveAttribute("data-status", "paused");
  await expect(bar).toContainText("Paused");
  await expect(notes).toContainText("Next item");
  await expect(notes).not.toContainText(/Next itemP/i);
  await expect(bar.getByTestId("meeting-dot")).not.toHaveCSS(
    "background-color",
    "rgb(180, 35, 24)",
  );
  await expect(window.getByTestId("meeting-resume")).toHaveAccessibleName(/Resume recording/);
  await shot(window, "meeting-paused-en");
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(bar).toHaveAttribute("data-status", "recording");

  // From another Mind, the key pauses the meeting that records: its tab's dot turns grey.
  // Resuming is for the meeting in view, so a paused meeting never starts from elsewhere.
  const meetingTab = window.getByTestId("mind-tab").filter({ hasText: "Meeting" });
  await window.keyboard.press("ControlOrMeta+KeyN");
  await expect(window.getByTestId("mind-tab")).toHaveCount(2);
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  const tabDot = meetingTab.getByTestId("mind-tab-recording");
  await expect(tabDot).toHaveAttribute("data-status", "paused");
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(tabDot).toHaveAttribute("data-status", "paused");
  await meetingTab.click();
  await expect(bar).toHaveAttribute("data-status", "paused");
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(bar).toHaveAttribute("data-status", "recording");

  await bar.getByTestId("meeting-stop").click();
  await expect(bar).toBeHidden();
  const audio = window.getByTestId("meeting-audio");
  await expect(audio).toContainText(/^Recorded 0:\d\d · Audio kept on this computer until /);
  // Stopped, the Document is read.
  await expect(
    window.getByTestId("document-list-item").filter({ hasText: "Agenda" }),
  ).toHaveAttribute("data-status", "ready");
  await shot(window, "meeting-stopped-en");

  // It can't be undone, so the line asks first; Esc keeps it.
  await audio.getByTestId("meeting-audio-delete").click();
  const confirm = window.getByTestId("meeting-audio-confirm");
  await expect(confirm).toContainText("It can’t be brought back. Your notes stay.");
  await expect(confirm.getByTestId("meeting-audio-delete-confirm")).toBeFocused();
  await window.keyboard.press("Escape");
  await expect(audio).toContainText("Audio kept on this computer");
  await audio.getByTestId("meeting-audio-delete").click();
  await shot(window, "meeting-delete-audio-en");
  await window.getByTestId("meeting-audio-delete-confirm").click();
  await expect(audio).toContainText("Audio deleted");
  const [meeting] = await bridge(window);
  expect(meeting?.audio.kept).toBe(false);
  expect(existsSync(join(dataDir, "meetings", meeting?.id ?? ""))).toBe(false);

  // The notice comes once on a device: the next meeting records at once.
  await startRecording(window, await recordMeeting(window));
  await expect(window.getByTestId("meeting-notice")).toBeHidden();
});

test("Show transcript opens the live Transcript beside the meeting, Hide transcript closes it, and at Stop it is saved", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useOutputLocation(window, sources);
  await turnOnMeetings(window);
  const bar = await startRecording(window, await recordMeeting(window), { notice: true });
  const toggle = bar.getByTestId("meeting-transcript-toggle");
  await expect(toggle).toHaveText("Show transcript");

  // Lines come from the first seconds of sound: it can be shown once the first is in.
  await expect(toggle).toBeEnabled({ timeout: 20_000 });
  await toggle.hover();
  await toggle.click();
  await expect(toggle).toHaveText("Hide transcript");
  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  const transcript = window.getByTestId("viewer-transcript");
  await expect(transcript).toBeVisible();
  await expect(window.getByTestId("viewer-title")).toHaveText(
    /^Transcript · Meeting, \d{1,2} \w+$/,
  );
  await expect(window.getByTestId("transcript-live")).toHaveText("Live");
  await expect(transcript.getByTestId("transcript-line").first()).toContainText(
    "Let's start with the launch date.",
  );
  await expect(transcript.getByTestId("transcript-line").nth(1)).toBeVisible({ timeout: 20_000 });
  // A tab of its own in the group beside the meeting's Mind, which stays where it was.
  await expect(window.getByTestId("tab-group")).toHaveCount(2);
  const mindGroup = await window.getByTestId("tab-group").first().boundingBox();
  const transcriptBox = await transcript.boundingBox();
  if (!mindGroup || !transcriptBox) throw new Error("The groups aren't drawn.");
  expect(transcriptBox.x).toBeGreaterThanOrEqual(mindGroup.x + mindGroup.width - 1);
  await expect(bar).toHaveAttribute("data-status", "recording");
  await shot(window, "meeting-transcript-wide-en");
  // A narrow window shows its groups merged: the Transcript is the tab after the Mind's.
  await setWindowSize(app, window, 900, 700);
  await expect(bar).toBeVisible();
  await shot(window, "meeting-bar-narrow-en");
  const transcriptTab = window.getByTestId("document-tab").filter({ hasText: "Transcript" });
  await transcriptTab.click();
  await expect(transcript).toBeVisible();
  await shot(window, "meeting-transcript-narrow-en");
  await setWindowSize(app, window, 1440, 900);

  // Hidden, it is still a Document: searchable and citable.
  await toggle.click();
  await expect(transcript).toBeHidden();
  await expect(toggle).toHaveText("Show transcript");
  const [meeting] = await bridge(window);
  const documentId = meeting?.transcript?.documentId ?? "";
  expect(documentId).not.toBe("");
  await expect
    .poll(() =>
      window.evaluate(async () => {
        const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
        const found = await core.searchPassages("launch date", { mode: "keyword" });
        return found.map((passage) => passage.documentId);
      }),
    )
    .toContain(documentId);
  await shot(window, "meeting-bar-transcript-en");

  // In Chinese, the same controls in Chinese words.
  await window.evaluate(async () =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "zh-CN" },
    }),
  );
  await expect(toggle).toHaveText("显示转写稿");
  await shot(window, "meeting-bar-transcript-zh");
  await toggle.click();
  await expect(toggle).toHaveText("隐藏转写稿");
  await expect(window.getByTestId("transcript-live")).toHaveText("实时");
  await shot(window, "meeting-transcript-wide-zh");
  await setWindowSize(app, window, 900, 700);
  await window.getByTestId("mind-tab").filter({ hasText: "Meeting" }).click();
  await expect(bar).toBeVisible();
  await shot(window, "meeting-bar-narrow-zh");
  // Named when its first line came, the Transcript keeps its name in another language.
  await transcriptTab.click();
  await expect(transcript).toBeVisible();
  await shot(window, "meeting-transcript-narrow-zh");
  await setWindowSize(app, window, 1440, 900);

  // At Stop it is finished and saved as WebVTT in the Output location; it opens from the Mind.
  await window.evaluate(async () =>
    (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
      user: { language: "en" },
    }),
  );
  await bar.getByTestId("meeting-stop").click();
  await expect(bar).toBeHidden();
  const saved = window.getByTestId("meeting-transcript-line");
  await expect(saved).toHaveText("Transcript saved · Open transcript", { timeout: 20_000 });
  await expect(window.getByTestId("transcript-live")).toHaveCount(0);
  const [stopped] = await bridge(window);
  expect(stopped?.transcript?.savedPath).toMatch(/Transcript · Meeting, .*\.vtt$/);
  expect(existsSync(stopped?.transcript?.savedPath ?? "")).toBe(true);
  await shot(window, "meeting-transcript-saved-en");

  // Opened at a Citation, the quote is washed in the line it was said in.
  await openDocumentAt(window, {
    documentId,
    pageFrom: 1,
    pageTo: 1,
    quote: "The launch moves to the eighteenth of November.",
    citation: { check: "found", number: 1, label: "0:01" },
  });
  const washed = window.getByTestId("viewer-transcript").locator("[data-quote-highlight]");
  await expect(washed.first()).toBeVisible();
  await expect(washed.first()).toHaveText("The launch moves to the eighteenth of November.");
  await shot(window, "meeting-transcript-citation-en");
});

test("notes during the call: typed lines get their time, Mark (⌘D) and Add to note work by keyboard, and the Citation is found", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useOutputLocation(window, sources);
  await turnOnMeetings(window);
  const bar = await startRecording(window, await recordMeeting(window), { notice: true });
  const toggle = bar.getByTestId("meeting-transcript-toggle");
  await expect(toggle).toBeEnabled({ timeout: 20_000 });
  await toggle.hover();
  await toggle.click();
  const transcript = window.getByTestId("viewer-transcript");
  const lines = transcript.getByTestId("transcript-line");
  await expect(lines.first()).toBeVisible();

  // A line typed while recording takes the meeting's time, shown when it is pointed at.
  const notes = window.getByTestId("mind-editor");
  await notes.click();
  await window.keyboard.type("Launch date");
  const typed = notes.locator("p", { hasText: "Launch date" });
  await expect(typed).toHaveAttribute("data-meeting-time", /^\d+:\d{2}$/);
  const [minutes = 0, seconds = 0] = ((await typed.getAttribute("data-meeting-time")) ?? "")
    .split(":")
    .map(Number);
  const typedSeconds = minutes * 60 + seconds;
  await typed.hover();
  await shot(window, "meeting-typed-time-wide-en");

  // It falls within the line the Transcript has from that moment: after its start, before the next one.
  const starts = () =>
    lines.evaluateAll((all) => all.map((each) => Number(each.getAttribute("data-start"))));
  await expect
    .poll(async () => (await starts()).some((start) => start / 1000 > typedSeconds + 0.5), {
      timeout: 30_000,
    })
    .toBe(true);
  const all = await starts();
  const here = all.filter((start) => start / 1000 <= typedSeconds + 1).at(-1);
  const next = all.find((start) => start / 1000 > typedSeconds + 0.5);
  expect(here).toBeDefined();
  expect(next).toBeDefined();
  expect(Math.floor((here ?? 0) / 1000)).toBeLessThanOrEqual(typedSeconds);
  expect(Math.ceil((next ?? 0) / 1000)).toBeGreaterThanOrEqual(typedSeconds);

  // Mark: ⌘D leaves an empty bullet with its time, the cursor in it.
  await window.keyboard.press("ControlOrMeta+KeyD");
  const marked = notes.locator("li p[data-meeting-time]");
  await expect(marked).toHaveCount(1);
  await expect(marked).toHaveText("");
  await expect(marked).toHaveAttribute("data-meeting-time", /^\d+:\d{2}$/);
  await window.keyboard.type("Ask about the payment review");
  await expect(marked).toHaveText("Ask about the payment review");
  await expect(window.getByTestId("meeting-mark")).toBeEnabled();
  await shot(window, "meeting-mark-wide-en");

  // Add to note, by keyboard: the line takes the focus, Enter adds it with a Citation that is found.
  const second = lines.nth(1);
  await expect(second).toBeVisible({ timeout: 20_000 });
  await second.hover();
  await second.click();
  await expect(second).toBeFocused();
  await expect(second.getByTestId("transcript-add")).toBeVisible();
  await shot(window, "meeting-add-to-note-wide-en");
  const quote = (await second.locator(".transcript-text").textContent()) ?? "";
  await window.keyboard.press("ArrowUp");
  await expect(lines.first()).toBeFocused();
  await window.keyboard.press("ArrowDown");
  await expect(second).toBeFocused();
  await window.keyboard.press("Enter");
  const citation = notes.getByTestId("citation");
  await expect(citation).toHaveCount(1);
  await expect(citation).toHaveAttribute("data-check", "found");
  await expect(notes.locator("li", { hasText: quote })).toHaveCount(1);
  await expect(window.getByTestId("transcript-added")).toHaveText("Added to the note");
  // The cursor stays in the Transcript, where the User is reading.
  await expect(second).toBeFocused();

  // The same line again is shown, not added twice.
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("transcript-added")).toHaveText("Already in the note");
  await expect(citation).toHaveCount(1);

  // Undo takes it back, outside a text field.
  await window.getByTestId("undo-action").click();
  await expect(citation).toHaveCount(0);

  // A transcript line can also be added with the mouse.
  const third = lines.nth(2);
  await expect(third).toBeVisible({ timeout: 20_000 });
  await third.hover();
  await third.getByTestId("transcript-add").click();
  await expect(citation).toHaveCount(1);
  await expect(citation).toHaveAttribute("data-check", "found");

  // In a pause no time is taken, and Mark is off.
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(bar).toHaveAttribute("data-status", "paused");
  await expect(window.getByTestId("meeting-mark")).toBeDisabled();
  await typed.click();
  await expect(notes).toBeFocused();
  await window.keyboard.press("End");
  await window.keyboard.press("Enter");
  await window.keyboard.type("Typed in the pause");
  await expect(notes.locator("p", { hasText: "Typed in the pause" })).not.toHaveAttribute(
    "data-meeting-time",
  );
  const timed = await notes.locator("[data-meeting-time]").count();
  await window.keyboard.press("ControlOrMeta+KeyD");
  await expect(notes.locator("[data-meeting-time]")).toHaveCount(timed);

  // Resumed, it times lines again.
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(bar).toHaveAttribute("data-status", "recording");
  // Chinese and a narrow window hold the same controls.
  const setLanguage = (language: "en" | "zh-CN") =>
    window.evaluate(
      async (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    );
  await setLanguage("zh-CN");
  await expect(window.getByTestId("meeting-mark")).toContainText("标记");
  await typed.hover();
  await shot(window, "meeting-typed-time-wide-zh");
  const fourth = lines.nth(3);
  await expect(fourth).toBeVisible({ timeout: 20_000 });
  await fourth.hover();
  await fourth.getByTestId("transcript-add").click();
  await expect(window.getByTestId("undo-message")).toHaveText("已将转写稿中的一句加入笔记");
  await third.hover();
  await expect(third.getByTestId("transcript-add")).toContainText("加入笔记");
  // Reserved at the line's end: it covers no words, and nothing moves when it shows.
  const words = await third.locator(".transcript-text").boundingBox();
  const button = await third.getByTestId("transcript-add").boundingBox();
  if (!words || !button) throw new Error("The line isn't drawn.");
  expect(button.x).toBeGreaterThanOrEqual(words.x + words.width - 1);
  await shot(window, "meeting-add-to-note-wide-zh");
  await setWindowSize(app, window, 900, 700);
  await window
    .getByTestId("mind-tab")
    .filter({ hasText: /Meeting|会议/ })
    .click();
  await expect(bar).toBeVisible();
  await expect(window.getByTestId("meeting-mark")).toBeVisible();
  await typed.hover();
  await shot(window, "meeting-mark-narrow-zh");
  await setLanguage("en");
  await shot(window, "meeting-mark-narrow-en");
  await window.getByTestId("document-tab").filter({ hasText: "Transcript" }).click();
  await third.hover();
  await shot(window, "meeting-add-to-note-narrow-en");
  await window.getByTestId("mind-tab").filter({ hasText: "Meeting" }).click();
  await setWindowSize(app, window, 1440, 900);
  await expect(third).toBeVisible();

  // The times are kept once the meeting is stopped.
  await bar.getByTestId("meeting-stop").click();
  await expect(bar).toBeHidden();
  await expect(typed).toHaveAttribute("data-meeting-time", /^\d+:\d{2}$/);
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
  if (SCREENSHOTS) {
    await window.getByTestId("sidebar-footer").getByRole("button", { name: "设置" }).click();
    await showSettingsPage(window, "meetings");
    await shot(window, "settings-meetings-zh");
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("settings")).toBeHidden();
  }

  // At the narrowest window, the bar keeps its controls on one row.
  await setWindowSize(app, window, 900, 660);
  await expect(bar).toBeVisible();
  const box = await bar.getByTestId("meeting-controls").boundingBox();
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
