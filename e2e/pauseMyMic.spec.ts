import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { FakeCapture } from "../src/core/meetings/fakeCapture";
import {
  createDataFolder,
  dismissChatSetup,
  heldGlobalKeys,
  launchApp,
  panelOf,
  pointAndClick,
  pressHeldKeys,
  recordMeeting,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  startRecording,
  turnOnMeetings,
} from "./app";

/**
 * Pause my mic (#1003), with the fake capture and transcriber every smoke test gets: the
 * microphone makes a tone that the transcriber reads as "Me", the computer's sound a quieter one
 * it reads as "Them". Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function shot(page: Page | Locator, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await page.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const mac = process.platform === "darwin";
/** The shortcut by default: ⌃⌥M on macOS, Ctrl+Shift+U elsewhere. */
const MIC_KEYS = mac ? "Ctrl+Alt+M" : "Ctrl+Shift+U";
const MIC_LABEL = mac ? "⌃⌥M" : "Ctrl+Shift+U";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const capture = (app: ElectronApplication) =>
  app.evaluate(() => {
    const fake = (globalThis as { incarnamindFakeAudio?: FakeCapture }).incarnamindFakeAudio;
    if (!fake) throw new Error("The app isn't recording with the fake capture.");
    return { running: fake.running, microphoneOn: fake.microphoneOn };
  });

/** A meeting recording from both sides, its first line in. */
async function recordingMeeting(window: Page): Promise<Locator> {
  await turnOnMeetings(window);
  const bar = await startRecording(window, await recordMeeting(window), { notice: true });
  await expect(bar.getByTestId("meeting-transcript-toggle")).toBeEnabled({ timeout: 20_000 });
  return bar;
}

/** Runs a command from ⌘K by its name. */
async function fromPalette(window: Page, name: string): Promise<void> {
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type(name);
  await expect(window.getByTestId("palette-item").filter({ hasText: name }).first()).toBeVisible();
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("palette-search")).toBeHidden();
}

/**
 * How long after the next click or key the button read `pressed`, measured in the page so the
 * test's own round trips don't add to it.
 */
async function pressedAfter(
  window: Page,
  testId: string,
  pressed: boolean,
  act: () => Promise<void>,
): Promise<number> {
  // Listening before acting: the click can reach the page before a listener installed with it.
  await window.evaluate(
    ([id, expected]) => {
      (globalThis as { pressedAfter?: Promise<number> }).pressedAfter = new Promise<number>(
        (resolve) => {
          let from: number | null = null;
          const start = (event: Event) => {
            from ??= event.timeStamp;
          };
          document.addEventListener("click", start, true);
          document.addEventListener("keydown", start, true);
          const watch = new MutationObserver(() => {
            const button = document.querySelector(`[data-testid="${id}"]`);
            if (from === null || button?.getAttribute("aria-pressed") !== String(expected)) return;
            watch.disconnect();
            document.removeEventListener("click", start, true);
            document.removeEventListener("keydown", start, true);
            resolve(performance.now() - from);
          });
          watch.observe(document.body, { attributes: true, childList: true, subtree: true });
        },
      );
    },
    [testId, pressed] as const,
  );
  await act();
  return window.evaluate(
    async () => (globalThis as { pressedAfter?: Promise<number> }).pressedAfter ?? -1,
  );
}

/** "12:05" read back, in ms. */
const msOf = (clock: string) => {
  const [minutes = 0, seconds = 0] = clock.split(":").map(Number);
  return (minutes * 60 + seconds) * 1000;
};

/** Every line of the open Transcript: who said it, and where it starts, in ms. */
const linesOf = (transcript: Locator) =>
  transcript.getByTestId("transcript-line").evaluateAll((items) =>
    items.map((item) => ({
      start: Number(item.getAttribute("data-start")),
      speaker: item.querySelector('[data-testid="transcript-speaker"]')?.textContent ?? "",
    })),
  );

/** What a mark in the Transcript says, apart from the time that starts its row. */
const wordsOf = (row: Locator) => row.locator(":scope > :last-child");

/** The stretches the Transcript marks, in ms, from its "My mic paused 12:05–12:40" rows. */
async function marksOf(transcript: Locator): Promise<{ from: number; to: number }[]> {
  const texts = await transcript
    .locator('[data-testid="transcript-gap"][data-reason="micPaused"]')
    .allTextContents();
  return texts.flatMap((text) => {
    const found = /(\d+:\d\d)\s*–\s*(\d+:\d\d)/.exec(text);
    return found ? [{ from: msOf(found[1] as string), to: msOf(found[2] as string) }] : [];
  });
}

/** Every control sits inside the bar, on rows that hold their words. */
async function expectBarHolds(bar: Locator): Promise<void> {
  await expect
    .poll(() =>
      bar.evaluate((element) => {
        const frame = element.getBoundingClientRect();
        const controls = [...element.querySelectorAll(".meeting-button, .meeting-state")];
        const heights = new Set(
          [...element.querySelectorAll(".meeting-button")].map((each) =>
            Math.round(each.getBoundingClientRect().height),
          ),
        );
        return {
          inside: controls.every((each) => {
            const box = each.getBoundingClientRect();
            return box.left >= frame.left - 1 && box.right <= frame.right + 1;
          }),
          fits: element.scrollWidth <= element.clientWidth + 1,
          oneLine: heights.size === 1,
          cut: [...element.querySelectorAll(".meeting-button")].some((button) =>
            [button, ...button.querySelectorAll("span")].some(
              (each) => each.scrollWidth > each.clientWidth + 1,
            ),
          ),
        };
      }),
    )
    .toEqual({ inside: true, fits: true, oneLine: true, cut: false });
}

test("Pause my mic: the bar's button, ⌘K and the shortcut pause and resume the microphone while the other side is recorded, no Me line comes for the stretch, and the Transcript marks it", async () => {
  test.setTimeout(150_000);
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  const bar = await recordingMeeting(window);
  const mic = bar.getByTestId("meeting-mic");

  // Beside the timer, before Pause: not pressed, and it says the other side is still recorded.
  await expect(mic).toHaveAttribute("aria-pressed", "false");
  await expect(mic).toHaveAccessibleName("Pause my mic");
  await expect(mic).toHaveAttribute(
    "title",
    `Pause my mic. The other side is still recorded (${MIC_LABEL})`,
  );
  const [state, button, pause] = await Promise.all(
    [bar.locator(".meeting-state"), mic, bar.getByTestId("meeting-pause")].map((each) =>
      each.boundingBox(),
    ),
  );
  if (!state || !button || !pause) throw new Error("The bar isn't drawn.");
  expect(state.x + state.width).toBeLessThanOrEqual(button.x + 1);
  expect(button.x + button.width).toBeLessThanOrEqual(pause.x + 1);
  expect(Math.abs(state.y + state.height / 2 - (button.y + button.height / 2))).toBeLessThan(6);
  expect(await capture(app)).toMatchObject({ running: true, microphoneOn: true });
  await expect.poll(() => heldGlobalKeys(app, MIC_KEYS)).toBe(true);

  const toggle = bar.getByTestId("meeting-transcript-toggle");
  await toggle.click();
  const transcript = window.getByTestId("viewer-transcript");
  const speakers = transcript.getByTestId("transcript-speaker");
  await expect(speakers.filter({ hasText: /^Me$/ }).first()).toBeVisible({ timeout: 20_000 });
  await expect(speakers.filter({ hasText: /^Them$/ }).first()).toBeVisible({ timeout: 20_000 });
  await shot(window, "mic-before-en");

  // The button pauses it: pressed, named for what it is now, with the line in the bar and the
  // live mark in the Transcript, each with Resume.
  expect(
    await pressedAfter(window, "meeting-mic", true, () => pointAndClick(window, mic)),
  ).toBeLessThan(100);
  await expect(mic).toHaveAttribute("aria-pressed", "true");
  await expect(mic).toHaveAccessibleName("My mic paused");
  await expect(mic).toHaveAttribute("title", `Resume my mic (${MIC_LABEL})`);
  await expect.poll(async () => (await capture(app)).microphoneOn).toBe(false);
  const line = bar.getByTestId("meeting-line-microphone-paused");
  await expect(line).toHaveText(/^My mic paused since \d+:\d\d · Resume$/);
  const open = transcript.getByTestId("transcript-mic-paused");
  await expect(wordsOf(open)).toHaveText(/^My mic paused since \d\d:\d\d · Resume$/);
  const since = msOf(/(\d+:\d\d)/.exec((await line.textContent()) ?? "")?.[1] ?? "0:00");
  await expect(bar).toHaveAttribute("data-status", "recording");
  await shot(window, "mic-paused-en");

  // The other side is still recorded and transcribed; nothing of the User's comes meanwhile.
  await expect
    .poll(async () => {
      const lines = await linesOf(transcript);
      return lines.filter((each) => each.speaker === "Them" && each.start >= since).length;
    })
    .toBeGreaterThanOrEqual(2);

  // ⌘K has the same words and the shortcut, and resumes it.
  await window.keyboard.press("ControlOrMeta+k");
  await window.keyboard.type("my mic");
  await expect(
    window.getByTestId("palette-item").filter({ hasText: "Resume my mic" }),
  ).toContainText(MIC_LABEL);
  await shot(window, "mic-palette-en");
  expect(
    await pressedAfter(window, "meeting-mic", false, () => window.keyboard.press("Enter")),
  ).toBeLessThan(100);
  await expect(mic).toHaveAttribute("aria-pressed", "false");
  await expect(line).toBeHidden();
  await expect(open).toBeHidden();
  await expect.poll(async () => (await capture(app)).microphoneOn).toBe(true);
  const span = transcript.locator('[data-testid="transcript-gap"][data-reason="micPaused"]');
  await expect(wordsOf(span)).toHaveText(/^My mic paused \d\d:\d\d–\d\d:\d\d$/);
  await shot(window, "mic-after-en");

  // The shortcut, held only while recording, pauses it from another app.
  await pressHeldKeys(app, "mic");
  await expect(mic).toHaveAttribute("aria-pressed", "true");
  await expect.poll(async () => (await capture(app)).microphoneOn).toBe(false);
  await pressHeldKeys(app, "mic");
  await expect(mic).toHaveAttribute("aria-pressed", "false");
  await expect.poll(async () => (await capture(app)).microphoneOn).toBe(true);

  // The button works from the keyboard, too, and keeps the focus.
  await mic.focus();
  expect(
    await pressedAfter(window, "meeting-mic", true, () => window.keyboard.press("Enter")),
  ).toBeLessThan(100);
  await expect(mic).toBeFocused();

  // Pause recording still stops both sides, and the keys go back to other apps; the microphone
  // stays paused when recording goes on, and its button resumes it.
  await window.keyboard.press("ControlOrMeta+Shift+KeyP");
  await expect(bar).toHaveAttribute("data-status", "paused");
  await expect.poll(async () => (await capture(app)).running).toBe(false);
  await expect.poll(() => heldGlobalKeys(app, MIC_KEYS)).toBe(false);
  await expect(mic).toHaveAttribute("aria-pressed", "true");
  await bar.getByTestId("meeting-resume").click();
  await expect(bar).toHaveAttribute("data-status", "recording");
  await expect.poll(async () => capture(app)).toEqual({ running: true, microphoneOn: false });
  await expect.poll(() => heldGlobalKeys(app, MIC_KEYS)).toBe(true);
  await pointAndClick(window, mic);
  await expect(mic).toHaveAttribute("aria-pressed", "false");
  await expect.poll(async () => (await capture(app)).microphoneOn).toBe(true);
  await expect
    .poll(async () => {
      const marks = await marksOf(transcript);
      const lines = await linesOf(transcript);
      const last = marks.at(-1)?.to ?? Number.POSITIVE_INFINITY;
      return lines.filter((each) => each.speaker === "Me" && each.start >= last).length;
    })
    .toBeGreaterThanOrEqual(1);

  // Stopped, the Transcript keeps its marks, and no Me line is inside one.
  await bar.getByTestId("meeting-stop").click();
  await expect(bar).toBeHidden();
  await expect.poll(() => heldGlobalKeys(app, MIC_KEYS)).toBe(false);
  await expect(window.getByTestId("meeting-transcript-line")).toContainText("Transcript saved", {
    timeout: 20_000,
  });
  const marks = await marksOf(transcript);
  const [firstMark] = marks;
  if (!firstMark) throw new Error("The Transcript marks no stretch.");
  expect(marks.length).toBeGreaterThanOrEqual(3);
  const lines = await linesOf(transcript);
  for (const { from, to } of marks) {
    expect(
      lines.filter((each) => each.speaker === "Me" && each.start >= from && each.start < to),
    ).toEqual([]);
  }
  expect(
    lines.filter((each) => each.speaker === "Them" && each.start >= firstMark.from).length,
  ).toBeGreaterThanOrEqual(2);

  // A new meeting starts with the microphone on.
  const second = await startRecording(window, await recordMeeting(window));
  const secondMic = second.getByTestId("meeting-mic");
  await expect(secondMic).toHaveAttribute("aria-pressed", "false");
  await expect(second.getByTestId("meeting-line-microphone-paused")).toHaveCount(0);
  expect(await capture(app)).toMatchObject({ running: true, microphoneOn: true });
  await app.close();
});

test("Pause my mic in Chinese: the tooltips, the line, ⌘K and the mark in Chinese, and the bar holds at 1,100 px in English and Chinese", async () => {
  test.setTimeout(120_000);
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1100, 800);
  await dismissChatSetup(window);
  const bar = await recordingMeeting(window);
  const mic = bar.getByTestId("meeting-mic");
  await window.mouse.move(1, 1);
  await expectBarHolds(bar);
  await shot(bar, "mic-bar-1100-en");

  await setLanguage(window, "zh-CN");
  await expect(mic).toHaveAccessibleName("暂停我的麦克风");
  await expect(mic).toHaveAttribute("title", `暂停我的麦克风，对方仍会录制（${MIC_LABEL}）`);
  await expectBarHolds(bar);
  await shot(bar, "mic-bar-1100-zh");

  await pointAndClick(window, mic);
  await expect(mic).toHaveAttribute("aria-pressed", "true");
  await expect(mic).toHaveAccessibleName("我的麦克风已暂停");
  await expect(mic).toHaveAttribute("title", `恢复我的麦克风（${MIC_LABEL}）`);
  const line = bar.getByTestId("meeting-line-microphone-paused");
  await expect(line).toHaveText(/^我的麦克风自 \d+:\d\d 起已暂停 · 恢复$/);
  await window.mouse.move(1, 1);
  await expectBarHolds(bar);
  await shot(bar, "mic-bar-paused-1100-zh");
  await setLanguage(window, "en");
  await expect(line).toHaveText(/^My mic paused since \d+:\d\d · Resume$/);
  await expectBarHolds(bar);
  await shot(bar, "mic-bar-paused-1100-en");
  await setLanguage(window, "zh-CN");

  // The live mark in the Transcript, in Chinese, beside the meeting where the window has room.
  await setWindowSize(app, window, 1440, 900);
  await bar.getByTestId("meeting-transcript-toggle").click();
  const transcript = window.getByTestId("viewer-transcript");
  const open = transcript.getByTestId("transcript-mic-paused");
  await expect(wordsOf(open)).toHaveText(/^我的麦克风自 \d\d:\d\d 起已暂停 · 恢复$/);
  const since = msOf(/(\d+:\d\d)/.exec((await open.textContent()) ?? "")?.[1] ?? "0:00");
  await expect
    .poll(async () => {
      const lines = await linesOf(transcript);
      return lines.filter((each) => each.speaker === "对方" && each.start >= since).length;
    })
    .toBeGreaterThanOrEqual(1);
  await fromPalette(window, "恢复我的麦克风");
  await expect(mic).toHaveAttribute("aria-pressed", "false");
  await expect(wordsOf(transcript.getByTestId("transcript-gap"))).toHaveText(
    /^我的麦克风在 \d\d:\d\d–\d\d:\d\d 暂停$/,
  );
  await expect(transcript.getByTestId("transcript-gap")).toHaveAttribute(
    "data-reason",
    "micPaused",
  );
  await shot(window, "mic-after-zh");

  // ⌘K names it by what it does now.
  await window.keyboard.press("ControlOrMeta+k");
  await window.keyboard.type("麦克风");
  await expect(
    window.getByTestId("palette-item").filter({ hasText: "暂停我的麦克风" }),
  ).toContainText(MIC_LABEL);
  await window.keyboard.press("Escape");
  await app.close();
});

test("Pause my mic in the floating panel: the header's button and the line under it, in step with the bar, in English and Chinese", async () => {
  test.setTimeout(120_000);
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  const bar = await recordingMeeting(window);
  const panel = await panelOf(app);
  const mic = panel.getByTestId("panel-mic");
  const barMic = bar.getByTestId("meeting-mic");

  await expect(mic).toHaveAttribute("aria-pressed", "false");
  await expect(mic).toHaveAttribute(
    "title",
    `Pause my mic. The other side is still recorded (${MIC_LABEL})`,
  );
  await expect(panel.getByTestId("panel-mic-line")).toHaveCount(0);
  const [recordingState, button, template] = await Promise.all(
    [panel.locator(".panel-state"), mic, panel.getByTestId("panel-template")].map((each) =>
      each.boundingBox(),
    ),
  );
  if (!recordingState || !button || !template) throw new Error("The panel isn't drawn.");
  expect(recordingState.x + recordingState.width).toBeLessThanOrEqual(button.x + 1);
  expect(button.x + button.width).toBeLessThanOrEqual(template.x + 1);

  await pointAndClick(panel, mic);
  await expect(mic).toHaveAttribute("aria-pressed", "true");
  await expect(mic).toHaveAttribute("title", `Resume my mic (${MIC_LABEL})`);
  const line = panel.getByTestId("panel-mic-line");
  await expect(line).toHaveText(/^My mic paused since \d+:\d\d·Resume$/);
  await expect(barMic).toHaveAttribute("aria-pressed", "true");
  await expect.poll(async () => (await capture(app)).microphoneOn).toBe(false);
  await shot(panel.locator("body"), "mic-panel-paused-en");

  // The line is under the header, and the header holds its controls in Chinese too.
  await setLanguage(window, "zh-CN");
  await expect(line).toHaveText(/^我的麦克风自 \d+:\d\d 起已暂停·恢复$/);
  await expect(mic).toHaveAttribute("title", `恢复我的麦克风（${MIC_LABEL}）`);
  const header = await panel.locator(".panel-header").boundingBox();
  const lineBox = await line.boundingBox();
  const closeBox = await panel.getByTestId("panel-close").boundingBox();
  if (!header || !lineBox || !closeBox) throw new Error("The panel isn't drawn.");
  expect(lineBox.y).toBeGreaterThanOrEqual(header.y + header.height - 1);
  expect(closeBox.x + closeBox.width).toBeLessThanOrEqual(header.x + header.width + 1);
  await shot(panel.locator("body"), "mic-panel-paused-zh");

  // Resume, in the panel, resumes it in the bar and in the capture.
  await pointAndClick(panel, panel.getByTestId("panel-mic-resume"));
  await expect(line).toHaveCount(0);
  await expect(mic).toHaveAttribute("aria-pressed", "false");
  await expect(mic).toHaveAttribute("title", `暂停我的麦克风，对方仍会录制（${MIC_LABEL}）`);
  await expect(barMic).toHaveAttribute("aria-pressed", "false");
  await expect.poll(async () => (await capture(app)).microphoneOn).toBe(true);
  await app.close();
});
