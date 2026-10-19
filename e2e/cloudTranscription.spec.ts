import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  recordMeeting,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  turnOnMeetings,
  useOutputLocation,
} from "./app";

/**
 * Cloud transcription with the User's own key (#141), with the pretend cloud service every smoke
 * test's fake transcriber has: it takes any key but one that says it's wrong, and here drops
 * the connection after a few seconds of audio. Set INCARNAMIND_SCREENSHOTS to a folder to also
 * save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function shot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const core = (window: Page) =>
  window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return {
      settings: await bridge.getSettings(),
      meetings: await bridge.listMeetings(),
      flows: await bridge.listDataFlows(),
    };
  });

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

test("with a Soniox key, the first recording asks where to transcribe; choosing Soniox records, and the bar says when the cloud dropped", async () => {
  const { app, window } = await launchApp(dataDir, {
    transcriber: { cloud: { dropAfterMs: 4000 } },
  });
  await setWindowSize(app, window, 1600, 900);
  await dismissChatSetup(window);
  await useOutputLocation(window, sources);

  // Settings › Meetings: a Soniox key of its own, checked before it is kept.
  await turnOnMeetings(window, async () => {
    const settings = window.getByTestId("settings");
    const row = settings.getByTestId("transcription-key-soniox");
    await expect(row).toContainText("About $0.12 an hour · your Soniox key · names speakers live");
    await row.getByTestId("transcription-key-add").click();
    const input = row.getByTestId("transcription-key-input");
    await expect(input).toBeFocused();
    await input.fill("soniox-wrong-key");
    await input.press("Enter");
    await expect(row.getByTestId("transcription-key-problem")).toContainText(
      "Soniox didn’t accept this key.",
    );
    await input.fill("soniox-test-key");
    await row.getByTestId("transcription-key-save").click();
    await expect(row.getByTestId("transcription-key-saved")).toHaveText("Saved");
    // Offered, not chosen: this computer stays until the User picks.
    const route = settings.getByTestId("meetings-route");
    await expect(route.locator("option:not([hidden])")).toHaveText(["On this computer", "Soniox"]);
    await expect(route).toHaveValue("");
    await expect(settings.getByTestId("meetings-route-line")).toHaveText(
      "Private and free: the audio never leaves this computer.",
    );
    await shot(window, "settings-meetings-key-en");
  });

  // The first Start asks where; Esc records nothing.
  const start = await recordMeeting(window);
  await start.getByTestId("meeting-start-record").click();
  const sheet = window.getByTestId("transcription-sheet");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByRole("heading")).toHaveText("Where should meeting audio be transcribed?");
  await expect(sheet.getByTestId("transcription-sheet-use-local")).toBeFocused();
  await expect(sheet.getByTestId("transcription-sheet-soniox")).toContainText(
    "While you record, meeting audio, other people’s voices included, goes to Soniox.",
  );
  await shot(window, "transcription-sheet-en-1600");
  await window.keyboard.press("Escape");
  await expect(sheet).toBeHidden();
  expect((await core(window)).meetings[0]?.status).toBe("ready");

  // In Chinese, at the narrowest width, it reads as a Chinese product says it.
  await start.getByTestId("meeting-start-record").click();
  await expect(sheet).toBeVisible();
  await setLanguage(window, "zh-CN");
  await expect(sheet.getByRole("heading")).toHaveText("会议音频在哪里转写？");
  await expect(sheet.getByTestId("transcription-sheet-soniox")).toContainText(
    "录制期间，会议音频（包括其他参会者的声音）会发送给 Soniox。",
  );
  await setWindowSize(app, window, 900, 720);
  const card = await sheet.getByTestId("transcription-sheet-soniox").boundingBox();
  const frame = await sheet.boundingBox();
  expect(card && frame && card.x + card.width <= frame.x + frame.width).toBe(true);
  await shot(window, "transcription-sheet-zh-900");
  await setLanguage(window, "en");
  await setWindowSize(app, window, 1600, 900);

  // Choosing Soniox is the consent, named for it, and recording starts at once.
  await sheet.getByTestId("transcription-sheet-use-soniox").click();
  await expect(sheet).toBeHidden();
  const bar = window.getByTestId("meeting-bar");
  await expect(bar).toHaveAttribute("data-status", "recording");
  const chosen = await core(window);
  expect(chosen.settings.device).toMatchObject({
    transcriptionRoute: "soniox",
    meetingNoticeAccepted: true,
  });
  expect(chosen.flows).toContainEqual(
    expect.objectContaining({
      flow: expect.objectContaining({
        id: "transcription",
        service: expect.objectContaining({ name: "Soniox" }),
      }),
      consent: "accepted",
    }),
  );

  // The pretend service drops: this computer takes over, recording goes on, and the bar says when.
  const line = bar.getByTestId("meeting-line-cloud");
  await expect(line).toContainText(
    /^Transcribing on this computer \(the cloud dropped at \d+:\d\d\)/,
    {
      timeout: 20_000,
    },
  );
  await expect(bar).toHaveAttribute("data-status", "recording");
  await shot(window, "meeting-bar-cloud-dropped-en");
  await setLanguage(window, "zh-CN");
  await expect(line).toContainText(/^正在这台电脑上转写（云服务在 \d+:\d\d 断开）/);
  await shot(window, "meeting-bar-cloud-dropped-zh");
});
