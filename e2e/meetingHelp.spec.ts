import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openSettings,
  pointAndClick,
  recordMeeting,
  removeDataFolder,
  setWindowSize,
  startRecording,
  turnOnMeetings,
  useLocalChatModel,
} from "./app";

/**
 * One-click help during a call, on the scripted chat model and the fake capture and
 * transcriber every smoke test gets: the floating panel above the meeting with the buttons and
 * the latest suggestion, the buttons at the top of the assistant, the shortcuts held only while
 * a meeting records, and the level bars. Set INCARNAMIND_SCREENSHOTS to a folder to also save
 * screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function shot(page: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await mkdir(SCREENSHOTS, { recursive: true });
  await page.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const PRICE = "Enterprise plan: $45 per seat a month, single sign-on included.";
/** The other side of a sales call, as the fake transcriber hears them. */
const BUYER = [
  "Can you do thirty-five dollars a seat for Enterprise?",
  "How much is the Enterprise plan per seat, and does it include single sign-on?",
];
const mac = process.platform === "darwin";
/** The first button's shortcut by default: ⌃⌥1 on macOS, Ctrl+Shift+1 elsewhere. */
const FIRST_KEYS = mac ? "Ctrl+Alt+1" : "Ctrl+Shift+1";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const core = (window: Page) =>
  window.evaluateHandle(() => (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind);

async function panelOf(app: ElectronApplication): Promise<Page> {
  const find = () => app.windows().find((page) => page.url().includes("meetingPanel.html"));
  await expect.poll(() => find() !== undefined, { timeout: 15_000 }).toBe(true);
  const page = find();
  if (!page) throw new Error("The floating panel didn't open.");
  await page.getByTestId("meeting-panel").waitFor();
  return page;
}

const panelOpen = (app: ElectronApplication) =>
  app.windows().some((page) => !page.isClosed() && page.url().includes("meetingPanel.html"));

const held = (app: ElectronApplication, keys: string) =>
  app.evaluate(({ globalShortcut }, asked) => globalShortcut.isRegistered(asked), keys);

/** A meeting recording from both sides, its first lines in. */
async function recordingMeeting(window: Page) {
  await turnOnMeetings(window);
  const bar = await startRecording(window, await recordMeeting(window), { notice: true });
  await expect(bar.getByTestId("meeting-transcript-toggle")).toBeEnabled({ timeout: 20_000 });
  return bar;
}

async function pressIn(page: Page, name: string): Promise<Locator> {
  const button = page.getByTestId("help-button").filter({ hasText: name });
  await pointAndClick(page, button);
  const suggestion = page.getByTestId("help-suggestion").last();
  await expect(suggestion).toHaveAttribute("data-status", "done", { timeout: 20_000 });
  await expect(suggestion.getByTestId("help-heading")).toContainText(name);
  return suggestion;
}

/** One or two options, the first recommended and starred, each with what it leads to. */
async function expectOptions(suggestion: Locator): Promise<void> {
  const options = suggestion.getByTestId("help-option");
  await expect(options).toHaveCount(2);
  await expect(options.first()).toHaveAttribute("data-recommended", "true");
  await expect(options.first()).toContainText("★");
  await expect(options.nth(1)).toHaveAttribute("data-recommended", "false");
  for (const option of await options.all()) {
    await expect(option.getByTestId("help-option-say")).toHaveText(/^“.+”/);
    await expect(option.getByTestId("help-option-effect")).toHaveText(/^→ \S/);
  }
}

test("help during the call: the floating panel above the meeting gives one or two options per button in English and Chinese, never takes the focus, changes with the template, and folds to level bars", async () => {
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    transcriber: { theirLines: BUYER },
  });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  const bar = await recordingMeeting(window);

  // With no AI model yet, the assistant and the panel each say so in one line, with Set up.
  await window.getByTestId("mind-editor").click();
  await expect(window.getByTestId("meeting-set-up")).toHaveText(
    "Choose an AI model to get help during the call · Set up",
  );
  const panel = await panelOf(app);
  await pointAndClick(
    panel,
    panel.getByTestId("help-button").filter({ hasText: "Say no politely" }),
  );
  await expect(panel.getByTestId("help-set-up")).toHaveText(
    "Choose an AI model to get help during the call · Set up",
  );
  await useLocalChatModel(window);
  await expect(window.getByTestId("meeting-starter")).toHaveText([
    "Recap the last 5 minutes",
    "What should I ask next?",
    "Action items so far",
  ]);

  // The panel opened with the recording: above other windows, never focused, kept out of
  // capture where the system can.
  const panelWindow = await app.browserWindow(panel);
  const state = () =>
    panelWindow.evaluate((win) => ({
      onTop: win.isAlwaysOnTop(),
      focusable: win.isFocusable(),
      focused: win.isFocused(),
      hidden: process.platform === "linux" ? null : win.isContentProtected(),
    }));
  expect(await state()).toEqual({
    onTop: true,
    focusable: false,
    focused: false,
    hidden: process.platform === "linux" ? null : true,
  });
  await expect(panel.getByTestId("panel-hidden")).toHaveText(
    "Hidden from your screen share where your system allows",
  );
  const buttons = panel.getByTestId("help-button");
  await expect(buttons).toHaveText([
    /^What should I say\?/,
    /^Answer their question/,
    /^What do they want\?/,
    /^Sum up so far/,
    /^Say no politely/,
    /^What to ask next/,
    /^Check what was said/,
  ]);
  await expect(buttons.first().getByTestId("help-button-keys")).toHaveText(
    mac ? "⌃⌥1" : "Ctrl+Shift+1",
  );

  // A press shows the outline at once, then the options; the panel still isn't focused.
  await pointAndClick(panel, buttons.filter({ hasText: "Say no politely" }));
  await expect(panel.getByTestId("help-outline").first()).toBeVisible();
  const sayNo = panel.getByTestId("help-suggestion");
  await expect(sayNo).toHaveAttribute("data-status", "done", { timeout: 20_000 });
  await expect(sayNo.getByTestId("help-heading")).toHaveText(/^Say no politely · They said: “.+”$/);
  await expectOptions(sayNo);
  expect((await state()).focused).toBe(false);
  await shot(panel, "help-panel-say-no-en");

  // "What do they want?": readings, each resting on a moment of the transcript it cites.
  const want = await pressIn(panel, "What do they want?");

  await expect(want.getByTestId("help-heading")).toHaveText(
    "What do they want? · Read from what they said",
  );
  await expect(
    want.getByTestId("help-option-note").first().getByTestId("help-citation"),
  ).toHaveAttribute("data-check", "found");

  // While the panel shows, the assistant has no second row of the same buttons.
  await expect(window.getByTestId("help-strip")).toHaveCount(0);

  // The template adds its buttons, and takes them back.
  await pointAndClick(panel, panel.getByTestId("panel-template"));
  await pointAndClick(
    panel,
    panel.locator('[data-testid="panel-template-option"][data-template="sales"]'),
  );
  await expect(buttons).toHaveCount(10);
  await expect(buttons.nth(7)).toHaveText("Handle the objection");
  await expect(panel.getByTestId("panel-template")).toHaveText("Sales call");
  await pointAndClick(panel, panel.getByTestId("panel-template"));
  await pointAndClick(
    panel,
    panel.locator('[data-testid="panel-template-option"][data-template="interview"]'),
  );
  await expect(buttons.nth(8)).toHaveText("Am I leading them?");
  await pointAndClick(panel, panel.getByTestId("panel-template"));
  await pointAndClick(
    panel,
    panel.locator('[data-testid="panel-template-option"][data-template="team"]'),
  );
  await expect(buttons).toHaveCount(9);
  await expect(buttons.nth(8)).toHaveText("Where we disagree");
  await shot(panel, "help-panel-team-en");

  // In Chinese, the same buttons and options in Chinese words.
  await (await core(window)).evaluate((bridge) =>
    bridge.updateSettings({ user: { language: "zh-CN" } }),
  );
  await expect(buttons.first()).toContainText("我该怎么说？");
  await expect(panel.getByTestId("panel-hidden")).toHaveText("在系统支持时，共享屏幕中不显示");
  const sayNoZh = await pressIn(panel, "礼貌地拒绝");
  await expect(sayNoZh.getByTestId("help-heading")).toHaveText(/^礼貌地拒绝 · 对方说：“.+”$/);
  await expectOptions(sayNoZh);
  await expect(sayNoZh.getByTestId("help-option-say").first()).toContainText("周五前回复你");
  await shot(panel, "help-panel-say-no-zh");
  await (await core(window)).evaluate((bridge) =>
    bridge.updateSettings({ user: { language: "en" } }),
  );

  // Folded to a pill: the level bars move with the voices, and pointed at or focused, are Stop.
  await pointAndClick(panel, panel.getByTestId("panel-collapse"));
  const pill = panel.getByTestId("panel-pill");
  await expect(pill).toBeVisible();
  await expect(pill.getByTestId("panel-keys")).toHaveText(mac ? "⌃⌥1–7" : "Ctrl+Shift+1–7");
  const stop = pill.getByTestId("level-stop");
  await expect(stop.getByTestId("level-bar")).toHaveCount(5);
  await shot(panel, "help-panel-pill-en");
  await stop.hover();
  await expect(stop.getByTestId("level-stop-icon")).toBeVisible();
  await pill.getByTestId("panel-expand").hover();
  await expect(stop.getByTestId("level-bars")).toBeVisible();
  await stop.focus();
  await expect(stop.getByTestId("level-stop-icon")).toBeVisible();
  await stop.blur();
  await pointAndClick(panel, pill.getByTestId("panel-expand"));
  await expect(panel.getByTestId("help-buttons")).toBeVisible();

  // Closed, it stays closed for this meeting, and the buttons sit at the top of the assistant,
  // which shows the help asked there; ⌘K shows the panel again.
  await pointAndClick(panel, panel.getByTestId("panel-close"));
  await expect.poll(() => panelOpen(app)).toBe(false);
  const strip = window.getByTestId("help-strip");
  await expect(strip.getByTestId("help-button")).toHaveCount(9);
  await pointAndClick(
    window,
    strip.getByTestId("help-button").filter({ hasText: "Sum up so far" }),
  );
  await expect(window.getByTestId("assistant-question").last()).toContainText("Sum up so far");
  await expect(window.getByTestId("assistant-answer").last()).toHaveAttribute(
    "data-status",
    "done",
    { timeout: 20_000 },
  );
  await window.keyboard.press("ControlOrMeta+k");
  await window.keyboard.type("Show the floating panel");
  await window.keyboard.press("Enter");
  const shown = await panelOf(app);
  await expect(shown.getByTestId("help-heading")).toContainText("Sum up so far");
  await expect(strip).toHaveCount(0);

  // Stop ends the meeting, and the panel goes with it.
  await bar.getByTestId("meeting-stop").click();
  await expect.poll(() => panelOpen(app)).toBe(false);
  await app.close();
});

test("help during the call: Answer their question cites the price sheet, checked; the shortcuts are held only while recording, work with another app in front and can be changed; the level bars hold still with reduced motion", async () => {
  const sources = await createDataFolder();
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    transcriber: { theirLines: BUYER },
  });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  // Documents wait while a meeting records: the price sheet is added first.
  const sheet = join(sources, "Price sheet.txt");
  await writeFile(
    sheet,
    `Price sheet 2026\n\nTeam plan: $25 per seat a month.\n${PRICE}\nPilot: 30 days, free.\n`,
  );
  await (await core(window)).evaluate((bridge, path) => bridge.addDocuments([path]), sheet);
  await expect
    .poll(async () =>
      (await core(window)).evaluate(async (bridge) =>
        (await bridge.listDocuments()).map((document) => document.status),
      ),
    )
    .toEqual(["ready"]);
  const bar = await recordingMeeting(window);
  const panel = await panelOf(app);

  const answer = await pressIn(panel, "Answer their question");
  await expect(answer.getByTestId("help-option-say").first()).toContainText("$45 per seat");
  await expect(answer.getByTestId("help-citation").first()).toHaveAttribute("data-check", "found");
  await expect(answer.getByTestId("help-source")).toHaveText([/^1 · Price sheet · /]);
  await shot(panel, "help-panel-answer-en");

  // Held while recording; a press with another app in front writes that button's options.
  await expect.poll(() => held(app, FIRST_KEYS)).toBe(true);
  await app.evaluate((_electron, kind) => {
    (
      globalThis as unknown as { incarnamindPressHelpKeys(kind: string): void }
    ).incarnamindPressHelpKeys(kind);
  }, "say");
  const said = panel.getByTestId("help-suggestion");
  await expect(said.getByTestId("help-heading")).toContainText("What should I say?", {
    timeout: 20_000,
  });
  await expect(said).toHaveAttribute("data-status", "done", { timeout: 20_000 });

  // Paused, the keys go back to other apps; resumed, they are held again.
  await bar.getByTestId("meeting-pause").click();
  await expect.poll(() => held(app, FIRST_KEYS)).toBe(false);
  await bar.getByTestId("meeting-resume").click();
  await expect.poll(() => held(app, FIRST_KEYS)).toBe(true);

  // The level bars beside the timer; with reduced motion they hold still and only change height.
  const meter = bar.getByTestId("level-bars");
  await expect(meter).toHaveAttribute("data-motion", "on");
  await (await core(window)).evaluate((bridge) =>
    bridge.updateSettings({ device: { motion: "reduced" } }),
  );
  await expect(meter).toHaveAttribute("data-motion", "reduced");
  await expect
    .poll(async () => {
      const heights = await meter
        .getByTestId("level-bar")
        .evaluateAll((bars) => bars.map((each) => (each as HTMLElement).offsetHeight));
      return (
        heights[0] === heights[4] &&
        heights[1] === heights[3] &&
        (heights[2] ?? 0) >= (heights[1] ?? 0)
      );
    })
    .toBe(true);

  // Changed in Settings › Meetings, held at once; keys another app holds say so and stay unchosen.
  const settings = await openSettings(window, "meetings");
  const first = settings.locator('[data-testid="help-keys"][data-kind="say"]');
  await first.click();
  await expect(first).toHaveText("Press the new keys…");
  await window.keyboard.press("Control+Alt+9");
  await expect(first).toHaveText(mac ? "⌃⌥9" : "Ctrl+Alt+9");
  await expect.poll(() => held(app, "Ctrl+Alt+9")).toBe(true);
  await expect.poll(() => held(app, FIRST_KEYS)).toBe(false);
  await app.evaluate(({ globalShortcut }) =>
    globalShortcut.register("Ctrl+Alt+8", () => undefined),
  );
  await first.click();
  await window.keyboard.press("Control+Alt+8");
  await expect(settings.getByTestId("help-keys-problem")).toHaveText(
    "In use by another app · Choose another",
  );
  await expect(first).toHaveText(mac ? "⌃⌥9" : "Ctrl+Alt+9");
  await closeSettings(window);

  // Stopped, nothing is held.
  await bar.getByTestId("meeting-stop").click();
  await expect.poll(() => held(app, "Ctrl+Alt+9")).toBe(false);
  await app.close();
  await removeDataFolder(sources);
});
