/**
 * Records the README's demo of the built app, once in English and once in Chinese, with a real
 * model, the real speech and search models, and test Documents only. Build the test app first
 * (`npx electron-vite build --mode test`), then:
 *
 *   set -a; . ./.env.local; set +a; npm run demo:record
 *
 * `DEMO_LANGUAGE` records one language, `DEMO_OUT` is where the GIFs go (docs/images by default)
 * and `INCARNAMIND_MODEL_CACHE` the folder whose `models/` hold the speech and search models.
 * The recording makes only the cuts, the pointer and the captions: every Answer, Citation,
 * verdict, table, transcript and note on screen is the app's own.
 */
import { mkdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../../src/core/api";
import {
  addToNote,
  askInAssistant,
  assistantAnswers,
  createDataFolder,
  launchApp,
  leaveText,
  openDocumentAt,
  removeDataFolder,
  useOutputLocation as saveFilesIn,
  setLanguage,
  setWindowSize,
} from "../app";
import { connectCall, say, speech } from "./call";
import { FILMED, Film, Hand, installOverlay, renderPills } from "./film";
import { type DemoScript, SCRIPTS } from "./script";
import { buildWorkspace, connectClaude, seedModels, spent, writeDraft } from "./workspace";

const repository = resolve(__dirname, "../..");
const NAME = "demo-tour-2026-10-24";
const OUT = process.env.DEMO_OUT ?? join(repository, "docs", "images");
const FPS = 15;
/** Where the pointer waits at the start of a scene: over nothing, so no card opens under it. */
const PARK = { x: 720, y: 700 };
const LANGUAGES = process.env.DEMO_LANGUAGE
  ? [process.env.DEMO_LANGUAGE as "en" | "zh-CN"]
  : (["en", "zh-CN"] as const);

test.describe.configure({ mode: "serial" });

for (const language of LANGUAGES) {
  test(`records the README demo in ${language}`, async () => {
    await record(SCRIPTS[language]);
  });
}

const bridgeOf = (window: Page) =>
  window.evaluateHandle(() => (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind);

/** Opens a Mind by its name from ⌘K, as a person does. */
async function openMind(window: Page, title: string): Promise<void> {
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type(title);
  const item = window.getByTestId("palette-item").filter({ hasText: title }).first();
  await expect(item).toBeVisible();
  await item.click();
  await expect(window.getByTestId("palette-search")).toBeHidden();
}

/** Closes the Documents opened beside the Mind, and the card and the notice that follow them. */
async function clearAside(window: Page): Promise<void> {
  const tabs = window.getByTestId("document-tab");
  while ((await tabs.count()) > 0) await tabs.first().getByTestId("tab-close").click();
  await window.keyboard.press("Escape");
  // A click on nothing puts away the Citation's card, which Esc leaves open.
  await leaveText(window);
  await expect(window.getByTestId("citation-card")).toHaveCount(0);
  await dismissNotice(window);
}

async function dismissNotice(window: Page): Promise<void> {
  const notice = window.getByTestId("undo-message");
  if ((await notice.count()) > 0) await window.keyboard.press("Escape");
  await expect(notice).toHaveCount(0);
}

interface Take {
  window: Page;
  hand: Hand;
  film: Film;
  script: DemoScript;
  app: Awaited<ReturnType<typeof launchApp>>["app"];
}

/** Runs a command from ⌘K: types the start of its name, as a hand does, and picks it. */
async function command({ window, hand }: Take, typed: string, name: string): Promise<void> {
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await hand.type(typed);
  await expect(window.getByTestId("palette-item").filter({ hasText: name }).first()).toBeVisible();
  await hand.hold(250);
  await window.keyboard.press("Enter");
}

async function record(script: DemoScript): Promise<void> {
  const dataDir = await createDataFolder();
  const sources = await createDataFolder();
  const files = await createDataFolder();
  const output = join(files, script.outputFolder);
  await mkdir(output, { recursive: true });
  await seedModels(dataDir);
  const { app, window } = await launchApp(dataDir, { realEngines: true });
  let film: Film | undefined;
  let finished = false;
  try {
    await setWindowSize(app, window, FILMED.width, FILMED.height);
    if (script.language !== "en") await setLanguage(window, script.language);
    console.log(
      `${script.language}: model ${(await connectClaude(window)).replace(/.*_/, "key ")}`,
    );
    await saveFilesIn(window, output);
    const workspace = await buildWorkspace(window, script, sources);

    // The draft: its title, a cited paragraph the assistant adds below it, then the User's own text.
    await openMind(window, script.paper.title);
    await window.getByTestId("mind-editor").click();
    const seeded = await askInAssistant(window, script.seedQuestion, { timeout: 90_000 });
    await expect(seeded.getByTestId("citation").nth(1)).toBeVisible();
    await addToNote(window, seeded);
    await writeDraft(window, script, workspace.paperId);
    const covered = script.paper.blocks[5]?.text ?? "";
    await expect(window.getByTestId("mind-editor")).toContainText(covered.slice(0, 30));

    // The sidebar shows both Folders open, as a workspace in use has them.
    const sidebar = window.getByTestId("library-folders");
    if (!(await sidebar.getByText(script.options.title).isVisible())) {
      await sidebar.getByText(script.folders.contracts).first().click();
    }
    if (!(await sidebar.getByText(script.paper.title).isVisible())) {
      await sidebar.getByText(script.folders.credit).first().click();
    }
    const voices = join(files, "voices");
    const them = await speech(voices, "them", script.call.voices.them, script.call.them);
    const me = await speech(voices, "me", script.call.voices.me, script.call.me);
    await connectCall(app);

    await installOverlay(window);
    const pills = await renderPills(Object.values(script.captions), join(files, "pills"));
    film = await Film.start(window, join(files, "frames"), pills);
    const hand = new Hand(window, film);
    const take: Take = { window, hand, film, script, app };

    await window.getByTestId("assistant-new-chat").click();
    await window.keyboard.press("ControlOrMeta+j");
    await expect(window.getByTestId("composer-input")).toBeFocused();
    await window.keyboard.type(script.question);
    await hand.place(1130, 640);
    await hand.begin();

    const scenes: [string, () => Promise<void>][] = [
      ["ask", () => askAndClick(take)],
      ["check", () => checkTheDraft(take)],
      ["compare", () => compareTermSheets(take)],
      ["call", () => recordACall(take, { them, me })],
      ["export", () => exportToWord(take)],
      ["end card", () => endCard(take)],
    ];
    for (const [name, run] of scenes) {
      const before = film.kept;
      await run();
      console.log(`${script.language}: ${name} ${(film.kept - before).toFixed(1)} s`);
    }

    await film.stop();
    const gif = join(OUT, `${NAME}-${script.language}.gif`);
    const size = await film.encodeGif(gif, { fps: FPS, colors: 256 });
    console.log(
      `${script.language}: ${gif} ${(size / 1e6).toFixed(2)} MB, ${film.kept.toFixed(1)} s, spent $${(await spent(window)).toFixed(3)}`,
    );
    finished = true;
  } finally {
    if (film && !finished) {
      await film.stop().catch(() => undefined);
      await film
        .encodeGif(join(OUT, `${NAME}-${script.language}-unfinished.gif`), {
          fps: FPS,
          colors: 256,
        })
        .catch(() => undefined);
    }
    await app.close().catch(() => undefined);
    for (const folder of [dataDir, sources, files]) await removeDataFolder(folder);
  }
}

/** Scene 1: the Question is already typed; the Answer streams in with Citations; one is clicked. */
async function askAndClick({ window, hand, film, script }: Take): Promise<void> {
  hand.caption(script.captions.ask);
  await hand.hold(800);
  await window.keyboard.press("Enter");
  const answer = assistantAnswers(window).last();
  await hand.hold(1000);
  await hand.cut(() =>
    expect
      .poll(async () => (await answer.getByTestId("assistant-answer-text").innerText()).length, {
        timeout: 90_000,
      })
      .toBeGreaterThan(40),
  );
  film.speed(1.8);
  const streamed = film.kept;
  while ((await answer.getAttribute("data-status")) !== "done" && film.kept - streamed < 2.8) {
    await hand.hold(100);
  }
  film.speed(1);
  await hand.cut(() => expect(answer).toHaveAttribute("data-status", "done", { timeout: 90_000 }));
  // Markdown the Answer shows as typed ("**") is the app's own problem, but not what the demo opens on.
  expect(await answer.getByTestId("assistant-answer-text").innerText()).not.toContain("**");
  const citation = answer.getByTestId("citation").first();
  await expect(citation).toHaveAttribute("data-check", "found");
  await hand.hold(1200);
  hand.caption(script.captions.click);
  await hand.click(citation.getByTestId("citation-chip"));
  await expect(
    window.getByTestId("viewer").locator("[data-quote-highlight]").first(),
  ).toBeVisible();
  await hand.hold(2400);
}

/** Scene 2: Check this Mind marks each sentence; the contradicted one opens its card. */
async function checkTheDraft(take: Take): Promise<void> {
  const { window, hand, script } = take;
  await hand.cut(() => clearAside(window), PARK);
  hand.caption(script.captions.check);
  await hand.hold(300);
  await command(take, script.words.typed.check, script.words.checkCommand);
  // The question comes with its estimate once the claims are found: a moment the recording leaves out.
  const ask = window.getByTestId("mind-check-check");
  await hand.cut(() => expect(ask).toBeVisible({ timeout: 90_000 }));
  await hand.click(ask);
  await hand.hold(900);
  await hand.cut(async () => {
    await expect(window.getByTestId("badge-stop")).toHaveCount(0, { timeout: 180_000 });
    await expect(window.getByTestId("claim-mark").first()).toBeVisible();
    await window.keyboard.press("Escape");
  });
  const verdicts = await window
    .getByTestId("claim-mark")
    .evaluateAll((all) => all.map((mark) => mark.getAttribute("data-check")));
  expect(verdicts.filter((verdict) => verdict === "contradicted")).toHaveLength(1);
  await hand.hold(300);
  await hand.click(window.locator('[data-testid="claim-mark"][data-check="contradicted"]'));
  await expect(window.getByTestId("claim-card")).toContainText(script.words.contradicted);
  await expect(window.getByTestId("claim-card")).toContainText("36.8");
  await hand.hold(2300);
}

/** Scene 3: Compare Documents over three term sheets, and a cell's page. */
async function compareTermSheets(take: Take): Promise<void> {
  const { window, hand, script } = take;
  await hand.cut(async () => {
    await window.keyboard.press("Escape");
    await openMind(window, script.options.title);
    await window.getByTestId("assistant-new-chat").click();
  }, PARK);
  hand.caption(script.captions.compare);
  await hand.hold(300);
  await command(take, script.words.typed.compare, script.words.compareCommand);
  const sheet = window.getByTestId("compare-sheet");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByTestId("compare-sheet-counted")).toHaveText(/^3/);
  for (const [index, question] of script.compareQuestions.entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await hand.type(question);
  }
  await hand.hold(300);
  await hand.click(sheet.getByTestId("compare-sheet-make"));
  const table = window.getByTestId("comparison-table");
  await expect(table).toBeVisible();
  // A provider that refuses stops the table's cells: that take is not kept.
  await expect(table.locator(".comparison-cell-reading").first()).toBeVisible();
  await hand.hold(900);
  await hand.cut(async () => {
    await expect(table.getByTestId("comparison-progress")).toHaveCount(0, { timeout: 180_000 });
    await expect(table.locator(".comparison-cell-reading:visible")).toHaveCount(0);
  });
  await hand.hold(400);
  await hand.click(
    table.getByTestId("comparison-cell").nth(2).getByTestId("citation-chip").first(),
  );
  await expect(
    window.getByTestId("viewer").locator("[data-quote-highlight]").first(),
  ).toBeVisible();
  await hand.hold(1400);
}

/** Scene 4: a test call; the transcript, call help, and the notes after Stop. */
async function recordACall(take: Take, voices: { them: number[]; me: number[] }): Promise<void> {
  const { window, hand, script, app } = take;
  const notice = window.getByTestId("meeting-notice");
  await hand.cut(async () => {
    await clearAside(window);
    await openMind(window, script.paper.title);
    await command(take, script.words.typed.record, script.words.recordCommand);
    await window.getByTestId("meeting-start").getByTestId("meeting-start-record").click();
    await expect(notice).toBeVisible();
  }, PARK);
  hand.caption(script.captions.call);
  await hand.hold(1200);
  await hand.click(notice.getByTestId("meeting-notice-start"));
  const bar = window.getByTestId("meeting-bar");
  await expect(bar).toHaveAttribute("data-status", "recording");
  await hand.hold(400);

  await hand.cut(async () => {
    await window.getByTestId("assistant-new-chat").click();
    await bar.getByTestId("meeting-call-help").click();
    await window.getByTestId("mind-editor").click();
    await window.keyboard.type(script.callNote);
    await say(app, "system", voices.them);
    await expect(bar.getByTestId("meeting-transcript-toggle")).toBeEnabled({ timeout: 240_000 });
    await bar.getByTestId("meeting-transcript-toggle").click();
    await expect(window.getByTestId("transcript-line")).toHaveCount(1);
  });
  await hand.hold(900);
  await hand.click(
    window.getByTestId("help-button").filter({ hasText: script.words.answerTheirQuestion }).first(),
  );
  const help = assistantAnswers(window).last();
  await hand.hold(600);
  await hand.cut(() => expect(help).toHaveAttribute("data-status", "done", { timeout: 120_000 }));
  await hand.hold(1800);

  await hand.cut(async () => {
    await say(app, "microphone", voices.me);
    const lines = window.getByTestId("transcript-line");
    await expect(lines.last().getByTestId("transcript-speaker")).toHaveText(script.words.me, {
      timeout: 120_000,
    });
    // The line grows as the engine reads the rest of what was said: the cut ends on the figures.
    await expect
      .poll(async () => (await lines.allInnerTexts()).join(" "), { timeout: 60_000 })
      .toMatch(script.words.heard);
  });
  await hand.hold(500);
  await hand.click(bar.getByTestId("meeting-stop"));
  await hand.cut(() =>
    expect(window.getByTestId("meeting-fill-line")).toContainText(script.words.filled, {
      timeout: 240_000,
    }),
  );
  await hand.hold(1800);
}

/** Scene 5: Export to Word, the Saved line, and the file as the app shows it. */
async function exportToWord({ window, hand, script }: Take): Promise<void> {
  await hand.cut(async () => {
    await clearAside(window);
    await openMind(window, script.paper.title);
  }, PARK);
  hand.caption(script.captions.export);
  await hand.hold(300);
  await hand.click(window.getByTestId("export-mind"));
  const menu = window.getByTestId("export-menu");
  await expect(menu).toBeVisible();
  if ((await menu.getByTestId("export-as-docx").count()) === 0) {
    await hand.click(menu.getByTestId("export-more"));
  }
  await hand.click(window.getByTestId("export-as-docx"));
  const done = window.getByTestId("export-result").getByTestId("export-done");
  await expect(done).toContainText(script.words.saved, { timeout: 60_000 });
  await hand.hold(1300);

  await hand.cut(async () => {
    const path = (await done.getAttribute("title")) ?? "";
    const documentId = await (await bridgeOf(window)).evaluate(async (bridge, file) => {
      const [added] = (await bridge.addDocuments([file], {})).documents;
      return added?.id ?? "";
    }, path);
    await window.waitForFunction(
      async (id) =>
        (
          await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments()
        ).some((each) => each.id === id && each.status === "ready"),
      documentId,
      { timeout: 60_000 },
    );
    // The file alone beside the assistant, so its page and footnotes read at the recording's size.
    const open = window.getByTestId("mind-tab");
    while ((await open.count()) > 0) {
      await open.first().click();
      await window.keyboard.press("ControlOrMeta+w");
    }
    await openDocumentAt(window, { documentId });
    await expect(window.getByTestId("viewer")).toBeVisible();
    await dismissNotice(window);
    await expect(window.getByTestId("viewer-docx")).toHaveAttribute("data-rendered", "yes");
  });
  await hand.hold(1500);
  // Down the page to its footnotes, with the wheel.
  const pane = window.getByTestId("viewer-docx");
  await hand.moveTo(pane);
  const footnote = pane.getByText(script.words.footnote).last();
  const [paneBox, noteBox] = await Promise.all([pane.boundingBox(), footnote.boundingBox()]);
  let remaining =
    (noteBox?.y ?? 0) + (noteBox?.height ?? 0) + 30 - ((paneBox?.y ?? 0) + (paneBox?.height ?? 0));
  while (remaining > 0) {
    const step = Math.min(70, remaining);
    await window.mouse.wheel(0, step);
    remaining -= step;
    await hand.hold(60);
  }
  await hand.hold(1500);
}

/** The end card: a still, from the README's own words. */
async function endCard({ hand, script }: Take): Promise<void> {
  const readme = await readFile(join(repository, script.readme), "utf8");
  const [headline, shows, free, address] = script.endCard.lines;
  const lines = [
    headline,
    shows,
    ...(readme.includes(script.endCard.audience) ? [script.endCard.audience] : []),
    free,
    address,
  ].filter((line): line is string => line !== undefined);
  const icon = await readFile(join(repository, "build/icon/svg/mac-light.svg"));
  hand.caption(null);
  await hand.cut(() =>
    hand.endCard({ icon: `data:image/svg+xml;base64,${icon.toString("base64")}`, lines }),
  );
  await hand.hold(2500);
}
