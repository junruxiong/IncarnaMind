import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  acceptPlainTextKeys,
  addDocuments,
  addTideConnector,
  assistantAnswers,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openSteps,
  pointAndClick,
  pointAt,
  removeDataFolder,
  setWindowSize,
  sidebarFolder,
} from "./app";

/**
 * An Answer's steps in the assistant (#391): while it is written, one line
 * says the step it is on, each held a second; written, the line folds to what
 * it searched and how many sources it cites, and opens, by click, Enter or
 * Space, to a timeline of the reasoning, the Mind read (#544), the searches
 * with their sources, the calls and the check, with Regenerate at its end. An
 * empty conversation says once what the next Question reads and searches;
 * nothing sits under the header, and only Copy · Add to note under an Answer.
 * The scripted model reasons as a model that summarizes its reasoning does
 * (an Anthropic key, so it is kept). Set INCARNAMIND_SCREENSHOTS to a folder
 * to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
async function screenshot(window: Page, name: string): Promise<void> {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

const SPRING = "Spring tides happen at new moon and at full moon.";
const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", SPRING] },
]);
const MODEL = "claude-haiku-5-5";
const LONG_FOLDER = "Thesis: urban heat islands and the night-time cooling of city parks";

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

const core = (window: Page) => ({
  /** An Anthropic key, whose models summarize their reasoning, with Questions and tagging allowed. */
  anthropic: () =>
    window.evaluate(async (modelId) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const provider = await bridge.saveChatProvider({
        kind: "anthropic",
        apiKey: "sk-ant-test",
        modelId,
      });
      if (!provider.service) return;
      await bridge.allowDataFlow("chat", provider.service.id);
      await bridge.allowDataFlow("tagging", provider.service.id);
    }, MODEL),
  /** A Mind, in a Folder if given (its id and name), opened from the sidebar. */
  mind: async (title: string, folder: { id: string; name: string } | null = null) => {
    await window.evaluate(
      async ({ name, folderId }) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.createMind({
          title: name,
          folderId,
        }),
      { name: title, folderId: folder?.id ?? null },
    );
    if (folder) {
      const row = sidebarFolder(window, folder.name);
      if ((await row.getAttribute("data-expanded")) !== "true") {
        await pointAndClick(window, row.getByTestId("folder-row"));
      }
    }
    await pointAndClick(window, window.getByTestId("mind-list-item").filter({ hasText: title }));
    await expect(window.getByTestId("mind-title")).toHaveValue(title);
  },
  /** A Folder holding every Document so far. */
  folderOfAll: (name: string) =>
    window.evaluate(async (folderName) => {
      const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const folder = await bridge.createLibraryGroup({ name: folderName, description: "" });
      for (const document of await bridge.listDocuments()) {
        await bridge.assignDocumentGroup(document.id, folder.id);
      }
      return folder.id;
    }, name),
  language: (language: "en" | "zh-CN") =>
    window.evaluate(
      (value) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
          user: { language: value },
        }),
      language,
    ),
});

/** The app with Tides added; the chat model on an Anthropic key, or, `local`, on this computer. */
async function start(options: { documents?: boolean; local?: boolean } = {}) {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  if (options.local) {
    await connectLocalChatModel(window);
  } else {
    await acceptPlainTextKeys(window);
    await core(window).anthropic();
  }
  if (options.documents !== false) {
    await writeFile(join(sources, "Tides.pdf"), TIDES);
    await addDocuments(window, [join(sources, "Tides.pdf")]);
  }
  return { app, window };
}

/** Asks in the assistant, as a person does, and resolves with the Answer once written. */
async function ask(window: Page, question: string): Promise<Locator> {
  const answers = assistantAnswers(window);
  const before = await answers.count();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type(question);
  await window.keyboard.press("Enter");
  await expect(answers).toHaveCount(before + 1, { timeout: 15_000 });
  const answer = answers.last();
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 30_000 });
  return answer;
}

/** The steps the live line showed, each with when it showed, from now on. */
async function watchLiveLine(window: Page): Promise<() => Promise<{ step: string; at: number }[]>> {
  await window.evaluate(() => {
    const shown: { step: string; at: number }[] = [];
    (globalThis as unknown as { shownSteps: typeof shown }).shownSteps = shown;
    const record = () => {
      const live = document.querySelector(
        '[data-testid="assistant"] [data-testid="answer-steps-live"]',
      );
      const step = live?.getAttribute("data-step") ?? "none";
      if (shown.at(-1)?.step !== step) shown.push({ step, at: performance.now() });
    };
    new MutationObserver(record).observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["data-step"],
    });
  });
  return () =>
    window.evaluate(
      () => (globalThis as unknown as { shownSteps: { step: string; at: number }[] }).shownSteps,
    );
}

const stepsOf = (timeline: Locator) =>
  timeline
    .getByTestId("answer-step")
    .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-step")));

test("while an Answer is written, its line moves Reading → Searching → Writing → Checking, each up a second or more; written, it folds to what it searched and its sources", async () => {
  const { app, window } = await start();
  await core(window).mind("Lit review: tides");
  const shown = await watchLiveLine(window);

  // "Slowly": the scripted model takes a while over each step.
  const answer = await ask(window, "When do spring tides happen? Answer slowly.");

  // The Answer's own line takes over from the one on its way, on the same step.
  const steps = (await shown())
    .filter((each) => each.step !== "none")
    .filter((each, index, all) => each.step !== all[index - 1]?.step);
  expect(steps.map((each) => each.step)).toEqual(["reading", "searching", "writing", "checking"]);
  // Each is up at least a second before the next replaces it; the last goes as the Answer ends.
  for (let index = 0; index + 1 < steps.length; index++) {
    const step = steps[index];
    const next = steps[index + 1];
    expect((next?.at ?? 0) - (step?.at ?? 0)).toBeGreaterThanOrEqual(950);
  }
  const line = answer.getByTestId("answer-steps");
  await expect(line).toHaveText("Searched All Documents · 1 source");
  await expect(line).toHaveAttribute("aria-expanded", "false");
  await expect(answer.getByTestId("answer-timeline")).toHaveCount(0);
  await screenshot(window, "steps-folded-en");
  await app.close();
});

test("the folded line opens by click, Enter and Space, at once, to the reasoning, the Mind read, the search and its source, and the check; the source opens beside the Mind", async () => {
  const { app, window } = await start();
  await core(window).mind("Moon notes");
  await core(window).mind("Lit review: tides");
  const answer = await ask(window, "When do spring tides happen?");
  const line = answer.getByTestId("answer-steps");

  // A click opens it, with nothing moving: the chevron turns down.
  await pointAndClick(window, line);
  await expect(line).toHaveAttribute("aria-expanded", "true");
  const timeline = answer.getByTestId("answer-timeline");
  await expect(timeline).toBeVisible();
  expect(await answer.evaluate((node) => node.getAnimations({ subtree: true }).length)).toBe(0);
  expect(await stepsOf(timeline)).toEqual(["reasoning", "read", "search", "check"]);
  const reasoning = timeline.locator('[data-step="reasoning"]');
  await expect(reasoning).toContainText(new RegExp(`^${MODEL} thought for \\d+s`));
  await expect(reasoning.getByTestId("answer-reasoning")).toContainText(
    "When do spring tides happen?",
  );
  await expect(reasoning.getByTestId("answer-reasoning-more")).toHaveCount(0);
  await expect(timeline.locator('[data-step="read"]')).toHaveText("Read Lit review: tides");
  const search = timeline.locator('[data-step="search"]');
  await expect(search).toContainText("Searched All Documents for “When do spring tides happen?”");
  await expect(search.getByTestId("answer-step-found")).toContainText(/passages? found, 1 used/);
  const source = search.getByTestId("answer-step-source");
  await expect(source).toHaveCount(1);
  await expect(source).toHaveText("Tidesp. 2");
  await expect(timeline.locator('[data-step="check"]')).toHaveText(
    "Checked 1 Citation against its source · found",
  );
  await expect(answer.getByTestId("answer-regenerate")).toBeVisible();
  await screenshot(window, "steps-open-en");

  // Enter and Space fold and open it again.
  await line.focus();
  await window.keyboard.press("Enter");
  await expect(line).toHaveAttribute("aria-expanded", "false");
  await expect(timeline).toHaveCount(0);
  await window.keyboard.press("Space");
  await expect(line).toHaveAttribute("aria-expanded", "true");
  await expect(timeline).toBeVisible();

  // The source opens its Document beside the Mind, at its page, the quote washed.
  await pointAndClick(window, source);
  await expect(window.getByTestId("viewer-title")).toHaveText("Tides");
  await expect(window.getByTestId("pdf-page-number")).toHaveValue("2");
  await expect(
    window.getByTestId("viewer").locator('[data-page-number="2"] [data-quote-highlight]'),
  ).toHaveText([SPRING]);
  const groups = window.getByTestId("tab-group");
  await expect(
    groups.first().getByTestId("mind-tab").filter({ hasText: "Lit review: tides" }),
  ).toHaveAttribute("aria-selected", "true");

  // Read goes to that Mind's tab.
  await pointAndClick(
    window,
    window.getByTestId("mind-list-item").filter({ hasText: "Moon notes" }),
  );
  await expect(window.getByTestId("mind-title")).toHaveValue("Moon notes");
  await pointAndClick(window, timeline.getByTestId("answer-step-read"));
  await expect(window.getByTestId("mind-title")).toHaveValue("Lit review: tides");
  await app.close();
});

test("in Chinese, the line and the timeline; after a restart the steps are still there, folded", async () => {
  const first = await start();
  let { window } = first;
  await core(window).mind("Lit review: tides");
  let answer = await ask(window, "When do spring tides happen?");
  // Kept in the Mind, so its link leads back to the conversation after the restart.
  await pointAt(window, answer.getByTestId("assistant-answer-text"));
  await pointAndClick(window, answer.getByTestId("answer-add"));
  await expect(answer.getByTestId("answer-added")).toBeVisible();
  await core(window).language("zh-CN");
  await expect(answer.getByTestId("answer-steps")).toHaveText("已搜索全部文档 · 1 个来源");
  let timeline = await openSteps(window, answer);
  await expect(timeline.locator('[data-step="reasoning"]')).toContainText(
    new RegExp(`^${MODEL} 思考了 \\d+ 秒`),
  );
  await expect(timeline.locator('[data-step="read"]')).toHaveText("读了“Lit review: tides”");
  await expect(timeline.locator('[data-step="search"]')).toContainText(
    "在全部文档中搜索“When do spring tides happen?”",
  );
  await expect(timeline.getByTestId("answer-step-found")).toContainText("用了 1 段");
  await expect(timeline.locator('[data-step="check"]')).toHaveText(
    "把 1 处引用和原文核对 · 找到了",
  );
  await expect(answer.getByTestId("answer-regenerate")).toHaveText("重新生成");
  await screenshot(window, "steps-open-zh");
  const summary = await timeline.getByTestId("answer-reasoning").textContent();
  await first.app.close();

  const second = await launchApp(dataDir, { fakeChat: true });
  window = second.window;
  await setWindowSize(second.app, window, 1440, 900);
  await pointAndClick(window, window.getByTestId("mind-list-item"));
  await pointAndClick(window, window.getByTestId("conversation-link"));
  answer = assistantAnswers(window).first();
  const line = answer.getByTestId("answer-steps");
  await expect(line).toHaveText("已搜索全部文档 · 1 个来源");
  await expect(line).toHaveAttribute("aria-expanded", "false");
  timeline = await openSteps(window, answer);
  expect(await stepsOf(timeline)).toEqual(["reasoning", "read", "search", "check"]);
  await expect(timeline.getByTestId("answer-reasoning")).toHaveText(summary ?? "");
  await second.app.close();
});

test("an empty conversation says once what it reads and searches, following focus; nothing under the header, and only Copy · Add to note under an Answer", async () => {
  const { app, window } = await start();
  const thread = window.getByTestId("assistant-thread");
  await core(window).mind("Alpha");
  const reads = window.getByTestId("assistant-reads");
  await expect(reads).toHaveText("Reads “Alpha” · searches All Documents");
  // In the conversation, above its words: the header has nothing under it.
  await expect(thread.getByTestId("assistant-reads")).toBeVisible();
  const header = await window.getByTestId("assistant-header").boundingBox();
  const threadBox = await thread.boundingBox();
  expect((threadBox?.y ?? 0) - ((header?.y ?? 0) + (header?.height ?? 0))).toBeLessThan(2);
  await expect(reads).toHaveCSS("font-size", "14px");
  await expect(window.getByTestId("assistant-empty")).toHaveCSS("font-size", "14px");
  await core(window).mind("Beta");
  await expect(reads).toHaveText("Reads “Beta” · searches All Documents");
  await pointAndClick(window, window.getByTestId("open-library"));
  await expect(reads).toHaveText("Searches All Documents");
  await screenshot(window, "steps-empty-en");

  // Asked: the line goes; under the Answer, only Copy · Add to note.
  await pointAndClick(window, window.getByTestId("mind-list-item").filter({ hasText: "Beta" }));
  const answer = await ask(window, "When do spring tides happen?");
  await expect(reads).toHaveCount(0);
  await window.mouse.move(10, 10);
  const actions = answer.getByTestId("answer-actions");
  await expect(actions).toHaveText("CopyAdd to note");
  await expect(actions).not.toContainText(MODEL);
  await expect(answer.getByTestId("answer-copy")).toHaveCSS("font-size", "13px");
  await expect(answer.getByTestId("answer-copy")).toHaveCSS("font-weight", "600");
  await app.close();
});

test("at a 320px assistant the folded line is cut, never wrapped, in English and Chinese; the reasoning shows Show more only when it is longer than three lines; one reading size", async () => {
  const { app, window } = await start();
  const folder = await core(window).folderOfAll(LONG_FOLDER);
  await core(window).mind("Night cooling", { id: folder, name: LONG_FOLDER });
  expect(Math.round((await window.getByTestId("assistant").boundingBox())?.width ?? 0)).toBe(320);
  const answer = await ask(window, "When do spring tides happen?");
  const line = answer.getByTestId("answer-steps");
  const full = `Searched ${LONG_FOLDER} · 1 source`;
  await expect(line).toHaveAttribute("title", full);
  const cut = async () => {
    const box = await line.boundingBox();
    expect(box?.height ?? 0).toBeLessThan(30);
    expect(
      await line
        .locator(".answer-steps-text")
        .evaluate((text) => text.scrollWidth > text.clientWidth),
    ).toBe(true);
  };
  await cut();
  // One reading size: the Question, the Answer, the steps are 14px; the note stays larger.
  await expect(
    window.getByTestId("assistant-question").locator(".assistant-question-bubble"),
  ).toHaveCSS("font-size", "14px");
  await expect(answer.locator(".assistant-answer-text p").first()).toHaveCSS("font-size", "14px");
  await expect(answer.locator(".assistant-answer-text p").first()).toHaveCSS("line-height", "22px");
  await expect(line).toHaveCSS("font-size", "14px");
  const timeline = await openSteps(window, answer);
  await expect(timeline).toHaveCSS("font-size", "14px");
  // A short reasoning shows whole.
  await expect(timeline.getByTestId("answer-reasoning-more")).toHaveCount(0);
  await screenshot(window, "steps-narrow-en");

  await core(window).language("zh-CN");
  await expect(line).toHaveAttribute("title", `已搜索“${LONG_FOLDER}” · 1 个来源`);
  await cut();
  await screenshot(window, "steps-narrow-zh");
  await core(window).language("en");

  // A long one is cut at three lines, with Show more, which shows it whole.
  const long = await ask(
    window,
    "When do spring tides happen, how much higher do they rise than neap tides along the coast, and does the Sun's pull add to the Moon's at new and full moon in every harbour?",
  );
  const steps = await openSteps(window, long);
  const reasoning = steps.getByTestId("answer-reasoning");
  const more = steps.getByTestId("answer-reasoning-more");
  await expect(more).toHaveText("Show more");
  const lineHeight = 20;
  expect((await reasoning.boundingBox())?.height ?? 0).toBeLessThanOrEqual(3 * lineHeight + 1);
  await pointAndClick(window, more);
  await expect(more).toHaveText("Show less");
  expect((await reasoning.boundingBox())?.height ?? 0).toBeGreaterThan(3 * lineHeight + 1);
  await screenshot(window, "steps-show-more-en");
  await app.close();
});

test("past three sources, +N more shows the rest, and each source names its Document and where", async () => {
  const { app, window } = await start({ documents: false });
  const places = ["Brest", "Dover", "Calais", "Leeds", "Hull"];
  for (const place of places) {
    await writeFile(
      join(sources, `${place}.md`),
      `# ${place}\n\nSpring tides at ${place} rise high.`,
    );
  }
  await addDocuments(
    window,
    places.map((place) => join(sources, `${place}.md`)),
  );
  await core(window).mind("Harbours");
  // "Each": the scripted model cites every Document its search found.
  const answer = await ask(window, "How high do spring tides rise at each harbour?");
  // Its search finds more than three of them: each cited is a source.
  const line = await answer.getByTestId("answer-steps").textContent();
  const cited = Number(/· (\d+) sources$/.exec(line ?? "")?.[1]);
  expect(cited).toBeGreaterThan(3);
  const timeline = await openSteps(window, answer);
  const listed = timeline.getByTestId("answer-step-source");
  await expect(listed).toHaveCount(3);
  await expect(listed.first()).toContainText(/^(Brest|Dover|Calais|Leeds|Hull)\S/);
  const more = timeline.getByTestId("answer-step-sources-more");
  await expect(more).toHaveText(`+${cited - 3} more`);
  await pointAndClick(window, more);
  await expect(listed).toHaveCount(cited);
  await expect(more).toHaveCount(0);
  await app.close();
});

test("a Connector call waiting for approval: the line says so, its card waits outside the steps, and allowing it carries the steps on", async () => {
  const { app, window } = await start({ documents: false, local: true });
  await addTideConnector(window);
  await core(window).mind("Boat trips");
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("Please book_boat from Dover");
  await window.keyboard.press("Enter");
  const answer = assistantAnswers(window).first();
  const card = answer.getByTestId("approval-card");
  await expect(card).toBeVisible({ timeout: 15_000 });
  const live = answer.getByTestId("answer-steps-live");
  await expect(live).toHaveAttribute("data-step", "waiting");
  await expect(live).toHaveText("Waiting for your approval");
  await expect(live.locator(".answer-steps-dot--waiting")).toBeVisible();
  // The card is under the line, never inside the steps.
  await expect(answer.getByTestId("answer-timeline")).toHaveCount(0);
  const lineBox = await live.boundingBox();
  const cardBox = await card.boundingBox();
  expect(cardBox?.y ?? 0).toBeGreaterThan((lineBox?.y ?? 0) + (lineBox?.height ?? 0));
  await screenshot(window, "steps-approval-en");

  await pointAndClick(window, card.getByTestId("approval-allow-once"));
  const consent = window.getByTestId("consent-dialog");
  await pointAndClick(window, consent.getByTestId("consent-allow"));
  await expect(answer).toHaveAttribute("data-status", "done", { timeout: 15_000 });
  const timeline = await openSteps(window, answer);
  // No search: the folded line says the Mind read, so the timeline doesn't.
  expect(await stepsOf(timeline)).toEqual(["answered", "call"]);
  await expect(timeline.getByTestId("answer-connector-call")).toHaveAttribute(
    "data-approval",
    "allowed",
  );
  await app.close();
});
