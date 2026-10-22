import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  askInAssistant,
  assistantAnswers,
  createDataFolder,
  dismissChatSetup,
  expectReadyDocuments,
  gettingStarted,
  launchApp,
  nameMind,
  newMind,
  openPageMenu,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setSidebarWidth,
  setWindowSize,
  slideSelect,
  welcomeStep as step,
  useLocalChatModel,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function markSteps(window: Page, done: Record<string, boolean>): Promise<void> {
  await window.evaluate(async (patch) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const { device } = await bridge.getSettings();
    await bridge.updateSettings({
      device: { gettingStarted: { ...device.gettingStarted, ...patch } },
    });
  }, done);
}

const NONE = { mistakeSeen: false, askedOwn: false, askedReady: false, indexed: false };

const STATES = [
  { name: "Welcome", done: NONE, steps: ["open", "later", "later"] },
  {
    name: "first step done",
    done: { ...NONE, mistakeSeen: true },
    steps: ["done", "open", "later"],
  },
  {
    name: "second step done",
    done: { ...NONE, mistakeSeen: true, askedReady: true },
    steps: ["done", "done", "open"],
  },
  {
    name: "all done, no model",
    done: { ...NONE, mistakeSeen: true, askedReady: true, indexed: true },
    steps: ["done", "done", "done"],
  },
];

async function measure(window: Page) {
  const card = await window.getByTestId("first-open-steps").boundingBox();
  const title = await window.getByTestId("mind-editor").locator("h1").boundingBox();
  if (!card || !title) throw new Error("The first screen isn't laid out.");
  return { card: Math.round(card.height), title: Math.round(title.y) };
}

async function showState(window: Page, { done, steps }: (typeof STATES)[number]): Promise<void> {
  await markSteps(window, done);
  for (const [index, state] of steps.entries()) {
    await expect(window.getByTestId("first-open-step").nth(index)).toHaveAttribute(
      "data-state",
      state,
    );
  }
  if (steps.every((state) => state === "done")) {
    await expect(window.getByTestId("first-open-choose-model")).toBeVisible();
  }
}

async function expectOneHeight(window: Page, where: string) {
  const first = STATES[0];
  if (!first) throw new Error("No states.");
  await showState(window, first);
  const reference = await measure(window);
  for (const state of STATES.slice(1)) {
    await showState(window, state);
    expect(await measure(window), `${where}, ${state.name}`).toEqual(reference);
  }
  return reference;
}

test("the Welcome card keeps one height through every step, so the page's text never moves (#1195)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await expect(window.getByTestId("first-open-steps")).toBeVisible();

  const heights = new Map<string, Awaited<ReturnType<typeof measure>>>();
  for (const language of ["en", "zh-CN"] as const) {
    await setLanguage(window, language);
    for (const width of [900, 1100, 1440]) {
      await setWindowSize(app, window, width, 900);
      heights.set(
        `${language} ${width}`,
        await expectOneHeight(window, `${language} at ${width} px`),
      );
    }
  }

  await useLocalChatModel(window);
  for (const language of ["en", "zh-CN"] as const) {
    await setLanguage(window, language);
    for (const width of [900, 1100, 1440]) {
      await setWindowSize(app, window, width, 900);
      await expect(window.getByTestId("first-open-next").first()).toBeVisible();
      expect(await measure(window), `${language} at ${width} px, all done, set up`).toEqual(
        heights.get(`${language} ${width}`),
      );
    }
  }
  await app.close();
});

test("the card keeps one height through every step with the sidebar dragged to its widest, in English and Chinese (#1195)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await setWindowSize(app, window, 900, 900);
  const card = window.getByTestId("first-open-steps");
  await expect(card).toBeVisible();
  const widthOf = async () => {
    const box = await card.boundingBox();
    if (!box) throw new Error("The card isn't laid out.");
    return box.width;
  };
  const usual = await widthOf();

  await setSidebarWidth(window, 480);
  for (const language of ["en", "zh-CN"] as const) {
    await setLanguage(window, language);
    expect(await widthOf()).toBeLessThan(usual);
    await expectOneHeight(window, `${language} with the sidebar at 480 px`);
  }
  await app.close();
});

test("asking the ready Question opens the assistant, and the page stays still as the steps tick before and after, at the narrowest window and at 1,100 px (#1195)", async () => {
  for (const width of [900, 1100]) {
    const folder = await createDataFolder();
    const { app, window } = await launchApp(folder, { examples: true, fakeChat: true });
    await setWindowSize(app, window, width, 900);
    await expect(window.getByTestId("first-open-steps")).toBeVisible();
    const before = await measure(window);

    await pointAndClick(window, window.getByTestId("claim-mark"));
    await expect(window.getByTestId("claim-card")).toBeVisible();
    await window.keyboard.press("Escape");
    await expect(window.getByTestId("first-open-step").first()).toHaveAttribute(
      "data-state",
      "done",
    );
    expect(await measure(window), `${width} px, step 1 done`).toEqual(before);

    await pointAndClick(window, window.getByTestId("first-open-ready"));
    await expect(window.getByTestId("assistant")).toBeVisible();
    await expect(window.getByTestId("first-open-add-files")).toBeVisible();
    // The page is narrower now, so the card is as tall as its tallest state at this width.
    const beside = await measure(window);

    await markSteps(window, { indexed: true });
    await expect(window.getByTestId("first-open-choose-model")).toBeVisible();
    expect(await measure(window), `${width} px, assistant open, all done`).toEqual(beside);
    await app.close();
    await removeDataFolder(folder);
  }
});

test("the Welcome card has no subtitle and no footer; its Examples row's ⋯ offers the other examples and removing them, which Undo takes back (#1195)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true });
  await setWindowSize(app, window, 1440, 900);
  const card = window.getByTestId("first-open-steps");
  await expect(card).toBeVisible();
  await expect(card.getByTestId("example-more")).toHaveCount(0);
  await expect(card.getByTestId("first-open-remove")).toHaveCount(0);

  const row = window.getByTestId("sidebar").getByTestId("not-in-a-folder");
  await row.hover();
  await pointAndClick(window, row.getByTestId("examples-menu"));
  const menu = row.getByTestId("examples-actions");
  await expect(menu).toBeVisible();
  const groups = menu.getByTestId("example-more-link");
  await expect(groups).toHaveCount(3);
  await pointAndClick(window, groups.first());
  await expect(window.getByTestId("mind-list-item")).toHaveCount(2);

  await row.hover();
  await pointAndClick(window, row.getByTestId("examples-menu"));
  await pointAndClick(window, menu.getByTestId("examples-remove"));
  await expect(window.getByTestId("mind-list-item")).toHaveCount(0);
  await expect(window.getByTestId("undo-line").getByTestId("undo-message")).toHaveText(
    "Removed the examples",
  );
  await window.keyboard.press("ControlOrMeta+z");
  await expect(window.getByTestId("mind-list-item")).toHaveCount(2);
  await app.close();
});

const WORDS = {
  en: { more: "Try an example closer to your work:", remove: "Remove the examples" },
  "zh-CN": { more: "换一个和你工作更相关的示例：", remove: "移除示例" },
} as const;

for (const language of ["en", "zh-CN"] as const) {
  test(`an example Mind's page ⋯ offers the other examples and removing them, after an own Mind has been made, with Undo (${language}) (#1195)`, async () => {
    const { app, window } = await launchApp(dataDir, { examples: true });
    await setWindowSize(app, window, 900, 900);
    await setLanguage(window, language);
    await dismissChatSetup(window);
    const examples = window
      .getByTestId("mind-list-item")
      .filter({ has: window.getByTestId("example-chip") });

    await newMind(window);
    const own = await openPageMenu(window);
    await expect(own.getByTestId("examples-remove")).toHaveCount(0);
    await expect(own.getByTestId("example-more-link")).toHaveCount(0);
    await window.keyboard.press("Escape");

    await pointAndClick(window, examples);
    const menu = await openPageMenu(window);
    await expect(menu.getByTestId("example-more-link")).toHaveCount(3);
    await expect(menu).toContainText(WORDS[language].more);
    const remove = menu.getByTestId("examples-remove");
    await expect(remove).toHaveText(WORDS[language].remove);
    const rule = remove.locator("xpath=preceding-sibling::*[1]");
    await expect(rule).toBeEmpty();
    await expect(rule.locator("xpath=preceding-sibling::*[1]")).toHaveAttribute(
      "data-testid",
      "example-more-link",
    );
    const [groupsAt, removeAt, deleteAt] = await Promise.all([
      menu.getByTestId("example-more-link").last().boundingBox(),
      remove.boundingBox(),
      menu.getByTestId("mind.delete").boundingBox(),
    ]);
    expect(groupsAt && removeAt && deleteAt && groupsAt.y < removeAt.y).toBe(true);
    expect(removeAt && deleteAt && removeAt.y < deleteAt.y).toBe(true);

    await pointAndClick(window, menu.getByTestId("example-more-link").first());
    await expect(examples).toHaveCount(2);
    const again = await openPageMenu(window);
    await pointAndClick(window, again.getByTestId("examples-remove"));
    await expect(examples).toHaveCount(0);
    await expect(window.getByTestId("undo-line").getByTestId("undo-message")).toHaveText(
      language === "en" ? "Removed the examples" : "已移除示例",
    );
    await window.keyboard.press("ControlOrMeta+z");
    await expect(examples).toHaveCount(2);
    await app.close();
  });
}

/** The system's file picker, closed without a file; `pickersShown` counts how often it was opened. */
async function cancelFilePicker(app: ElectronApplication): Promise<void> {
  await app.evaluate(({ dialog }) => {
    const shown = { count: 0 };
    (globalThis as { pickersShown?: { count: number } }).pickersShown = shown;
    dialog.showOpenDialog = (async () => {
      shown.count += 1;
      return { canceled: true, filePaths: [] };
    }) as unknown as typeof dialog.showOpenDialog;
  });
}

const pickersShown = (app: ElectronApplication) =>
  app.evaluate(() => (globalThis as { pickersShown?: { count: number } }).pickersShown?.count ?? 0);

const TITLES = {
  en: { question: "Ask a question", files: "Add your own files" },
  "zh-CN": { question: "提一个问题", files: "添加你自己的文件" },
} as const;

test("Ask a question and Add your own files do their step on a click, Enter or Space at any time, and doing it alone ticks nothing (#1489)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await setWindowSize(app, window, 900, 900);
  await cancelFilePicker(app);
  await expect(window.getByTestId("first-open-steps")).toBeVisible();

  for (const language of ["en", "zh-CN"] as const) {
    await setLanguage(window, language);
    const question = step(window, "question");
    const files = step(window, "files");
    await expect(question).toHaveAttribute("data-state", "later");
    await expect(files).toHaveAttribute("data-state", "later");
    const ask = question.getByTestId("first-open-step-do");
    const add = files.getByTestId("first-open-step-do");
    await expect(ask).toHaveAccessibleName(TITLES[language].question);
    await expect(add).toHaveAccessibleName(TITLES[language].files);

    const shown = await pickersShown(app);
    await pointAndClick(window, add);
    await expect.poll(() => pickersShown(app)).toBe(shown + 1);
    await add.focus();
    await window.keyboard.press("Enter");
    await expect.poll(() => pickersShown(app)).toBe(shown + 2);
    await window.keyboard.press("Space");
    await expect.poll(() => pickersShown(app)).toBe(shown + 3);

    await ask.focus();
    await window.keyboard.press("Enter");
    await expect(window.getByTestId("assistant")).toBeVisible();
    await expect(window.getByTestId("composer-input")).toBeFocused();
    await ask.focus();
    await expect(window.getByTestId("composer-input")).not.toBeFocused();
    await window.keyboard.press("Space");
    await expect(window.getByTestId("composer-input")).toBeFocused();
    await pointAndClick(window, ask);
    await expect(window.getByTestId("composer-input")).toBeFocused();

    await expect(window.locator('[data-testid="first-open-step"][data-state="done"]')).toHaveCount(
      0,
    );
    expect(await gettingStarted(window)).toMatchObject({
      mistakeSeen: false,
      askedOwn: false,
      askedReady: false,
      indexed: false,
    });
  }
  await app.close();
});

for (const language of ["en", "zh-CN"] as const) {
  test(`Ask a question ticks when the Answer arrives, never when the Question is only asked or its Answer fails, in ${language} (#1489)`, async () => {
    const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
    await setWindowSize(app, window, 1440, 900);
    await setLanguage(window, language);
    await useLocalChatModel(window);
    await markSteps(window, { mistakeSeen: true });
    const question = step(window, "question");
    await expect(question).toHaveAttribute("data-state", "open");

    await pointAndClick(window, question.getByTestId("first-open-step-do"));
    await expect(window.getByTestId("composer-input")).toBeFocused();
    await expect(question).toHaveAttribute("data-state", "open");
    await window.keyboard.type("Send a blank reply");
    await window.keyboard.press("Enter");
    await expect(assistantAnswers(window).first()).toHaveAttribute("data-status", "failed", {
      timeout: 15_000,
    });
    await expect(question).toHaveAttribute("data-state", "open");
    expect(await gettingStarted(window)).toMatchObject({ askedOwn: false, askedReady: false });

    await askInAssistant(window, "When are the tide tables published?");
    await expect(question).toHaveAttribute("data-state", "done");
    await expect(question).toContainText(language === "en" ? "Asked a question" : "已提过问题");
    expect(await gettingStarted(window)).toMatchObject({ askedOwn: true });
    await app.close();
  });
}

const TIDES = buildPdf([
  { lines: ["Harbour times", "Harbours publish the times of high water every 12 months."] },
]);

for (const language of ["en", "zh-CN"] as const) {
  test(`Saw a mistake it caught ticks only when the example's red ≠ is opened, not an ordinary Citation, in ${language} (#1489)`, async () => {
    const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
    await setWindowSize(app, window, 1440, 900);
    await setLanguage(window, language);
    const mind = window.getByTestId("mind-pane");
    const chip = mind.getByTestId("mind-editor").getByTestId("citation-chip").first();
    await expect(chip).toHaveAttribute("data-check", "found", { timeout: 20_000 });
    const mark = mind.getByTestId("claim-mark");
    await expect(mark).toHaveAttribute("data-check", "contradicted");

    await pointAndClick(window, chip);
    await expect(window.getByTestId("citation-card")).toBeVisible();
    await window.keyboard.press("Escape");
    expect(await gettingStarted(window)).toMatchObject({ mistakeSeen: false });
    await expect(step(window, "citation")).toHaveAttribute("data-state", "open");
    await expect(mark).toHaveAttribute("data-guided");

    await pointAndClick(window, mark);
    await expect(window.getByTestId("claim-card")).toBeVisible();
    await expect(step(window, "citation")).toHaveAttribute("data-state", "done");
    expect(await gettingStarted(window)).toMatchObject({ mistakeSeen: true });
    await app.close();
  });
}

test("a red ≠ opened in a Mind of the User's own ticks Saw a mistake it caught too (#1489)", async () => {
  test.setTimeout(150_000);
  const sources = await createDataFolder();
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { app, window } = await launchApp(dataDir, {
    examples: true,
    fakeChat: true,
    home: sources,
  });
  await setWindowSize(app, window, 1440, 900);
  await useLocalChatModel(window);
  await window.getByTestId("add-documents-input").setInputFiles([join(sources, "Tides.pdf")]);
  await expectReadyDocuments(window, 3);
  await newMind(window);
  await nameMind(window, "Memo");
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("Harbours publish the times of high water every 6 months.");
  const line = editor.locator(":scope > p").first();
  await slideSelect(window, line, "Harbours", { line, words: "months." });
  await pointAndClick(
    window,
    window.getByTestId("selection-menu").getByTestId("selection-action-find-support"),
  );
  const own = window.getByTestId("claim-mark");
  await expect(own).toHaveAttribute("data-check", "contradicted");
  expect(await gettingStarted(window)).toMatchObject({ mistakeSeen: false });
  await pointAndClick(window, own);
  await expect(window.getByTestId("claim-card")).toBeVisible();
  await expect.poll(() => gettingStarted(window)).toMatchObject({ mistakeSeen: true });
  await app.close();
  await removeDataFolder(sources);
});
