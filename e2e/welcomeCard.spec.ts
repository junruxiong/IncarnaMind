import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  newMind,
  openPageMenu,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setSidebarWidth,
  setWindowSize,
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

const NONE = { citationChecked: false, askedOwn: false, askedReady: false, indexed: false };

const STATES = [
  { name: "Welcome", done: NONE, steps: ["open", "later", "later"] },
  {
    name: "first step done",
    done: { ...NONE, citationChecked: true },
    steps: ["done", "open", "later"],
  },
  {
    name: "second step done",
    done: { ...NONE, citationChecked: true, askedReady: true },
    steps: ["done", "done", "open"],
  },
  {
    name: "all done, no model",
    done: { ...NONE, citationChecked: true, askedReady: true, indexed: true },
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

test("a file that couldn't be read says so in step 3 without changing the card's height or moving its buttons, at every width in English and Chinese (#1457)", async () => {
  const sources = await createDataFolder();
  const damaged = join(sources, "damaged.pdf");
  await writeFile(damaged, "%PDF-1.4 this is not a PDF");
  const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  try {
    await markSteps(window, { ...NONE, citationChecked: true, askedReady: true });
    const files = window.locator('[data-testid="first-open-step"][data-step="files"]');
    await expect(files).toHaveAttribute("data-state", "open");
    const add = files.getByTestId("first-open-add-files");
    const places = async () => ({
      ...(await measure(window)),
      buttons: Math.round((await add.boundingBox())?.y ?? -1),
    });
    const views = (["en", "zh-CN"] as const).flatMap((language) =>
      [900, 1100, 1440].map((width) => ({ language, width, name: `${language} ${width}` })),
    );
    const before: Record<string, Awaited<ReturnType<typeof places>>> = {};
    for (const { language, width, name } of views) {
      await setLanguage(window, language);
      await setWindowSize(app, window, width, 900);
      before[name] = await places();
    }

    await interceptOpenDialog(app, damaged);
    await pointAndClick(window, add);
    const unread = files.getByTestId("first-open-unread");
    await expect(unread).toBeVisible({ timeout: 30_000 });
    for (const { language, width, name } of views) {
      await setLanguage(window, language);
      await setWindowSize(app, window, width, 900);
      await expect(unread).toBeVisible();
      expect(await places(), `${name}, the line shown`).toEqual(before[name]);
    }

    // Show is the User's own click; one file's row still fits in the card.
    await pointAndClick(window, unread.getByTestId("first-open-unread-show"));
    await expect(unread.getByTestId("first-open-unread-list")).toBeVisible();
    const open = await measure(window);
    expect(open.card, "the list of one file, in the card's height").toBe(
      before["zh-CN 1440"]?.card,
    );
    expect(open.title).toBe(before["zh-CN 1440"]?.title);
  } finally {
    await app.close();
    await removeDataFolder(sources);
  }
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
