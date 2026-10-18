import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  launchApp,
  pointAndClick,
  removeDataFolder,
  setLanguage,
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

/** The card's height and where the Mind's title starts, which the card pushes down. */
async function measure(window: Page) {
  const card = await window.getByTestId("first-open-steps").boundingBox();
  const title = await window.getByTestId("mind-editor").locator("h1").boundingBox();
  if (!card || !title) throw new Error("The first screen isn't laid out.");
  return { card: Math.round(card.height), title: Math.round(title.y) };
}

test("the Welcome card keeps one height through every step, so the page's text never moves (#1195)", async () => {
  const { app, window } = await launchApp(dataDir, { examples: true, fakeChat: true });
  await expect(window.getByTestId("first-open-steps")).toBeVisible();

  const states: { name: string; done: Record<string, boolean> }[] = [
    { name: "Welcome", done: NONE },
    { name: "first step done", done: { ...NONE, citationChecked: true } },
    { name: "second step done", done: { ...NONE, citationChecked: true, askedReady: true } },
    {
      name: "all done, no model",
      done: { citationChecked: true, askedReady: true, indexed: true },
    },
  ];
  const seen: string[] = [];
  const check = async (name: string, width: number, language: string) => {
    const at = await measure(window);
    seen.push(`${language} ${width} ${name}: card ${at.card}, title ${at.title}`);
    return at;
  };
  let reference: Awaited<ReturnType<typeof measure>> | undefined;
  for (const language of ["en", "zh-CN"] as const) {
    await setLanguage(window, language);
    for (const width of [1100, 1440]) {
      await setWindowSize(app, window, width, 900);
      for (const { name, done } of states) {
        await markSteps(window, done);
        await expect(window.getByTestId("first-open-step")).toHaveCount(3);
        const at = await check(name, width, language);
        reference ??= at;
        expect(at, seen.join("\n")).toEqual(reference);
      }
    }
  }

  await useLocalChatModel(window);
  await expect(window.getByTestId("first-open-next").first()).toBeVisible();
  for (const language of ["en", "zh-CN"] as const) {
    await setLanguage(window, language);
    for (const width of [1100, 1440]) {
      await setWindowSize(app, window, width, 900);
      const at = await check("all done, set up", width, language);
      expect(at, seen.join("\n")).toEqual(reference);
    }
  }
  await app.close();
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
