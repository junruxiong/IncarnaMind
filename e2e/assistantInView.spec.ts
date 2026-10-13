import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  askInAssistant,
  assistantAnswers,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  removeDataFolder,
  setWindowSize,
  startScriptedTask,
} from "./app";

/**
 * The latest Answer stays in view in the assistant (#612): a conversation
 * read to its end is still read to its end after the assistant docks at the
 * bottom, floats, folds and unfolds, and after the window narrows to 900px,
 * so the latest Answer's last lines and its Copy · Add to note are never left
 * under the composer; a Task waiting in its tray, which shortens the thread, is
 * kept up with too. A conversation
 * scrolled up keeps the turn at its top where it was. The checks are
 * relations to the thread's box. Set INCARNAMIND_SCREENSHOTS to a folder to
 * also save screenshots.
 */
const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

let dataDir: string;
let folder: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  folder = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(folder);
});

type Place = "bottom" | "float" | "right";

async function placeAs(window: Page, place: Place) {
  await window.getByTestId("assistant-more").click();
  await window.getByTestId(`assistant-menu-${place}`).click();
  await expect(window.getByTestId("assistant")).toHaveAttribute("data-dock", place);
}

/** Whether a box lies inside the thread's own, with a pixel to spare. */
const inThread = (window: Page, target: Locator) =>
  target.evaluate((el, thread) => {
    const box = el.getBoundingClientRect();
    const view = (
      document.querySelector(`[data-testid="${thread}"]`) as HTMLElement
    ).getBoundingClientRect();
    return box.top >= view.top - 1 && box.bottom <= view.bottom + 1;
  }, "assistant-thread");

/** How far a turn's top is below the thread's top. */
const offsetIn = (target: Locator) =>
  target.evaluate((el) => {
    const thread = document.querySelector('[data-testid="assistant-thread"]') as HTMLElement;
    return el.getBoundingClientRect().top - thread.getBoundingClientRect().top;
  });

async function start({ task }: { task: boolean }) {
  const { app, window } = await launchApp(dataDir, {
    fakeChat: true,
    scriptedTasks: true,
    home: folder,
  });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await connectLocalChatModel(window);
  await newMind(window);
  await window.getByTestId("mind-title").fill("Credit memo");
  for (const question of ["First", "Second", "Third", "Fourth", "Fifth"])
    await askInAssistant(window, question);
  if (task) {
    const mind = await window.evaluate(
      async () =>
        (await (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listMinds()).find(
          (each) => each.title === "Credit memo",
        )?.id ?? "",
    );
    await startScriptedTask(app, mind, { folder, files: ["a.txt"] });
    await expect(window.getByTestId("task-tray")).toBeVisible();
  }
  return { app, window };
}

test("at the end of a conversation, docking at the bottom, floating, folding and unfolding it and narrowing the window leave the latest Answer's actions in view", async () => {
  test.setTimeout(150_000);
  const { app, window } = await start({ task: false });
  const latest = assistantAnswers(window).last();
  const actions = latest.getByTestId("answer-copy");
  await expect.poll(() => inThread(window, actions)).toBe(true);

  for (const place of ["bottom", "float", "right"] as const) {
    await placeAs(window, place);
    await expect.poll(() => inThread(window, actions), { message: `at ${place}` }).toBe(true);
    if (SCREENSHOTS) {
      await window
        .getByTestId("assistant")
        .screenshot({ path: join(SCREENSHOTS, `inview-${place}.png`) });
    }
  }

  // Folded and unfolded, on the right.
  await window.getByTestId("assistant-fold").click();
  await expect(window.getByTestId("assistant")).toHaveAttribute("data-folded", "true");
  await window.getByTestId("assistant-fold").click();
  await expect(window.getByTestId("assistant")).not.toHaveAttribute("data-folded");
  await expect.poll(() => inThread(window, actions), { message: "unfolded" }).toBe(true);

  // The window narrowed to 900px, where a side dock goes to the bottom.
  await setWindowSize(app, window, 900, 700);
  await expect.poll(() => inThread(window, actions), { message: "at 900px" }).toBe(true);
  await app.close();
});

test("scrolled up to an older turn, the same turn stays in view after the same changes", async () => {
  test.setTimeout(150_000);
  const { app, window } = await start({ task: false });
  const thread = window.getByTestId("assistant-thread");
  const first = window.locator(".assistant-question").first();
  // Scrolled up until the first Question is at the thread's top.
  await thread.evaluate((el) => {
    el.scrollTop = 0;
  });
  await expect.poll(() => offsetIn(first)).toBeLessThan(40);
  await thread.evaluate((el) => {
    el.dispatchEvent(new Event("scroll"));
  });
  const before = await offsetIn(first);

  for (const place of ["bottom", "float", "right"] as const) {
    await placeAs(window, place);
    await expect.poll(() => inThread(window, first), { message: `at ${place}` }).toBe(true);
    expect(Math.abs((await offsetIn(first)) - before), `at ${place}`).toBeLessThanOrEqual(2);
  }
  await setWindowSize(app, window, 900, 700);
  await expect.poll(() => inThread(window, first), { message: "at 900px" }).toBe(true);
  expect(Math.abs((await offsetIn(first)) - before), "at 900px").toBeLessThanOrEqual(2);
  await app.close();
});

test("a Task starting in the tray, which shortens the thread, leaves the latest Answer's actions in view, on the right and floating", async () => {
  test.setTimeout(150_000);
  const { window } = await start({ task: true });
  const actions = assistantAnswers(window).last().getByTestId("answer-copy");
  await expect.poll(() => inThread(window, actions), { message: "the tray shown" }).toBe(true);
  for (const place of ["float", "right"] as const) {
    await placeAs(window, place);
    await expect.poll(() => inThread(window, actions), { message: `at ${place}` }).toBe(true);
  }
  if (SCREENSHOTS) {
    await window
      .getByTestId("assistant")
      .screenshot({ path: join(SCREENSHOTS, "inview-task.png") });
  }
});
