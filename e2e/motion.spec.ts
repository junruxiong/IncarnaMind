import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import type { TestHooks } from "../src/shared/testHooks";
import {
  clickNow,
  closeSettings,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  motionSettled,
  openSettings,
  removeDataFolder,
  setMotionSpeed,
  setWindowSize,
} from "./app";

/*
 * Motion (DESIGN.md, Motion; src/renderer/src/motion.ts): a menu grows from
 * its button and shrinks back toward it, the Citation card from its check;
 * Reduced only fades, Off is instant, and nothing waits for an animation.
 * Where a test looks, or acts, during one, the animations run slowed down, so
 * it never races them; then it waits for them to settle, not for a time.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

/** How an element shows now: its opacity, its transform's scale and shift, its origin, and where it's laid out. */
function look(target: Locator) {
  return target.evaluate((element) => {
    const style = getComputedStyle(element);
    const matrix = new DOMMatrixReadOnly(style.transform === "none" ? undefined : style.transform);
    const [originX = "0", originY = "0"] = style.transformOrigin.split(" ");
    // Its box as laid out, without the transform.
    const box = element.getBoundingClientRect();
    const left = box.left - matrix.e - (1 - matrix.a) * Number.parseFloat(originX);
    const top = box.top - matrix.f - (1 - matrix.d) * Number.parseFloat(originY);
    return {
      opacity: Number(style.opacity),
      scale: matrix.a,
      y: matrix.f,
      originX: Number.parseFloat(originX),
      originY: Number.parseFloat(originY),
      left,
      top,
    };
  });
}

/** The copies of closed menus still fading where they were, which nothing else can see. */
const leavingCopies = (window: Page) =>
  window.evaluate(
    () =>
      (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks?.leavingCopies() ??
      [],
  );

const running = (window: Page) =>
  window.evaluate(
    () =>
      (globalThis as { incarnamindTestHooks?: TestHooks }).incarnamindTestHooks?.motionRunning() ??
      0,
  );

/** The scale a copy's transform says, 1 with none. */
const scaleOf = (transform: string) => Number(/^matrix\(([^,]+),/.exec(transform)?.[1] ?? 1);

async function chooseLanguage(window: Page, language: "en" | "zh-CN"): Promise<void> {
  await window.evaluate(async (chosen) => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.updateSettings({ user: { language: chosen } });
  }, language);
}

/** A first run, past the chat setup, in a window of a set size, with the system's motion as given. */
async function start(
  { width = 1100, reducedMotion = "no-preference" as "no-preference" | "reduce" } = {},
  options: Parameters<typeof launchApp>[1] = {},
) {
  const running = await launchApp(dataDir, options);
  await setWindowSize(running.app, running.window, width, 660);
  await running.window.emulateMedia({ reducedMotion });
  if (!options.examples) await dismissChatSetup(running.window);
  return running;
}

for (const language of ["en", "zh-CN"] as const) {
  test(`with motion on, a menu grows from its button and shrinks back toward it (${language})`, async () => {
    const { window } = await start({ width: language === "en" ? 1100 : 900 });
    await chooseLanguage(window, language);
    await expect(window.locator("html")).toHaveAttribute("data-motion", "on");
    await setMotionSpeed(window, 0.2);
    const button = window.getByTestId("plus-menu");
    const menu = window.getByTestId("plus-menu-items");
    const from = await button.boundingBox();
    if (!from) throw new Error("The + button isn't on screen.");

    await button.click();
    // Opening: under the button, smaller and faint, growing from its top edge at the button's middle.
    const opening = await look(menu);
    expect(opening.opacity).toBeLessThan(1);
    expect(opening.scale).toBeLessThan(1);
    expect(opening.scale).toBeGreaterThanOrEqual(0.96);
    expect(opening.y).toBeLessThanOrEqual(0);
    expect(opening.originY).toBe(0);
    expect(Math.abs(opening.left + opening.originX - (from.x + from.width / 2))).toBeLessThan(1.5);
    expect(opening.top).toBeGreaterThanOrEqual(from.y + from.height);
    await expect(menu.getByRole("menuitem").first()).toHaveText(
      language === "en" ? /New Mind/ : /新建 Mind/,
    );

    await motionSettled(window);
    expect(await look(menu)).toMatchObject({ opacity: 1, scale: 1, y: 0 });

    // Closed: gone at once; a copy that nothing can click goes back toward the button as it fades,
    // from the button's side.
    await window.keyboard.press("Escape");
    await expect(menu).toBeHidden();
    await expect
      .poll(async () => {
        const [copy] = await leavingCopies(window);
        return copy ? copy.opacity < 1 && scaleOf(copy.transform) < 1 : false;
      })
      .toBe(true);
    const [leaving] = await leavingCopies(window);
    expect(leaving?.origin).toMatch(/ 0px$/);
    expect(leaving?.box.top).toBeGreaterThanOrEqual(from.y + from.height);
    await motionSettled(window);
    expect(await leavingCopies(window)).toEqual([]);
  });
}

test("a click on an item while its menu is still opening acts on it", async () => {
  const { window } = await start();
  await setMotionSpeed(window, 0.1);
  const minds = window.getByTestId("mind-list-item");
  const before = await minds.count();

  await window.getByTestId("plus-menu").click();
  await window.evaluate(() => {
    // How far the menu had come in when the pointer pressed on its item.
    document.addEventListener(
      "pointerdown",
      () => {
        const menu = document.querySelector('[data-testid="plus-menu-items"]');
        (globalThis as { pressedAt?: number }).pressedAt = menu
          ? Number(getComputedStyle(menu).opacity)
          : -1;
      },
      { capture: true, once: true },
    );
  });
  await clickNow(window, window.getByTestId("plus-new-mind"));

  const pressedAt = await window.evaluate(() => (globalThis as { pressedAt?: number }).pressedAt);
  expect(pressedAt).toBeGreaterThanOrEqual(0);
  expect(pressedAt).toBeLessThan(1);
  await expect(minds).toHaveCount(before + 1);
  await expect(window.getByTestId("plus-menu-items")).toBeHidden();
});

test("Reduced, chosen in Settings, fades a menu without moving it; Off shows and hides it at once", async () => {
  const { window } = await start();
  const settings = await openSettings(window, "general");
  const motion = settings.getByTestId("motion-settings");
  await expect(motion.getByRole("heading")).toHaveText("Motion");
  await expect(motion.getByRole("radio")).toHaveCount(3);
  await expect(motion.getByRole("radio", { name: "On" })).toBeChecked();
  await expect(motion).toContainText("Until you choose, this follows your system's setting.");

  await motion.getByRole("radio", { name: "Reduced" }).check();
  await expect(window.locator("html")).toHaveAttribute("data-motion", "reduced");
  await expect(motion).not.toContainText("Until you choose");
  await closeSettings(window);

  await setMotionSpeed(window, 0.2);
  const button = window.getByTestId("plus-menu");
  const menu = window.getByTestId("plus-menu-items");
  await button.click();
  const opening = await look(menu);
  expect(opening.opacity).toBeLessThan(1);
  expect(opening).toMatchObject({ scale: 1, y: 0 });
  await motionSettled(window);
  expect((await look(menu)).opacity).toBe(1);
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await expect
    .poll(async () => {
      const [copy] = await leavingCopies(window);
      return copy ? copy.opacity < 1 && scaleOf(copy.transform) === 1 : false;
    })
    .toBe(true);
  await motionSettled(window);

  await openSettings(window, "general");
  await motion.getByRole("radio", { name: "Off" }).check();
  await expect(window.locator("html")).toHaveAttribute("data-motion", "off");
  await closeSettings(window);
  await button.click();
  expect(await look(menu)).toMatchObject({ opacity: 1, scale: 1, y: 0 });
  expect(await running(window)).toBe(0);
  await window.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  expect(await leavingCopies(window)).toEqual([]);
  expect(await running(window)).toBe(0);

  // Kept on this device.
  const kept = await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    return (await bridge.getSettings()).device.motion;
  });
  expect(kept).toBe("off");
});

test("until the User chooses, motion follows the system's Reduce motion, in English and Chinese; a choice holds over it", async () => {
  const { window } = await start({ width: 900, reducedMotion: "reduce" });
  const html = window.locator("html");
  await expect(html).toHaveAttribute("data-motion", "reduced");
  const settings = await openSettings(window, "general");
  const motion = settings.getByTestId("motion-settings");
  await expect(motion.getByRole("radio", { name: "Reduced" })).toBeChecked();
  await window.emulateMedia({ reducedMotion: "no-preference" });
  await expect(html).toHaveAttribute("data-motion", "on");
  await expect(motion.getByRole("radio", { name: "On" })).toBeChecked();

  await chooseLanguage(window, "zh-CN");
  await expect(motion.getByRole("heading")).toHaveText("动效");
  await expect(motion.getByRole("radio")).toHaveCount(3);
  await expect(motion.getByRole("radio", { name: "开" })).toBeChecked();
  await expect(motion.getByRole("radio", { name: "减弱" })).not.toBeChecked();
  await expect(motion.getByRole("radio", { name: "关" })).not.toBeChecked();
  await expect(motion).toContainText("在你选择之前，跟随系统设置。");

  // The one the system picked, chosen as it is: kept, so the system no longer decides.
  await motion.getByRole("radio", { name: "开" }).click();
  await expect(motion).not.toContainText("在你选择之前");
  await window.emulateMedia({ reducedMotion: "reduce" });
  await expect(html).toHaveAttribute("data-motion", "on");
  await expect(motion.getByRole("radio", { name: "开" })).toBeChecked();
});

test("the Citation card grows from its Citation and goes back toward it", async () => {
  const { window } = await start({ width: 1280 }, { examples: true });
  const pane = window.getByTestId("mind-pane");
  const citation = pane.getByTestId("citation-chip").first();
  await expect(pane.getByTestId("citation").first()).toHaveAttribute("data-check", "found", {
    timeout: 20_000,
  });
  await motionSettled(window);
  await setMotionSpeed(window, 0.2);

  await citation.click();
  const card = window.getByTestId("citation-card");
  await expect(card).toBeVisible();
  const opening = await look(card);
  expect(opening.opacity).toBeLessThan(1);
  expect(opening.scale).toBeLessThan(1);
  await motionSettled(window);
  expect(await look(card)).toMatchObject({ opacity: 1, scale: 1, y: 0 });

  await window.keyboard.press("Escape");
  await expect(card).toBeHidden();
  await expect.poll(async () => (await leavingCopies(window)).length).toBe(1);
  await motionSettled(window);
  expect(await leavingCopies(window)).toEqual([]);
});
