import { join } from "node:path";
import { type ElectronApplication, expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  openViewer,
  removeDataFolder,
  setWindowSize,
} from "./app";

/*
 * Hiding and showing the sidebar (#219): Cmd+\ (Ctrl+\), the button, View >
 * Hide Sidebar, peeking at the left edge, dragging the divider shut, narrow
 * windows, and what is remembered. Driven with a real mouse and keyboard.
 * Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const isMac = process.platform === "darwin";
const COMMAND = isMac ? "Meta" : "Control";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function screenshot(window: Page, name: string): Promise<void> {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

/** Where the sidebar is: its box in the window, or null when it is out of sight. */
async function sidebarLeft(window: Page): Promise<number> {
  const box = await window.getByTestId("sidebar").boundingBox();
  if (!box) throw new Error("The sidebar has no box.");
  return box.x;
}

/** Whether the sidebar takes room from the card: the card starts at 8px when it is hidden. */
async function cardLeft(window: Page): Promise<number> {
  const box = await window.getByTestId("card").boundingBox();
  if (!box) throw new Error("The card has no box.");
  return box.x;
}

const isHidden = (window: Page) =>
  window.getByTestId("sidebar-panel").evaluate((panel) => panel.hasAttribute("data-hidden"));

/** The View menu's item for the sidebar, as the app's menu has it. */
function viewMenuLabels(app: ElectronApplication): Promise<string[]> {
  return app.evaluate(({ Menu }) => {
    const view = Menu.getApplicationMenu()?.items.find(
      (item) => item.label === "View" || item.label === "显示",
    );
    return view?.submenu?.items.map((item) => item.label) ?? [];
  });
}

function clickViewMenu(app: ElectronApplication, label: string): Promise<void> {
  return app.evaluate(({ Menu }, wanted) => {
    const view = Menu.getApplicationMenu()?.items.find(
      (item) => item.label === "View" || item.label === "显示",
    );
    view?.submenu?.items.find((item) => item.label === wanted)?.click();
  }, label);
}

test("the shortcut, the button and the View menu hide and show the sidebar, and the User's place stays", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1280, 800);

  // A Mind with some text in its title, the focus in it.
  await window.getByTestId("new-mind").click();
  const title = window.getByTestId("mind-title");
  await title.fill("A place to keep");
  await title.click();
  await expect(title).toBeFocused();
  expect(await sidebarLeft(window)).toBe(0);
  expect(await cardLeft(window)).toBeGreaterThan(200);
  await screenshot(window, "shown");

  // The keyboard: the focus stays where it was, and the Mind keeps its state.
  await window.keyboard.press(`${COMMAND}+Backslash`);
  await expect.poll(() => isHidden(window)).toBe(true);
  await expect(title).toBeFocused();
  await expect(title).toHaveValue("A place to keep");
  await expect.poll(() => cardLeft(window)).toBe(8);
  await expect(window.getByTestId("sidebar-announcement")).toHaveText("Sidebar hidden");
  await expect.poll(() => viewMenuLabels(app)).toContain("Show Sidebar");
  await screenshot(window, "hidden");
  // Not a place the pointer can reach by accident: the sidebar is out of the window.
  expect(await sidebarLeft(window)).toBeLessThan(0);

  await window.keyboard.press(`${COMMAND}+Backslash`);
  await expect.poll(() => isHidden(window)).toBe(false);
  await expect(title).toBeFocused();
  await expect(window.getByTestId("sidebar-announcement")).toHaveText("Sidebar shown");
  expect(await cardLeft(window)).toBeGreaterThan(200);
  await expect.poll(() => viewMenuLabels(app)).toContain("Hide Sidebar");

  // The button, with the mouse: it says what it does, and keeps the keyboard's focus on itself.
  const button = window.getByTestId("sidebar-toggle");
  await expect(button).toHaveAccessibleName("Hide sidebar");
  await expect(button).toHaveAttribute("aria-expanded", "true");
  await button.hover();
  await window.mouse.down();
  await window.mouse.up();
  await expect.poll(() => isHidden(window)).toBe(true);
  await expect(button).toHaveAccessibleName("Show sidebar");
  await expect(button).toHaveAttribute("aria-expanded", "false");
  await window.mouse.move(700, 400, { steps: 6 });
  await button.click();
  await expect.poll(() => isHidden(window)).toBe(false);

  // The same button, by keyboard: Enter on it, and the focus is still on a sidebar button.
  await button.focus();
  await window.keyboard.press("Enter");
  await expect.poll(() => isHidden(window)).toBe(true);
  await expect(window.getByTestId("sidebar-toggle")).toBeFocused();
  await window.keyboard.press("Enter");
  await expect.poll(() => isHidden(window)).toBe(false);
  await expect(window.getByTestId("sidebar-toggle")).toBeFocused();

  // View > Hide Sidebar.
  await clickViewMenu(app, "Hide Sidebar");
  await expect.poll(() => isHidden(window)).toBe(true);
  await clickViewMenu(app, "Show Sidebar");
  await expect.poll(() => isHidden(window)).toBe(false);

  // The Mind is the one that was open, with its title.
  await expect(window.getByTestId("mind-title")).toHaveValue("A place to keep");
  await app.close();
});

test("a hidden sidebar peeks at the left edge, and keyboard focus shows it", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1280, 800);
  await window.keyboard.press(`${COMMAND}+Backslash`);
  await expect.poll(() => isHidden(window)).toBe(true);
  const panel = window.getByTestId("sidebar-panel");

  // The pointer slides to the left edge: the sidebar comes out over the content.
  await window.mouse.move(600, 300);
  await window.mouse.move(2, 300, { steps: 12 });
  await expect(panel).toHaveAttribute("data-open", "true");
  await expect.poll(() => sidebarLeft(window)).toBe(0);
  // Over the content: the card did not move.
  expect(await cardLeft(window)).toBe(8);
  await screenshot(window, "peek");

  // It can be used while it's out, and goes back when the pointer leaves.
  await window.mouse.move(120, 300, { steps: 4 });
  await expect(panel).toHaveAttribute("data-open", "true");
  await window.mouse.move(700, 300, { steps: 12 });
  await expect(panel).not.toHaveAttribute("data-open", "true");
  await expect.poll(() => sidebarLeft(window)).toBeLessThan(0);

  // Keyboard focus into it (Tab, as ⌘K will) shows it, and leaving the sidebar hides it.
  const newMind = window.getByTestId("new-mind");
  await newMind.focus();
  await window.keyboard.press("Shift+Tab");
  await window.keyboard.press("Tab");
  await expect(newMind).toBeFocused();
  await expect(panel).toHaveAttribute("data-open", "true");
  await window.getByTestId("mind-area").click({ position: { x: 400, y: 300 } });
  await expect(panel).not.toHaveAttribute("data-open", "true");
  await app.close();
});

test("dragging the divider past its minimum hides the sidebar, and its width and the choice survive a restart", async () => {
  const first = await launchApp(dataDir);
  let { app, window } = first;
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1280, 800);

  // Resize it: the divider between its minimum and maximum.
  const rod = window.getByTestId("sidebar-resize");
  let box = await rod.boundingBox();
  if (!box) throw new Error("No divider.");
  await window.mouse.move(box.x + box.width / 2, 400, { steps: 4 });
  await window.mouse.down();
  await window.mouse.move(box.x + 60, 400, { steps: 10 });
  await window.mouse.up();
  await expect.poll(() => window.getByTestId("sidebar").evaluate((el) => el.clientWidth)).toBe(304);
  // Below the minimum but not far past: it stops at the minimum.
  box = await rod.boundingBox();
  if (!box) throw new Error("No divider.");
  await window.mouse.move(box.x + box.width / 2, 400, { steps: 4 });
  await window.mouse.down();
  await window.mouse.move(154, 400, { steps: 10 });
  await window.mouse.up();
  await expect.poll(() => window.getByTestId("sidebar").evaluate((el) => el.clientWidth)).toBe(165);
  expect(await isHidden(window)).toBe(false);

  // Far past the minimum: it folds away.
  box = await rod.boundingBox();
  if (!box) throw new Error("No divider.");
  await window.mouse.move(box.x + box.width / 2, 400, { steps: 4 });
  await window.mouse.down();
  await window.mouse.move(40, 400, { steps: 12 });
  await window.mouse.up();
  await expect.poll(() => isHidden(window)).toBe(true);

  // Restart: hidden, and when shown again it has the width it had before it folded.
  await app.close();
  ({ app, window } = await launchApp(dataDir));
  await setWindowSize(app, window, 1280, 800);
  await expect.poll(() => isHidden(window)).toBe(true);
  await expect.poll(() => cardLeft(window)).toBe(8);
  await window.keyboard.press(`${COMMAND}+Backslash`);
  await expect.poll(() => isHidden(window)).toBe(false);
  expect(await window.getByTestId("sidebar").evaluate((el) => el.clientWidth)).toBe(165);

  // Shown, and kept shown after another restart.
  await app.close();
  ({ app, window } = await launchApp(dataDir));
  expect(await isHidden(window)).toBe(false);
  await app.close();
});

test("a narrow window with the viewer open hides the sidebar on its own, and widening brings it back unless the User chose", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1400, 800);
  await openViewer(window);
  expect(await isHidden(window)).toBe(false);

  await setWindowSize(app, window, 900, 800);
  await expect.poll(() => isHidden(window)).toBe(true);
  await expect(window.getByTestId("viewer")).toBeVisible();
  await screenshot(window, "narrow-viewer");
  await setWindowSize(app, window, 1400, 800);
  await expect.poll(() => isHidden(window)).toBe(false);

  // The User shows it in the narrow window: it stays while the window is narrow.
  await setWindowSize(app, window, 900, 800);
  await expect.poll(() => isHidden(window)).toBe(true);
  await window.getByTestId("sidebar-toggle").click();
  await expect.poll(() => isHidden(window)).toBe(false);
  await setWindowSize(app, window, 940, 800);
  expect(await isHidden(window)).toBe(false);
  // The choice was not remembered as "hidden": after the narrow spell it is shown.
  await setWindowSize(app, window, 1400, 800);
  expect(await isHidden(window)).toBe(false);

  // The User hides it in a wide window: widening or narrowing never brings it back.
  await window.keyboard.press(`${COMMAND}+Backslash`);
  await expect.poll(() => isHidden(window)).toBe(true);
  await setWindowSize(app, window, 900, 800);
  await setWindowSize(app, window, 1400, 800);
  expect(await isHidden(window)).toBe(true);
  await app.close();
});

test("the button, its announcement and the View menu are in Chinese too", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await setWindowSize(app, window, 1280, 800);
  await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.updateSettings({ user: { language: "zh-CN" } });
  });
  await expect(window.getByTestId("sidebar-toggle")).toHaveAccessibleName("隐藏侧边栏");
  await expect.poll(() => viewMenuLabels(app)).toContain("隐藏侧边栏");
  await clickViewMenu(app, "隐藏侧边栏");
  await expect.poll(() => isHidden(window)).toBe(true);
  await expect(window.getByTestId("sidebar-announcement")).toHaveText("侧边栏已隐藏");
  await expect(window.getByTestId("sidebar-toggle")).toHaveAccessibleName("显示侧边栏");
  await expect.poll(() => viewMenuLabels(app)).toContain("显示侧边栏");
  await screenshot(window, "hidden-zh");
  await app.close();
});
