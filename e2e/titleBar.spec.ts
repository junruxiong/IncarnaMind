import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import { createDataFolder, dismissChatSetup, launchApp, openViewer, removeDataFolder } from "./app";

/*
 * The window's title bar (src/shared/titleBar.ts): on macOS there is no title
 * row of the system's; its traffic lights sit in the sidebar's header, and the
 * 44px band of pane headers is the title bar, dragged by its empty parts.
 * Playwright's mouse reaches the page without the system's hit test, so these
 * tests check each point's app-region (what Electron hands macOS) and that the
 * controls in the band still work. Set INCARNAMIND_SCREENSHOTS to a folder to
 * also save screenshots of the band.
 */

test.skip(process.platform !== "darwin", "The traffic lights in the band are macOS's.");

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
/** The room the sidebar's header keeps for the traffic lights (styles.css). */
const LIGHTS_ROOM = 84;
/** macOS 26's traffic lights: 14px, 9px apart (src/main/titleBar.ts). */
const LIGHT = { size: 14, gap: 9 };

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

async function boxOf(locator: Locator) {
  const box = await locator.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  return box;
}

async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  const width = await window.evaluate(() => innerWidth);
  await window.screenshot({
    path: join(SCREENSHOTS, `${name}.png`),
    clip: { x: 0, y: 0, width, height: 120 },
  });
}

/**
 * What a press at this point does to the window: "drag" moves it, "no-drag"
 * goes to the page. The innermost element there with an app-region decides,
 * as the regions Electron hands the system nest.
 */
function regionAt(window: Page, x: number, y: number): Promise<string> {
  return window.evaluate(
    (point) => {
      for (let at = document.elementFromPoint(point.x, point.y); at; at = at.parentElement) {
        const region = getComputedStyle(at).getPropertyValue("-webkit-app-region");
        if (region && region !== "none") return region;
      }
      return "none";
    },
    { x, y },
  );
}

/** The app-region at the middle of an element, after checking nothing covers it there. */
async function regionOf(window: Page, locator: Locator): Promise<string> {
  const box = await boxOf(locator);
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  const onTop = await locator.evaluate(
    (element, point) => {
      const hit = document.elementFromPoint(point.x, point.y);
      return hit !== null && element.contains(hit);
    },
    { x, y },
  );
  expect(onTop).toBe(true);
  return regionAt(window, x, y);
}

/** Slides the mouse onto an element, as a hand does, and clicks it. */
async function slideAndClick(window: Page, locator: Locator): Promise<void> {
  const box = await boxOf(locator);
  await window.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 10 });
  await window.mouse.down();
  await window.mouse.up();
}

/** Drags a tab by the mouse onto another's left or right half: before it or after it. */
async function dragTab(
  window: Page,
  tab: Locator,
  onto: Locator,
  side: "before" | "after",
): Promise<void> {
  const from = await boxOf(tab);
  const to = await boxOf(onto);
  await window.mouse.move(from.x + from.width / 2, from.y + from.height / 2, { steps: 8 });
  await window.mouse.down();
  const x = side === "before" ? to.x + 12 : to.x + to.width - 12;
  await window.mouse.move(x, to.y + to.height / 2, { steps: 12 });
  await window.mouse.up();
}

const tabsOf = (window: Page) => window.getByTestId("mind-tab");
const titlesOf = (window: Page) => window.getByTestId("mind-tab-title");
const tabNamed = (window: Page, title: string) =>
  window.locator(`[data-testid="mind-tab"][title=${JSON.stringify(title)}]`);

/** The window's frame as Electron sees it. */
function frameOf(app: ElectronApplication) {
  return app.evaluate(({ BrowserWindow }) => {
    const window = BrowserWindow.getAllWindows()[0];
    if (!window) throw new Error("No window.");
    return {
      bounds: window.getBounds(),
      content: window.getContentBounds(),
      title: window.getTitle(),
      background: window.getBackgroundColor(),
      lights: window.getWindowButtonPosition(),
      fullScreen: window.isFullScreen(),
    };
  });
}

test("no title row of the system's: the traffic lights sit in the sidebar's header, and the band drags the window while its tabs and buttons still work", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);

  // The page fills the whole window, title row and all; the window keeps its name for Mission Control.
  const frame = await frameOf(app);
  expect(frame.content).toEqual(frame.bounds);
  expect(frame.title).toBe("IncarnaMind");
  expect(frame.background.toUpperCase()).toBe("#F4F5F7");
  await expect(window.locator("html")).toHaveAttribute("data-title-bar", "inset");

  // The lights: centred in the 44px band, the first on the sidebar's icon column, all
  // inside the room the sidebar's header keeps for them, where the app's mark would be.
  expect(frame.lights).toEqual({ x: 16, y: (44 - LIGHT.size) / 2 });
  expect(16 + 3 * LIGHT.size + 2 * LIGHT.gap).toBeLessThan(LIGHTS_ROOM);
  const header = window.getByTestId("sidebar-header");
  expect(await boxOf(header)).toMatchObject({ x: 0, y: 0, height: 44 });
  expect(await header.evaluate((element) => getComputedStyle(element).paddingLeft)).toBe(
    `${LIGHTS_ROOM}px`,
  );
  await expect(header.getByTestId("app-mark")).toBeHidden();
  await expect(header).not.toContainText("IncarnaMind");

  // Three Minds, each made with "+" in the band.
  const newTab = window.getByTestId("new-tab");
  expect(await regionOf(window, newTab)).toBe("no-drag");
  for (const title of ["Alpha", "Beta", "Gamma"]) {
    const count = await tabsOf(window).count();
    await slideAndClick(window, newTab);
    await expect(tabsOf(window)).toHaveCount(count + 1);
    await window.getByTestId("mind-title").fill(title);
  }
  await expect(titlesOf(window)).toHaveText(["Alpha", "Beta", "Gamma"]);
  await screenshot(window, "title-bar-band");

  // The band's empty parts drag the window: the sidebar's header, beside and between the
  // lights too, and the strip past the tabs. Tabs, their ✕, "+" and Export don't.
  const strip = await boxOf(window.getByTestId("mind-header"));
  const plus = await boxOf(newTab);
  expect(await regionAt(window, 30, 22)).toBe("drag");
  expect(await regionAt(window, 120, 22)).toBe("drag");
  expect(await regionAt(window, plus.x + plus.width + 40, 22)).toBe("drag");
  expect(await regionAt(window, plus.x + plus.width + 40, 4)).toBe("drag");
  expect(await regionAt(window, strip.x + 40, 4)).toBe("drag");
  for (const control of [
    tabNamed(window, "Alpha"),
    tabNamed(window, "Gamma"),
    tabNamed(window, "Gamma").getByTestId("mind-tab-close"),
    window.getByTestId("export-mind"),
  ]) {
    expect(await regionOf(window, control)).toBe("no-drag");
  }
  // The sidebar's divider can still be grabbed in the band, either side of its rule.
  const rod = await boxOf(window.locator("hr.pane-divider").first());
  expect(await regionAt(window, rod.x - 3, 22)).toBe("no-drag");
  expect(await regionAt(window, rod.x + 3, 22)).toBe("no-drag");

  // A click on a tab shows its Mind; its ✕ closes it; a drag moves it.
  await slideAndClick(window, tabNamed(window, "Alpha"));
  await expect(window.locator('[role="tab"][aria-selected="true"]')).toContainText("Alpha");
  await dragTab(window, tabNamed(window, "Alpha"), tabNamed(window, "Gamma"), "after");
  await expect(titlesOf(window)).toHaveText(["Beta", "Gamma", "Alpha"]);
  await window.mouse.move(400, 400, { steps: 4 });
  await slideAndClick(window, tabNamed(window, "Beta").getByTestId("mind-tab-close"));
  await expect(titlesOf(window)).toHaveText(["Gamma", "Alpha"]);

  // However narrow the sidebar, the tabs start right of the lights' room.
  const divider = window.locator("hr.pane-divider").first();
  const at = await boxOf(divider);
  await window.mouse.move(at.x - 30, 400);
  await window.mouse.move(at.x - 3, 400, { steps: 6 });
  await window.mouse.down();
  await window.mouse.move(at.x - 400, 400, { steps: 10 });
  await window.mouse.up();
  expect((await boxOf(tabsOf(window).first())).x).toBeGreaterThanOrEqual(LIGHTS_ROOM);
  await screenshot(window, "title-bar-narrow-sidebar");

  // The viewer's toolbar carries the band on: its empty part drags, its buttons click.
  await openViewer(window);
  const toolbar = window.getByTestId("viewer-header");
  const toolbarBox = await boxOf(toolbar);
  expect(toolbarBox).toMatchObject({ y: 0, height: 44 });
  expect(await regionAt(window, toolbarBox.x + 40, 22)).toBe("drag");
  const close = window.getByTestId("viewer-close");
  expect(await regionOf(window, close)).toBe("no-drag");
  await screenshot(window, "title-bar-viewer");
  await slideAndClick(window, close);
  await expect(window.getByTestId("viewer")).toHaveCount(0);

  // So does the Library's header, opened in the Mind's place.
  await slideAndClick(window, window.getByTestId("open-library"));
  const library = window.getByTestId("library").locator("header").first();
  expect(await regionAt(window, (await boxOf(library)).x + 200, 22)).toBe("drag");
  expect(await regionOf(window, library.getByRole("button"))).toBe("no-drag");
  await app.close();
});

test("in full screen the traffic lights hide, so the sidebar's header gives their room back to the app's mark, and takes it again on leaving", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const header = window.getByTestId("sidebar-header");
  const html = window.locator("html");
  const paddingLeft = () => header.evaluate((element) => getComputedStyle(element).paddingLeft);
  expect(await paddingLeft()).toBe(`${LIGHTS_ROOM}px`);

  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.setFullScreen(true));
  await expect(html).toHaveAttribute("data-full-screen", "", { timeout: 10_000 });
  expect((await frameOf(app)).fullScreen).toBe(true);
  expect(await paddingLeft()).toBe("16px");
  const mark = header.getByTestId("app-mark");
  await expect(mark).toBeVisible();
  expect((await boxOf(mark)).x).toBe(16);
  await screenshot(window, "title-bar-full-screen");

  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.setFullScreen(false));
  await expect(html).not.toHaveAttribute("data-full-screen", { timeout: 10_000 });
  expect((await frameOf(app)).fullScreen).toBe(false);
  expect(await paddingLeft()).toBe(`${LIGHTS_ROOM}px`);
  await expect(mark).toBeHidden();
  await app.close();
});
