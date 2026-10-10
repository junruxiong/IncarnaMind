import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind as newMindFromMenu,
  openViewer,
  removeDataFolder,
} from "./app";

/*
 * The Mind tabs (DESIGN.md, Components: Mind tabs): several Minds open at
 * once, as in a browser. Set INCARNAMIND_SCREENSHOTS to a folder to also save
 * a screenshot of the strip.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

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

const tabsOf = (window: Page) => window.getByTestId("mind-tab");
const titlesOf = (window: Page) => window.getByTestId("mind-tab-title");
const activeTab = (window: Page) => window.locator('[role="tab"][aria-selected="true"]');
/** The tab of the Mind with this title (a tab's tooltip is its full title). */
const tabNamed = (window: Page, title: string) =>
  window.locator(`[data-testid="mind-tab"][title=${JSON.stringify(title)}]`);
const sidebarMind = (window: Page, title: string) =>
  window.getByTestId("mind-list-item").filter({ hasText: title });

/** Creates a Mind with "+" in the tab strip, in a new tab, and names it. */
async function newMind(window: Page, title: string): Promise<void> {
  const count = await tabsOf(window).count();
  await window.getByTestId("new-tab").click();
  await expect(tabsOf(window)).toHaveCount(count + 1);
  await window.getByTestId("mind-title").fill(title);
  await expect(activeTab(window).getByTestId("mind-tab-title")).toHaveText(title);
}

test("Minds open as tabs; the sidebar switches to an open one, and ⌘-click or middle-click opens a new tab", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  // No tab yet: the empty state, and no Export.
  await expect(window.getByTestId("mind-none-open")).toBeVisible();
  await expect(window.getByTestId("export-mind")).toHaveCount(0);

  for (const title of ["Alpha", "Beta", "Gamma"]) await newMind(window, title);
  await expect(titlesOf(window)).toHaveText(["Alpha", "Beta", "Gamma"]);
  await expect(activeTab(window)).toHaveCount(1);
  await expect(activeTab(window)).toContainText("Gamma");
  await expect(window.getByRole("tablist", { name: "Open Minds" })).toBeVisible();

  // A Mind already open: the sidebar switches to its tab.
  await sidebarMind(window, "Alpha").click();
  await expect(tabsOf(window)).toHaveCount(3);
  await expect(activeTab(window)).toContainText("Alpha");
  const alphaId = await sidebarMind(window, "Alpha").getAttribute("data-mind-id");
  await expect(window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${alphaId}`);
  await expect(window.getByRole("tabpanel")).toHaveAttribute(
    "aria-labelledby",
    `mind-tab-${alphaId}`,
  );

  // Clicking a tab shows its Mind.
  await tabNamed(window, "Beta").click();
  await expect(activeTab(window)).toContainText("Beta");

  // Down to one tab; a plain click on another Mind opens it in that tab.
  await tabNamed(window, "Alpha").getByTestId("mind-tab-close").click();
  await tabNamed(window, "Gamma").getByTestId("mind-tab-close").click();
  await expect(titlesOf(window)).toHaveText(["Beta"]);
  await sidebarMind(window, "Gamma").click();
  await expect(titlesOf(window)).toHaveText(["Gamma"]);

  // ⌘-click (Ctrl-click) opens a new tab after the current one, and shows it.
  await sidebarMind(window, "Alpha").click({ modifiers: ["ControlOrMeta"] });
  await expect(titlesOf(window)).toHaveText(["Gamma", "Alpha"]);
  await expect(activeTab(window)).toContainText("Alpha");
  // So does a middle click.
  await tabNamed(window, "Gamma").click();
  await sidebarMind(window, "Beta").click({ button: "middle" });
  await expect(titlesOf(window)).toHaveText(["Gamma", "Beta", "Alpha"]);
  await expect(activeTab(window)).toContainText("Beta");
  await app.close();
});

test("tabs close, move and switch by mouse and keyboard; a deleted Mind's tab closes; the last tab closed shows the empty state", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  // ⌘T (Ctrl+T) is a new Mind in a new tab.
  for (const title of ["Alpha", "Beta", "Gamma"]) {
    const count = await tabsOf(window).count();
    await window.keyboard.press("ControlOrMeta+t");
    await expect(tabsOf(window)).toHaveCount(count + 1);
    await window.getByTestId("mind-title").fill(title);
    await window.getByTestId("mind-title").blur();
  }
  await expect(titlesOf(window)).toHaveText(["Alpha", "Beta", "Gamma"]);

  // ⌘1–⌘9 jump to a tab (⌘9 the last); Ctrl+Tab and Ctrl+Shift+Tab cycle.
  await window.keyboard.press("ControlOrMeta+1");
  await expect(activeTab(window)).toContainText("Alpha");
  await window.keyboard.press("ControlOrMeta+9");
  await expect(activeTab(window)).toContainText("Gamma");
  await window.keyboard.press("Control+Tab");
  await expect(activeTab(window)).toContainText("Alpha");
  await window.keyboard.press("Control+Shift+Tab");
  await expect(activeTab(window)).toContainText("Gamma");

  // The tabs are a tab list: the arrow keys move to the next tab and show it.
  await activeTab(window).focus();
  await window.keyboard.press("ArrowLeft");
  await expect(activeTab(window)).toContainText("Beta");
  await expect(activeTab(window)).toBeFocused();
  await expect(tabsOf(window).and(window.locator('[tabindex="0"]'))).toHaveCount(1);

  // Dragged onto the right half of Gamma, Alpha moves after it.
  const gamma = tabNamed(window, "Gamma");
  const gammaBox = await boxOf(gamma);
  await tabNamed(window, "Alpha").dragTo(gamma, {
    targetPosition: { x: gammaBox.width - 12, y: gammaBox.height / 2 },
  });
  await expect(titlesOf(window)).toHaveText(["Beta", "Gamma", "Alpha"]);
  // And onto the left half of Beta, before it.
  const beta = tabNamed(window, "Beta");
  await tabNamed(window, "Alpha").dragTo(beta, { targetPosition: { x: 12, y: 17 } });
  await expect(titlesOf(window)).toHaveText(["Alpha", "Beta", "Gamma"]);

  // A middle click closes a tab; closing the shown one shows the tab after it.
  await expect(activeTab(window)).toContainText("Beta");
  await tabNamed(window, "Beta").click({ button: "middle" });
  await expect(titlesOf(window)).toHaveText(["Alpha", "Gamma"]);
  await expect(activeTab(window)).toContainText("Gamma");
  // ⌘W (Ctrl+W) closes the shown tab, and not the window.
  await window.keyboard.press("ControlOrMeta+w");
  await expect(titlesOf(window)).toHaveText(["Alpha"]);
  await expect(activeTab(window)).toContainText("Alpha");
  expect(window.isClosed()).toBe(false);

  // Deleting a Mind closes its tab.
  await sidebarMind(window, "Gamma").click({ modifiers: ["ControlOrMeta"] });
  await expect(titlesOf(window)).toHaveText(["Alpha", "Gamma"]);
  const gammaRow = window.getByTestId("mind-row").filter({ hasText: "Gamma" });
  await gammaRow.hover();
  await gammaRow.getByTestId("mind-menu").click();
  await gammaRow.getByTestId("delete-mind").click();
  await window.getByTestId("confirm-delete-mind").click();
  await expect(titlesOf(window)).toHaveText(["Alpha"]);
  await expect(activeTab(window)).toContainText("Alpha");

  // Closing the last tab shows the empty state.
  await activeTab(window).getByTestId("mind-tab-close").click();
  await expect(tabsOf(window)).toHaveCount(0);
  await expect(window.getByTestId("mind-none-open")).toBeVisible();
  await expect(window.getByTestId("mind-pane")).toHaveCount(0);
  await expect(window.getByTestId("export-mind")).toHaveCount(0);
  await app.close();
});

test("open tabs, their order and the shown one come back after a restart", async () => {
  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  for (const title of ["Alpha", "Beta", "Gamma", "Delta"]) await newMind(first.window, title);
  // Delta is closed; Beta moves first and is shown.
  await tabNamed(first.window, "Delta").getByTestId("mind-tab-close").click();
  await tabNamed(first.window, "Beta").dragTo(tabNamed(first.window, "Alpha"), {
    targetPosition: { x: 12, y: 17 },
  });
  await tabNamed(first.window, "Beta").click();
  await expect(titlesOf(first.window)).toHaveText(["Beta", "Alpha", "Gamma"]);
  const betaId = await sidebarMind(first.window, "Beta").getAttribute("data-mind-id");
  await first.app.close();

  const second = await launchApp(dataDir);
  const { window } = second;
  await expect(titlesOf(window)).toHaveText(["Beta", "Alpha", "Gamma"]);
  await expect(activeTab(window)).toContainText("Beta");
  await expect(window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${betaId}`);
  await expect(window.getByTestId("mind-title")).toHaveValue("Beta");
  await second.app.close();
});

test("the shown tab joins the Mind below with no edge under it, and long titles truncate on one line", async () => {
  const long =
    "A very long Mind title about the scaling laws of neural language models, data and compute, and what they predict";
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window, "Short");
  await newMind(window, long);
  await newMind(window, "Third");
  await tabNamed(window, "Short").click();

  const strip = window.getByTestId("mind-header");
  const stripBox = await boxOf(strip);
  expect(stripBox.height).toBe(36);
  const active = activeTab(window);
  const activeBox = await boxOf(active);
  // Its bottom is the strip's bottom, and the Mind starts right there.
  expect(activeBox.height).toBe(32);
  expect(activeBox.y + activeBox.height).toBeCloseTo(stripBox.y + stripBox.height, 1);
  const panelBox = await boxOf(window.getByRole("tabpanel"));
  expect(panelBox.y).toBeCloseTo(activeBox.y + activeBox.height, 1);
  // No edge under it: no border, the Mind's own fill, and nothing drawn over its last row.
  const joined = await active.evaluate((tab) => {
    const style = getComputedStyle(tab);
    const box = tab.getBoundingClientRect();
    const panel = document.querySelector('[role="tabpanel"]');
    const below = panel ? getComputedStyle(panel.closest("main") ?? panel).backgroundColor : "";
    const atBottom = document.elementFromPoint(box.left + box.width / 2, box.bottom - 0.5);
    return {
      border: style.borderBottomWidth,
      fill: style.backgroundColor,
      below,
      onTop: atBottom !== null && tab.contains(atBottom),
    };
  });
  expect(joined).toEqual({
    border: "0px",
    fill: "rgb(255, 255, 255)",
    below: "rgb(255, 255, 255)",
    onTop: true,
  });
  // The strip has no bottom rule either.
  expect(await strip.evaluate((element) => getComputedStyle(element).borderBottomWidth)).toBe(
    "0px",
  );
  if (SCREENSHOTS) await strip.screenshot({ path: `${SCREENSHOTS}/tabs.png` });

  // The long title: one line, cut short with "…", in a tab no wider than 220px.
  const longTab = tabNamed(window, long);
  const longBox = await boxOf(longTab);
  expect(longBox.height).toBe(34);
  expect(longBox.width).toBeLessThanOrEqual(220);
  const title = longTab.getByTestId("mind-tab-title");
  expect(await boxOf(title).then((box) => box.height)).toBe(20);
  expect(
    await title.evaluate((element) => ({
      cut: element.scrollWidth > element.clientWidth,
      wrap: getComputedStyle(element).whiteSpace,
      ellipsis: getComputedStyle(element).textOverflow,
    })),
  ).toEqual({ cut: true, wrap: "nowrap", ellipsis: "ellipsis" });
  // Its full title is its tooltip.
  await expect(longTab).toHaveAttribute("title", long);
  await app.close();
});

test("a new Mind's title has the focus, so typing names it instead of making more Minds", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const minds = window.getByTestId("mind-list-item");

  // Clicked, the "+" menu's New Mind keeps no focus: a space typed next is part of the title.
  for (const testId of ["plus-menu", "plus-new-mind"]) {
    const button = await boxOf(window.getByTestId(testId));
    await window.mouse.move(button.x + button.width / 2, button.y + button.height / 2, {
      steps: 5,
    });
    await window.mouse.down();
    await window.mouse.up();
  }
  const title = window.getByTestId("mind-title");
  await expect(title).toBeFocused();
  await window.keyboard.type("Hello world");
  await expect(minds).toHaveCount(1);
  await expect(title).toHaveValue("Hello world");
  await expect(titlesOf(window)).toHaveText(["Hello world"]);

  // Enter moves on into the Mind's text; the tab strip's "+" does the same as New Mind.
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("mind-editor")).toBeFocused();
  await window.getByTestId("new-tab").click();
  await expect(title).toBeFocused();
  await window.keyboard.type("Second one");
  await expect(minds).toHaveCount(2);
  await expect(titlesOf(window)).toHaveText(["Hello world", "Second one"]);
  await app.close();
});

test("tabs are 220px while there is room, and a mouse wheel scrolls the strip to tabs out of sight", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.setSize(1280, 800));
  await expect.poll(() => window.evaluate(() => globalThis.innerWidth)).toBe(1280);
  for (const name of ["Alpha notes", "Beta research", "Gamma draft"]) {
    await newMindFromMenu(window);
    await expect(window.getByTestId("mind-title")).toBeFocused();
    await window.keyboard.type(name);
  }
  // Short titles show in full: the tabs take their 220px, not their text's width.
  for (const tab of await tabsOf(window).all())
    expect((await boxOf(tab)).width).toBeCloseTo(220, 0);
  await expect(titlesOf(window)).toHaveText(["Alpha notes", "Beta research", "Gamma draft"]);

  // More tabs than room: the first ones scroll out of sight, and the strip fades on that side.
  for (let index = 0; index < 13; index++) {
    await window.getByTestId("new-tab").click();
    await expect(tabsOf(window)).toHaveCount(4 + index);
  }
  const strip = window.getByTestId("mind-tabs");
  await expect(strip).toHaveAttribute("data-hidden-before", "true");
  // A mouse wheel, turned up, brings them back.
  const box = await boxOf(strip);
  await window.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await window.mouse.wheel(0, -2000);
  await expect(strip).not.toHaveAttribute("data-hidden-before");
  await expect(strip).toHaveAttribute("data-hidden-after", "true");
  await app.close();
});

/** Clicks the application menu's item with this label, as a person picks it from the menu bar. */
async function chooseMenuItem(app: ElectronApplication, label: string): Promise<void> {
  const found = await app.evaluate(({ Menu }, wanted) => {
    const find = (items: Electron.MenuItem[]): Electron.MenuItem | undefined => {
      for (const item of items) {
        if (item.label === wanted) return item;
        const inside = item.submenu && find(item.submenu.items);
        if (inside) return inside;
      }
      return undefined;
    };
    const item = find(Menu.getApplicationMenu()?.items ?? []);
    item?.click();
    return item !== undefined;
  }, label);
  expect(found, `the menu has "${label}"`).toBe(true);
}

/** The labels of the application menu's top-level menus. */
const menuTitles = (app: ElectronApplication) =>
  app.evaluate(({ Menu }) => (Menu.getApplicationMenu()?.items ?? []).map((item) => item.label));

test("the application menu makes a new Mind, opens Settings and closes the tab, in the interface's language", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);

  await chooseMenuItem(app, "New Mind");
  await expect(window.getByTestId("mind-list-item")).toHaveCount(1);
  await expect(window.getByTestId("mind-title")).toBeFocused();
  await chooseMenuItem(app, "Settings…");
  await expect(window.getByTestId("settings")).toBeVisible();
  await window.keyboard.press("Escape");
  await expect(window.getByTestId("settings")).toBeHidden();
  await chooseMenuItem(app, "Close Tab");
  await expect(tabsOf(window)).toHaveCount(0);

  // In Chinese, the menu is too.
  await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.updateSettings({ user: { language: "zh-CN" } });
  });
  await expect.poll(() => menuTitles(app)).toContain("文件");
  await chooseMenuItem(app, "新建 Mind");
  await expect(window.getByTestId("mind-list-item")).toHaveCount(2);
  await app.close();
});

test("a Mind is renamed in place from its sidebar row, and Not in a Folder folds away", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  for (const name of ["Alpha notes", "Beta research"]) {
    await newMindFromMenu(window);
    await expect(window.getByTestId("mind-title")).toBeFocused();
    await window.keyboard.type(name);
  }
  const rows = window.getByTestId("mind-list-item");
  const alpha = rows.filter({ hasText: "Alpha notes" });

  // A double click types a new title in place; Enter saves it, and the tab follows.
  await alpha.dblclick();
  const field = window.getByTestId("mind-rename");
  await expect(field).toBeFocused();
  await field.fill("Alpha, revised");
  await field.press("Enter");
  await expect(rows.filter({ hasText: "Alpha, revised" })).toHaveCount(1);
  await expect(titlesOf(window)).toContainText(["Alpha, revised"]);

  // From the row's menu, Esc changes nothing.
  const beta = window
    .getByTestId("mind-row")
    .filter({ has: rows.filter({ hasText: "Beta research" }) });
  await beta.hover();
  await beta.getByTestId("mind-menu").click();
  await beta.getByTestId("rename-mind").click();
  await field.fill("Not this");
  await field.press("Escape");
  await expect(rows.filter({ hasText: "Beta research" })).toHaveCount(1);

  // By keyboard: Enter on the open Mind's row (selected) types its title in place, as F2 does.
  const betaRow = rows.filter({ hasText: "Beta research" });
  await betaRow.click();
  await betaRow.focus();
  await window.keyboard.press("Enter");
  await expect(field).toBeFocused();
  await window.keyboard.press("Escape");
  await expect(betaRow).toBeFocused();
  await window.keyboard.press("F2");
  await expect(field).toBeFocused();
  await window.keyboard.press("Escape");

  // Folded, Not in a Folder gives its room to the rest.
  const toggle = window.getByTestId("not-in-a-folder-toggle");
  await toggle.click();
  await expect(rows).toHaveCount(0);
  await toggle.click();
  await expect(rows).toHaveCount(2);
  await app.close();
});

test("the card's band runs unbroken: tabs start at the card's edge, the viewer's toolbar sits on it, and no line crosses another", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  for (const name of ["First", "Second"]) {
    await newMindFromMenu(window);
    await expect(window.getByTestId("mind-title")).toBeFocused();
    await window.keyboard.type(name);
  }
  await tabsOf(window).first().click();
  await openViewer(window);

  // The first tab, shown, starts flush at the card's edge (8px from the sidebar), rounded at the top, with no left foot.
  const sidebar = await boxOf(window.getByTestId("sidebar"));
  const first = tabsOf(window).first();
  expect((await boxOf(first)).x).toBeCloseTo(sidebar.x + sidebar.width + 8, 0);
  expect(await first.evaluate((tab) => getComputedStyle(tab).borderTopLeftRadius)).toBe("10px");
  await expect(first.locator('.mind-tab-foot[data-side="left"]')).toBeHidden();
  // The sidebar header's rule is inset 8px each side, so it never meets the sidebar's edge.
  const rule = await window.getByTestId("sidebar-header").evaluate((element) => {
    const after = getComputedStyle(element, "::after");
    return { left: after.left, right: after.right, height: after.height };
  });
  expect(rule).toEqual({ left: "8px", right: "8px", height: "1px" });

  // The viewer's toolbar is on the band, with no rule under it; its divider starts below the band.
  const band = await window
    .getByTestId("mind-header")
    .evaluate((element) => getComputedStyle(element).backgroundColor);
  const toolbar = window.getByTestId("viewer-header");
  expect(await toolbar.evaluate((element) => getComputedStyle(element).backgroundColor)).toBe(band);
  expect(await toolbar.evaluate((element) => getComputedStyle(element).borderBottomWidth)).toBe(
    "0px",
  );
  expect(
    await window
      .getByTestId("viewer-resize")
      .evaluate((element) => getComputedStyle(element).backgroundImage),
  ).toContain("linear-gradient");
  await app.close();
});
