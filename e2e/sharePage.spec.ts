import { randomUUID } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Page, test } from "@playwright/test";
import type { CitationAttributes, CoreBridge, Document } from "../src/core/api";
import {
  addDocuments,
  blocksUpdate,
  createDataFolder,
  dismissChatSetup,
  exportFromMenu,
  launchApp,
  type MindChild,
  mindNode,
  newMind,
  openOptions,
  pointAndClick,
  removeDataFolder,
  savedPath,
  setLanguage,
  useOutputLocation,
} from "./app";

const core = (window: Page) => ({
  documents: () =>
    window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments(),
    ),
  write: (mindId: string, update: Uint8Array) =>
    window.evaluate(
      ([id, bytes]) =>
        (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.applyMindUpdate(
          id,
          new Uint8Array(bytes),
        ),
      [mindId, Array.from(update)] as const,
    ),
});

/** A Citation of `quote` on line `line` of a text file, as an Answer makes it. */
function citation(document: Document, quote: string, line: number, check: "found" | "not-found") {
  const attributes: CitationAttributes = {
    passageId: "passage-1",
    documentId: document.id,
    documentName: document.name,
    contentHash: document.contentHash,
    pageFrom: 1,
    pageTo: 1,
    location: { kind: "lines", from: line, to: line },
    quote,
    check,
    checkReason: check === "found" ? null : "quote-not-on-pages",
  };
  return mindNode("citation", { ...attributes });
}

const paragraph = (...children: MindChild[]) =>
  mindNode("paragraph", { id: randomUUID() }, children);

/**
 * Opens a saved page in a window of its own, in a session with the network off that records
 * every request it is asked for, as a reader opening the file in a browser would.
 */
async function openOffline(app: ElectronApplication, path: string): Promise<Page> {
  const opened = app.waitForEvent("window");
  await app.evaluate(async ({ BrowserWindow, session }, file) => {
    const offline = session.fromPartition("shared-page", { cache: false });
    offline.enableNetworkEmulation({ offline: true });
    const requested: string[] = [];
    (globalThis as { sharedPageRequests?: string[] }).sharedPageRequests = requested;
    offline.webRequest.onBeforeRequest((details, callback) => {
      requested.push(details.url);
      callback({ cancel: !details.url.startsWith("file:") });
    });
    const window = new BrowserWindow({
      width: 1100,
      height: 760,
      useContentSize: true,
      webPreferences: { session: offline, sandbox: true, contextIsolation: true },
    });
    await window.loadFile(file);
  }, path);
  const page = await opened;
  await page.waitForLoadState("load");
  return page;
}

const requestsMade = (app: ElectronApplication) =>
  app.evaluate(() => (globalThis as { sharedPageRequests?: string[] }).sharedPageRequests ?? []);

async function resize(app: ElectronApplication, page: Page, width: number, height: number) {
  const window = await app.browserWindow(page);
  await window.evaluate((win, size) => win.setContentSize(size.width, size.height), {
    width,
    height,
  });
  await expect
    .poll(() => page.evaluate(() => [globalThis.innerWidth, globalThis.innerHeight]))
    .toEqual([width, height]);
}

let dataDir: string;
let sources: string;
let home: string;
let out: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
  home = await createDataFolder();
  out = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
  await removeDataFolder(home);
  await removeDataFolder(out);
});

test("a web page to share: the sheet offers it, and the file loads nothing and opens each Citation's card by keyboard and as a sheet at 375px", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await dismissChatSetup(window);
  await newMind(window);
  await window.getByTestId("mind-title").fill("Tides");
  const mindId = await window.getByTestId("mind-pane").getAttribute("data-mind-id");
  if (!mindId) throw new Error("No Mind is open.");
  await writeFile(
    join(sources, "Tides.txt"),
    "Spring tides come twice a month, at new and full moon.\n\nNeap tides fall between them.\n",
  );
  await addDocuments(window, [join(sources, "Tides.txt")]);
  const [tides] = await core(window).documents();
  if (!tides) throw new Error("No Tides.");
  await core(window).write(
    mindId,
    blocksUpdate([
      paragraph(
        "Spring tides follow the moon",
        citation(tides, "Spring tides come twice a month", 1, "found"),
        ".",
      ),
      paragraph(
        "Neap tides come between",
        citation(tides, "Neap tides fall between them", 3, "found"),
        ".",
      ),
      paragraph(
        "Tides stop in winter",
        citation(tides, "Tides stop in winter", 1, "not-found"),
        ".",
      ),
    ]),
  );
  await expect(window.getByTestId("mind-editor").getByTestId("citation")).toHaveCount(3);
  await useOutputLocation(window, out);

  // No citation style: the page numbers each Citation.
  await openOptions(window, "Export options");
  const sheet = window.getByTestId("export-dialog");
  const format = sheet.getByTestId("export-format");
  await expect(format.locator("option")).toContainText(["HTML", "Web page to share (.html)"]);
  await format.selectOption("share-page");
  const cards = sheet.getByTestId("export-citation-cards");
  const line = sheet.getByTestId("export-checked-line");
  await expect(cards).toBeChecked();
  await expect(line).not.toBeChecked();
  await expect(sheet.getByTestId("export-citation-style")).toHaveCount(0);
  await pointAndClick(window, cards);
  await expect(cards).not.toBeChecked();
  await expect(line).toHaveCount(0);
  await pointAndClick(window, cards);
  await pointAndClick(window, line);
  await expect(line).toBeChecked();
  await pointAndClick(window, sheet.getByTestId("export-save"));
  await expect(sheet).toBeHidden();
  await expect(window.getByTestId("export-mind")).toHaveAttribute("data-format", "share-page");

  await exportFromMenu(window);
  const page = await openOffline(app, await savedPath(window, "Tides.html"));

  // Nothing but the file was asked for, and the page's policy refuses a request it tries.
  await expect(page.locator(".shared-foot a")).toHaveText("Citations checked with IncarnaMind");
  expect(
    await page.evaluate(() =>
      fetch("https://example.com/").then(
        () => "loaded",
        () => "refused",
      ),
    ),
  ).toBe("refused");
  const requests = await requestsMade(app);
  expect(requests.some((url) => url.startsWith("file:"))).toBe(true);
  expect(requests.filter((url) => !url.startsWith("file:"))).toEqual([]);

  const markers = page.locator("main a.cite");
  await expect(markers).toHaveCount(3);
  await expect(markers.first()).toHaveAttribute("role", "button");
  await expect(markers.first()).toHaveAccessibleName("Citation 1: Tides, line 1");
  await page.keyboard.press("Tab");
  await expect(markers.first()).toBeFocused();
  await page.keyboard.press("Enter");
  const first = page.locator("#card-1");
  await expect(first).toBeVisible();
  await expect(first).toBeFocused();
  await expect(first).toHaveAccessibleName("Citation 1: Tides, line 1");
  await expect(markers.first()).toHaveAttribute("aria-expanded", "true");
  const marker = await markers.first().boundingBox();
  const card = await first.boundingBox();
  if (!marker || !card) throw new Error("The marker or its card isn't shown.");
  expect(card.y).toBeGreaterThanOrEqual(marker.y + marker.height);
  expect(card.y - (marker.y + marker.height)).toBeLessThanOrEqual(16);
  expect(card.x).toBeLessThanOrEqual(marker.x);
  expect(card.x + card.width).toBeGreaterThan(marker.x);
  await expect(first.locator("mark.quote")).toHaveText("Spring tides come twice a month");

  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#card-2")).toBeFocused();
  await expect(first).toBeHidden();
  await page.keyboard.press("ArrowRight");
  const third = page.locator("#card-3");
  await expect(third).toBeFocused();
  await expect(third.locator(".card-heading")).toHaveText("Quote not found on line 1");
  await page.keyboard.press("ArrowRight");
  await expect(third).toBeVisible();
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await expect(first).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(first.locator(".card-close")).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(first.locator(".card-close")).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(first).toBeHidden();
  await expect(markers.first()).toBeFocused();
  await expect(markers.first()).toHaveAttribute("aria-expanded", "false");

  await page.keyboard.press(" ");
  await expect(first).toBeVisible();
  await page.mouse.click(1000, 700);
  await expect(first).toBeHidden();

  await resize(app, page, 375, 720);
  await pointAndClick(page, markers.nth(1));
  const second = page.locator("#card-2");
  await expect(second).toBeVisible();
  await expect(page.locator(".scrim")).toBeVisible();
  await expect(second.locator(".card-swipe")).toHaveText("Swipe down or tap outside to close.");
  const sheetBox = await second.boundingBox();
  if (!sheetBox) throw new Error("The card isn't shown.");
  expect(Math.abs(sheetBox.x)).toBeLessThanOrEqual(1);
  // Beside a classic scrollbar (Linux, Windows) the sheet spans what is left of the width.
  const room = await page.evaluate(() => document.documentElement.clientWidth);
  expect(Math.abs(sheetBox.width - room)).toBeLessThanOrEqual(1);
  expect(Math.abs(sheetBox.y + sheetBox.height - 720)).toBeLessThanOrEqual(1);
  expect(
    await second.locator(".card-source .date").evaluate((date) => date.getClientRects().length),
  ).toBe(1);
  const grab = await second.locator(".card-grab").boundingBox();
  if (!grab) throw new Error("The sheet has no grab bar.");
  await page.mouse.move(grab.x + grab.width / 2, grab.y + 2);
  await page.mouse.down();
  await page.mouse.move(grab.x + grab.width / 2, grab.y + 140, { steps: 6 });
  await page.mouse.up();
  await expect(second).toBeHidden();
  await expect(page.locator(".scrim")).toBeHidden();

  await pointAndClick(page, markers.nth(1));
  await expect(second).toBeVisible();
  await page.mouse.click(187, 20);
  await expect(second).toBeHidden();
  await pointAndClick(page, markers.nth(1));
  await page.keyboard.press("Escape");
  await expect(second).toBeHidden();
  expect((await requestsMade(app)).filter((url) => !url.startsWith("file:"))).toEqual([]);
  await page.close();

  await setLanguage(window, "zh-CN");
  await openOptions(window, "导出选项");
  await expect(sheet.getByTestId("export-format").locator("option:checked")).toHaveText(
    "可分享的网页（.html）",
  );
  await expect(sheet.getByText("读者可以打开每条引用")).toBeVisible();
  await expect(sheet.getByText("在页脚注明“引用已由 IncarnaMind 核对”")).toBeVisible();
  await app.close();
});
