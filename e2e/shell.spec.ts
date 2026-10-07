import { mkdir, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge, LinkedFolder } from "../src/core/api";
import {
  addDocuments,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  removeDataFolder,
  useLocalChatModel,
} from "./app";

/*
 * The app's chrome lines up (DESIGN.md, Layout): every sidebar row puts its
 * icon at x 16 and its text at x 40, a Folder's children are one 24px step
 * deeper, section labels share the icon column, and every pane header is
 * 44px. Set INCARNAMIND_SCREENSHOTS to a folder to also save screenshots of
 * the sidebar, first run and each Settings page there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

/** The tiny MCP server the core tests use: `book_boat` may change something, so it asks. */
const TIDE_SERVER = resolve(__dirname, "../tests/fixtures/mcp-server.mjs");

let dataDir: string;
/** Where the files a test adds are written: outside the data folder. */
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

async function screenshot(page: Page, name: string, clip?: Locator): Promise<void> {
  if (!SCREENSHOTS) return;
  const path = join(SCREENSHOTS, `${name}.png`);
  if (clip) await clip.screenshot({ path });
  else await page.screenshot({ path });
}

/** Where an element's text starts, in CSS pixels from the window's left edge. */
function textLeft(locator: Locator): Promise<number> {
  return locator.evaluate((element) => {
    const range = document.createRange();
    range.selectNodeContents(element);
    const first = range.getClientRects()[0];
    if (!first) throw new Error("The element has no text.");
    return first.left;
  });
}

async function boxOf(locator: Locator) {
  const box = await locator.boundingBox();
  if (!box) throw new Error("The element isn't visible.");
  return box;
}

/** Each row's icon and text, from the sidebar's left edge. */
async function rowEdges(sidebar: Locator, rows: Locator) {
  const left = (await boxOf(sidebar)).x;
  const edges: { icon: number; text: number; depth: number }[] = [];
  for (const row of await rows.all()) {
    const icon = await boxOf(row.locator("svg").first());
    edges.push({
      icon: icon.x - left,
      text: (await textLeft(row.getByTestId("row-text"))) - left,
      depth: Number((await row.getAttribute("data-depth")) ?? "0"),
    });
  }
  return edges;
}

/** Links a folder through the core's bridge, as "Add folder…" does once the User picked it. */
const linkFolder = (page: Page, path: string) =>
  page.evaluate(
    async (wanted) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.addLinkedFolder(wanted),
    path,
  ) as Promise<LinkedFolder>;

/** Writes `files` (relative paths) under `root`, each holding its own name. */
async function writeTree(root: string, files: string[]): Promise<void> {
  for (const file of files) {
    await mkdir(dirname(join(root, file)), { recursive: true });
    await writeFile(join(root, file), `${basename(file, ".txt")}.\n`);
  }
}

/** Long enough for a few dozen Passages, so "Embedding…" stays on screen for a moment. */
const LONG_TEXT = Array.from(
  { length: 4000 },
  (_, index) => `Line ${index}: test loss falls as a power law in model size and data.`,
).join("\n");

test("sidebar rows share one text edge, a Folder's children are one step deeper, and pane headers are 44px", async () => {
  // Two Linked folders, one with a folder inside, and a file added on its own.
  const papersPath = join(sources, "Papers");
  const reportsPath = join(sources, "Reports");
  await writeTree(papersPath, [
    "Language Models are Unsupervised Multitask Learners.txt",
    "2026/Attention Is All You Need.txt",
  ]);
  await writeTree(reportsPath, ["Quarterly report.txt"]);
  await writeTree(sources, ["Supervisor meeting notes.txt"]);
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);

  for (const title of [
    "Methods: evaluation design",
    "Literature review, chapter 2",
    "Reading notes: LM scaling",
  ]) {
    await window.getByTestId("new-mind").click();
    await window.getByTestId("mind-title").fill(title);
    await expect(window.getByTestId("mind-list-item").first()).toHaveText(title);
  }
  await addDocuments(window, [join(sources, "Supervisor meeting notes.txt")]);
  const papers = await linkFolder(window, papersPath);
  const reports = await linkFolder(window, reportsPath);
  const tree = window.getByTestId("sidebar-tree");
  const folderRow = (folderId: string) =>
    tree.locator(`[data-folder-id="${folderId}"][data-testid="folder-item"]`);
  const documents = window.getByTestId("document-list-item");
  await expect(documents.filter({ hasText: "Attention Is All You Need" })).toHaveAttribute(
    "data-depth",
    "2",
  );

  const sidebar = window.getByTestId("sidebar");
  // Every Mind, Folder, group and Document row: depth 0 at x 16 / 40, each level 24px deeper.
  const rows = tree.locator(
    '[data-testid="mind-list-item"], [data-testid="folder-item"], [data-testid="other-documents"], [data-testid="document-list-item"]',
  );
  await expect(rows).toHaveCount(3 + 3 + 1 + 4);
  const edges = await rowEdges(sidebar, rows);
  for (const { icon, text, depth } of edges) {
    expect(icon).toBeCloseTo(16 + depth * 24, 0);
    expect(text).toBeCloseTo(40 + depth * 24, 0);
  }
  // A Folder's children: their icon under its text, so their text is one step deeper.
  const papersText = await textLeft(folderRow(papers.folderId).getByTestId("row-text"));
  const child = documents.filter({ hasText: "Language Models" });
  expect(await boxOf(child.locator("svg").first()).then((box) => box.x)).toBeCloseTo(papersText, 0);
  expect(await textLeft(child.getByTestId("row-text"))).toBeCloseTo(papersText + 24, 0);
  // Minds, both Linked folders and "Other Documents" at the top; 2026 and the Documents in
  // Papers, Reports and Other Documents one step in; the Document in 2026 two.
  expect(edges.map((edge) => edge.depth).sort()).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2]);

  // "New Mind", Settings and the section labels share the columns.
  const sidebarLeft = (await boxOf(sidebar)).x;
  const newMind = window.getByTestId("new-mind");
  expect((await boxOf(newMind.locator("svg"))).x - sidebarLeft).toBeCloseTo(16, 0);
  expect((await textLeft(newMind.locator("span"))) - sidebarLeft).toBeCloseTo(40, 0);
  const settings = window.getByTestId("sidebar-footer").getByRole("button", { name: "Settings" });
  expect((await boxOf(settings.locator("svg"))).x - sidebarLeft).toBeCloseTo(16, 0);
  expect((await textLeft(settings.locator("span"))) - sidebarLeft).toBeCloseTo(40, 0);
  for (const label of [window.locator("#minds-heading"), window.locator("#documents-heading")]) {
    expect((await textLeft(label)) - sidebarLeft).toBeCloseTo(16, 0);
  }
  // The app's name sits on the same text edge as the rows.
  const header = window.getByTestId("sidebar-header");
  expect((await textLeft(header.locator("span").last())) - sidebarLeft).toBeCloseTo(40, 0);

  // Rows are 28px.
  for (const row of [newMind, ...(await rows.all())]) {
    expect((await boxOf(row)).height).toBeCloseTo(28, 0);
  }

  // Both pane headers are 44px and line up: the sidebar's, and the Mind pane's strip of tabs.
  const mindHeader = window.getByTestId("mind-header");
  const sidebarHeader = await boxOf(header);
  const mindHeaderBox = await boxOf(mindHeader);
  expect(sidebarHeader.height).toBe(44);
  expect(mindHeaderBox.height).toBe(44);
  expect(mindHeaderBox.y).toBe(sidebarHeader.y);
  await expect(
    mindHeader.locator('[role="tab"][aria-selected="true"]').getByTestId("mind-tab-title"),
  ).toHaveText("Reading notes: LM scaling");

  // Folding a Folder hides what's inside it.
  const reportsRow = folderRow(reports.folderId);
  await reportsRow.getByTestId("folder-toggle").click();
  await expect(documents.filter({ hasText: "Quarterly report" })).toHaveCount(0);

  // A processing Document shows its progress at the row's end, on one line.
  await writeFile(join(sources, "Scaling Laws for Neural Language Models.txt"), LONG_TEXT);
  await window
    .getByTestId("add-documents-input")
    .setInputFiles([join(sources, "Scaling Laws for Neural Language Models.txt")]);
  const processing = documents.filter({ hasText: "Scaling Laws" });
  await expect(processing).toHaveAttribute("data-status", "embedding");
  await expect(processing.getByTestId("document-status")).toContainText("Embedding…");
  await expect(processing).toHaveAttribute("data-depth", "1");
  expect((await boxOf(processing)).height).toBeCloseTo(28, 0);
  // A Folder folded with the mouse keeps no highlight once the pointer moves on.
  await window.mouse.move(800, 400);
  await expect
    .poll(() => reportsRow.evaluate((element) => getComputedStyle(element).backgroundColor))
    .toBe("rgba(0, 0, 0, 0)");
  if (SCREENSHOTS) {
    // For the picture, a moment past 0%; no matter if it's done first.
    await processing
      .getByTestId("document-status")
      .filter({ hasText: /Embedding… [1-9]/ })
      .waitFor({ timeout: 5_000 })
      .catch(() => undefined);
  }
  await screenshot(window, "sidebar");
  await screenshot(window, "sidebar-only", sidebar);
  await expect(processing).toHaveAttribute("data-status", "ready", { timeout: 30_000 });
  await app.close();
});

test("the status footer doesn't move the tree when tagging starts or stops", async () => {
  const paper = join(sources, "Report on attention.txt");
  await writeFile(paper, "A report on attention in language models.\n");
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await window.getByTestId("new-mind").click();
  await window.getByTestId("mind-title").fill("Field notes");
  // An empty Linked folder: its own Folder is listed, with nothing in it to tag.
  await mkdir(join(sources, "Projects"));
  await linkFolder(window, join(sources, "Projects"));

  const tree = window.getByTestId("sidebar-tree");
  const footer = window.getByTestId("sidebar-footer");
  const status = window.getByTestId("sidebar-status");
  const watched = [
    window.getByTestId("new-mind"),
    window.getByTestId("mind-list-item"),
    window.getByTestId("folder-item"),
    window.locator("#documents-heading"),
  ];
  const layout = async () => ({
    rows: await Promise.all(watched.map(async (each) => (await boxOf(each)).y)),
    tree: await boxOf(tree),
    footer: await boxOf(footer),
    status: await boxOf(status),
  });
  await expect(watched[2] as Locator).toBeVisible();
  const before = await layout();

  // No chat model yet: the Document is ready, and tagging waits. The footer says so.
  await addDocuments(window, [paper]);
  const item = window.getByTestId("document-list-item");
  await expect(item).toHaveAttribute("data-tagging", "waiting-for-provider");
  await expect(status.getByTestId("tagging-waiting")).toBeVisible();
  await screenshot(window, "sidebar-tagging-waits", window.getByTestId("sidebar"));
  expect(await layout()).toEqual(before);

  // A chat model: tagging starts, then stops. Nothing above the footer moves.
  await useLocalChatModel(window);
  await expect(item).toHaveAttribute("data-tagging", "tagged");
  await expect(status.locator("[role]")).toHaveCount(0);
  expect(await layout()).toEqual(before);
  await app.close();
});

test("a Mind whose Answer waits for the User's approval shows an amber dot in the sidebar", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await window.evaluate(
    async ({ node, server }) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const connector = await core.addConnector({ name: "Tides", command: node, args: [server] });
      for (let tries = 0; tries < 300; tries++) {
        const found = (await core.listConnectors()).find((each) => each.id === connector.id);
        if (found?.state === "ready") return;
        await new Promise((done) => setTimeout(done, 100));
      }
      throw new Error("The Connector didn't get ready.");
    },
    { node: process.execPath, server: TIDE_SERVER },
  );

  await window.getByTestId("new-mind").click();
  await window.getByTestId("mind-title").fill("Trip");
  await window.getByTestId("new-mind").click();
  await window.getByTestId("mind-title").fill("Other notes");
  const trip = window.getByTestId("mind-list-item").filter({ hasText: "Trip" });
  await trip.click();
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("Please book_boat from Dover");
  await window.keyboard.press("Enter");

  const card = editor.getByTestId("approval-card");
  await expect(card).toBeVisible({ timeout: 15_000 });
  const dot = trip.getByRole("img", { name: "Waiting for your approval" });
  await expect(dot).toBeVisible();
  await expect(
    window
      .getByTestId("mind-list-item")
      .filter({ hasText: "Other notes" })
      .getByTestId("mind-approval"),
  ).toHaveCount(0);
  // It sits at the row's end, inside the row.
  const row = await boxOf(trip);
  const dotBox = await boxOf(dot);
  expect(dotBox.width).toBe(6);
  expect(dotBox.x + dotBox.width).toBeLessThanOrEqual(row.x + row.width - 8);
  // Its tab shows the same amber dot, in place of the Mind icon; the other tab doesn't.
  const tripId = await trip.getAttribute("data-mind-id");
  const tabs = window.getByTestId("mind-tab");
  const tripTab = tabs.and(window.locator(`[data-mind-id="${tripId}"]`));
  const tabDot = tripTab.getByRole("img", { name: "Waiting for your approval" });
  await expect(tabDot).toHaveAttribute("data-status", "waiting-for-approval");
  await expect(tabs.filter({ hasText: "Other notes" }).getByTestId("mind-tab-status")).toHaveCount(
    0,
  );
  expect((await boxOf(tabDot)).width).toBe(6);
  await screenshot(window, "sidebar-approval", window.getByTestId("sidebar"));
  await screenshot(window, "tabs-approval", window.getByTestId("mind-header"));

  await card.getByTestId("approval-deny").click();
  await expect(dot).toHaveCount(0);
  // The Answer goes on (its tab shows it writing) and then ends, as slowly as the machine is.
  await expect(tripTab.getByTestId("mind-tab-status")).toHaveCount(0, { timeout: 15_000 });
  await app.close();
});

test("first run and every Settings page", async () => {
  test.skip(!SCREENSHOTS, "Only takes screenshots: set INCARNAMIND_SCREENSHOTS to a folder.");
  const { app, window } = await launchApp(dataDir);
  const setup = window.getByTestId("chat-setup");
  await expect(setup).toBeVisible();
  await screenshot(window, "first-run");
  await setup.getByTestId("chat-choice-api-key").check();
  await screenshot(window, "first-run-api-key");
  await setup.getByTestId("chat-setup-later").click();
  await expect(setup).toBeHidden();

  await window.getByRole("button", { name: "Settings" }).click();
  const settings = window.getByTestId("settings");
  for (const page of [
    "general",
    "chat-model",
    "search",
    "connectors",
    "skills",
    "approvals",
    "privacy",
  ]) {
    await settings.getByTestId(`settings-nav-${page}`).click();
    await expect(settings).toHaveAttribute("data-page", page);
    await expect(settings.getByTestId(`settings-page-${page}`)).not.toBeEmpty();
    await screenshot(window, `settings-${page}`);
  }
  await app.close();
});
