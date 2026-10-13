import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { type ElectronApplication, expect, type Locator, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  coreOf,
  createDataFolder,
  dismissChatSetup,
  editFolderInLibrary,
  launchApp,
  linkedFolderRow,
  linkFolderFromSidebar,
  motionSettled,
  newMind,
  openLinkedFolderMenu,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  showSourceLocations,
  sidebarFolder,
} from "./app";

/*
 * Undo, the remaining gaps (#606): a Mind made from a Document, in a Folder, with ⌘N or the
 * sidebar's + is taken back by ⌘Z to Recently deleted and made again by ⇧⌘Z; a run of typing in
 * a multi-line field is one step; pausing a Linked folder and changing its layout are taken
 * back. Each is checked in the core, through the bridge. Set INCARNAMIND_SCREENSHOTS to a folder
 * to also save screenshots.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (!SCREENSHOTS) return;
  await motionSettled(window);
  await mkdir(SCREENSHOTS, { recursive: true });
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

let dataDir: string;
let sources: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  sources = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(sources);
});

/** Reads the core through the bridge. */
async function read<T>(window: Page, query: (core: CoreBridge) => Promise<T>): Promise<T> {
  return (await coreOf(window)).evaluate(query);
}

const undoMessage = (window: Page) => window.getByTestId("undo-message");
const undo = (window: Page) => window.keyboard.press("ControlOrMeta+z");
const redo = (window: Page) => window.keyboard.press("ControlOrMeta+Shift+z");

/** Clicks the empty foot of the sidebar's tree, where ⌘Z goes to the app's changes. */
async function leaveText(window: Page): Promise<void> {
  const box = await window.getByTestId("sidebar-tree").boundingBox();
  if (!box) throw new Error("The sidebar isn't showing.");
  await window.mouse.move(box.x + box.width / 2, box.y + box.height - 8, { steps: 8 });
  await window.mouse.down();
  await window.mouse.up();
}

const mindIds = (window: Page) =>
  read(window, async (core) => (await core.listMinds()).map((mind) => mind.id).sort());
const deletedMindIds = (window: Page) =>
  read(window, async (core) =>
    (await core.listDeleted()).filter((item) => item.kind === "mind").map((item) => item.id),
  );

/** Runs a command that makes a Mind, and gives the Mind it made. */
async function madeBy(window: Page, make: () => Promise<void>, line: string): Promise<string> {
  const before = await mindIds(window);
  await make();
  await expect(undoMessage(window)).toHaveText(line);
  await expect.poll(async () => (await mindIds(window)).length).toBe(before.length + 1);
  const made = (await mindIds(window)).find((id) => !before.includes(id));
  if (!made) throw new Error("No Mind was made.");
  return made;
}

/** ⌘Z sends the Mind to Recently deleted and ⇧⌘Z makes it again, twice over, with the line each time. */
async function takenBackAndMadeAgain(window: Page, made: string, line: string): Promise<void> {
  const withIt = await mindIds(window);
  for (let round = 0; round < 2; round++) {
    await leaveText(window);
    await undo(window);
    await expect.poll(() => mindIds(window)).toEqual(withIt.filter((id) => id !== made));
    expect(await deletedMindIds(window)).toContain(made);
    await redo(window);
    await expect.poll(() => mindIds(window)).toEqual(withIt);
    expect(await deletedMindIds(window)).not.toContain(made);
    await expect(undoMessage(window)).toHaveText(line);
  }
}

const documentRow = (window: Page, name: string) =>
  window.getByTestId("document-list-item").filter({ hasText: name });

async function documentMenuCommand(window: Page, name: string, command: string): Promise<void> {
  await pointAndClick(window, documentRow(window, name).getByTestId("document-file-menu"));
  await pointAndClick(
    window,
    window.getByTestId("document-file-actions").filter({ visible: true }).getByTestId(command),
  );
}

async function startWithADocument(language: "en" | "zh-CN" = "en") {
  const running = await launchApp(dataDir);
  const { window } = running;
  await dismissChatSetup(window);
  await writeFile(join(sources, "Tides.txt"), "Neap tides are the smallest of the month.\n");
  await addDocuments(window, [join(sources, "Tides.txt")]);
  await read(window, (core) => core.createLibraryGroup({ name: "Finance", description: "" }));
  if (language !== "en") await setLanguage(window, language);
  return running;
}

test("Ask about a Document with no Mind to ask in, Start a Mind from it, New Mind here, ⌘N and +: each is taken back to Recently deleted by ⌘Z and made again by ⇧⌘Z, twice over", async () => {
  const { window } = await startWithADocument();
  const made = "Made a new Mind";

  // With no Mind, asking about a Document makes one.
  const asked = await madeBy(
    window,
    () => documentMenuCommand(window, "Tides", "document.ask"),
    made,
  );
  await takenBackAndMadeAgain(window, asked, made);

  const started = await madeBy(
    window,
    () => documentMenuCommand(window, "Tides", "document.startMind"),
    "Started “Tides”",
  );
  await takenBackAndMadeAgain(window, started, "Started “Tides”");

  const finance = sidebarFolder(window, "Finance");
  const here = await madeBy(
    window,
    async () => {
      await finance.getByTestId("folder-row").click({ button: "right" });
      await finance.getByTestId("folder.newMind").click();
    },
    made,
  );
  expect(await read(window, async (core) => (await core.listMinds()).length)).toBe(3);
  await takenBackAndMadeAgain(window, here, made);

  const byKeys = await madeBy(window, () => window.keyboard.press("ControlOrMeta+n"), made);
  await takenBackAndMadeAgain(window, byKeys, made);

  const byPlus = await madeBy(window, () => newMind(window), made);
  await takenBackAndMadeAgain(window, byPlus, made);
});

const LINES = {
  en: { started: "Started “Tides”", made: "Made a new Mind" },
  "zh-CN": { started: "已新建“Tides”", made: "已新建 Mind" },
} as const;

for (const language of ["en", "zh-CN"] as const) {
  test(`the line of a Mind made holds at 900 and 1,600px, in ${language}`, async () => {
    const { app, window } = await startWithADocument(language);
    const lines = LINES[language];
    await documentMenuCommand(window, "Tides", "document.startMind");
    await expect(undoMessage(window)).toHaveText(lines.started);
    await screenshot(window, `start-a-mind-${language}-1600`);
    await setWindowSize(app, window, 900, 660);
    await leaveText(window);
    await undo(window);
    await redo(window);
    await expect(undoMessage(window)).toHaveText(lines.started);
    await expect(window.getByTestId("undo-line")).toBeInViewport();
    await screenshot(window, `start-a-mind-${language}-900`);

    const finance = sidebarFolder(window, "Finance");
    await finance.getByTestId("folder-row").click({ button: "right" });
    await finance.getByTestId("folder.newMind").click();
    await expect(undoMessage(window)).toHaveText(lines.made);
    await leaveText(window);
    await undo(window);
    await expect(window.getByTestId("sidebar").getByTestId("mind-row")).toHaveCount(1);
  });
}

test("⌘N twice in a row makes two Minds, and ⌘Z twice and ⇧⌘Z twice take them back and make them again, one at a time", async () => {
  const { window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await window.keyboard.press("ControlOrMeta+n");
  await window.keyboard.press("ControlOrMeta+n");
  await expect.poll(() => mindIds(window)).toHaveLength(2);
  const both = await mindIds(window);

  await leaveText(window);
  await undo(window);
  await expect.poll(() => mindIds(window)).toHaveLength(1);
  await undo(window);
  await expect.poll(() => mindIds(window)).toHaveLength(0);
  expect((await deletedMindIds(window)).sort()).toEqual(both);
  await redo(window);
  await expect.poll(() => mindIds(window)).toHaveLength(1);
  await redo(window);
  await expect.poll(() => mindIds(window)).toEqual(both);
  expect(await deletedMindIds(window)).toHaveLength(0);
});

test("a Mind made, then written in, goes to Recently deleted with its text, and comes back with it", async () => {
  const { window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await newMind(window);
  await window.getByTestId("mind-title").fill("Ferries");
  await window.getByTestId("mind-title").press("Enter");
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, editor);
  await window.keyboard.type("Ferries run at noon");
  await expect(editor).toContainText("Ferries run at noon");

  // The rename is the last step, then the Mind: the text goes with it, and is kept.
  await leaveText(window);
  await undo(window);
  await expect.poll(() => mindIds(window)).toHaveLength(1);
  await undo(window);
  await expect.poll(() => mindIds(window)).toHaveLength(0);
  expect(await deletedMindIds(window)).toHaveLength(1);
  await redo(window);
  await redo(window);
  await expect.poll(() => mindIds(window)).toHaveLength(1);
  await expect(window.getByTestId("mind-title")).toHaveValue("Ferries");
  await expect(window.getByTestId("mind-editor")).toContainText("Ferries run at noon");
});

/** A word typed in one go in a multi-line field is one step: ⌘Z gives back the text as it was, ⇧⌘Z the word. */
async function typedWordIsOneStep(window: Page, field: Locator, before: string): Promise<void> {
  await pointAndClick(window, field);
  await window.keyboard.press("ControlOrMeta+End");
  await window.keyboard.type("tides");
  await expect(field).toHaveValue(`${before}tides`);
  await undo(window);
  await expect(field).toHaveValue(before);
  await redo(window);
  await expect(field).toHaveValue(`${before}tides`);
  await undo(window);
  await expect(field).toHaveValue(before);
}

test("typing in a multi-line field is one step: a form in a dialog, and a form in a page", async () => {
  const { window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await writeFile(join(sources, "Tides.txt"), "Neap tides are the smallest of the month.\n");
  await addDocuments(window, [join(sources, "Tides.txt")]);
  await read(window, (core) => core.createLibraryGroup({ name: "Finance", description: "Rent" }));

  // A dialog: the addresses of web pages.
  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-add-web-page").click();
  const dialog = window.getByTestId("add-web-page-dialog");
  await typedWordIsOneStep(window, dialog.getByTestId("add-web-page-addresses"), "");
  await window.keyboard.press("Escape");
  await expect(dialog).toBeHidden();

  // A page: a Folder's description in the Library.
  const form = await editFolderInLibrary(window, "Finance");
  const description = form.locator("textarea");
  await typedWordIsOneStep(window, description, "Rent");
});

test("Chinese typed by an input method into a multi-line field is one step too", async () => {
  const { window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await window.getByTestId("plus-menu").click();
  await window.getByTestId("plus-add-web-page").click();
  const addresses = window.getByTestId("add-web-page-dialog").getByTestId("add-web-page-addresses");
  await pointAndClick(window, addresses);
  const session = await window.context().newCDPSession(window);
  try {
    await session.send("Input.imeSetComposition", {
      text: "chao",
      selectionStart: 4,
      selectionEnd: 4,
    });
    await session.send("Input.imeSetComposition", {
      text: "chaoxi",
      selectionStart: 6,
      selectionEnd: 6,
    });
    await session.send("Input.insertText", { text: "潮汐" });
  } finally {
    await session.detach();
  }
  await expect(addresses).toHaveValue("潮汐");
  await undo(window);
  await expect(addresses).toHaveValue("");
  await redo(window);
  await expect(addresses).toHaveValue("潮汐");
});

async function linkPapers(window: Page, app: ElectronApplication) {
  const papers = join(sources, "Papers");
  await mkdir(papers, { recursive: true });
  await writeFile(join(papers, "Attention.txt"), "Attention lets a model focus.\n");
  await writeFile(join(papers, "Tides.txt"), "Neap tides are the smallest of the month.\n");
  await linkFolderFromSidebar(app, window, papers);
  await showSourceLocations(window);
  const linked = () => read(window, async (core) => (await core.listLinkedFolders())[0]);
  await expect.poll(async () => (await linked())?.status).toBe("watching");
  return { row: linkedFolderRow(window, "Papers"), linked };
}

test("pausing a Linked folder and showing it as a flat list: ⌘Z sets it back as it was, ⇧⌘Z does it again, with the line in English and Chinese", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  const { row, linked } = await linkPapers(window, app);

  await (await openLinkedFolderMenu(row)).getByTestId("linkedFolder.pause").click();
  await expect(undoMessage(window)).toHaveText("Paused indexing in “Papers”");
  await screenshot(window, "pause-a-linked-folder-en-1600");
  expect((await linked())?.status).toBe("paused");
  await setWindowSize(app, window, 900, 660);
  await leaveText(window);
  await undo(window);
  await expect.poll(async () => (await linked())?.status).toBe("watching");
  await redo(window);
  await expect.poll(async () => (await linked())?.status).toBe("paused");
  await expect(undoMessage(window)).toHaveText("Paused indexing in “Papers”");
  await screenshot(window, "pause-a-linked-folder-en-900");
  await undo(window);
  await expect.poll(async () => (await linked())?.status).toBe("watching");

  await (await openLinkedFolderMenu(row)).getByTestId("linkedFolder.layout").click();
  await expect(undoMessage(window)).toHaveText("Showing “Papers” as a flat list");
  expect((await linked())?.layout).toBe("flat");
  await leaveText(window);
  await undo(window);
  await expect.poll(async () => (await linked())?.layout).toBe("tree");
  await redo(window);
  await expect.poll(async () => (await linked())?.layout).toBe("flat");
  await undo(window);
  await expect.poll(async () => (await linked())?.layout).toBe("tree");

  await setLanguage(window, "zh-CN");
  await (await openLinkedFolderMenu(row)).getByTestId("linkedFolder.pause").click();
  await expect(undoMessage(window)).toHaveText("已暂停“Papers”的索引");
  await screenshot(window, "pause-a-linked-folder-zh-900");
  await setWindowSize(app, window, 1600, 660);
  await leaveText(window);
  await undo(window);
  await redo(window);
  await expect(undoMessage(window)).toHaveText("已暂停“Papers”的索引");
  await screenshot(window, "pause-a-linked-folder-zh-1600");
  await undo(window);
  await expect.poll(async () => (await linked())?.status).toBe("watching");
  await (await openLinkedFolderMenu(row)).getByTestId("linkedFolder.layout").click();
  await expect(undoMessage(window)).toHaveText("已将“Papers”显示为平铺列表");
});

test("quitting between the change and its Undo leaves the change made, and ⌘Z in the next launch takes back nothing", async () => {
  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  const { row } = await linkPapers(first.window, first.app);
  await (await openLinkedFolderMenu(row)).getByTestId("linkedFolder.pause").click();
  await expect(undoMessage(first.window)).toHaveText("Paused indexing in “Papers”");
  await newMind(first.window);
  await expect(undoMessage(first.window)).toHaveText("Made a new Mind");
  await first.app.close();

  const { window } = await launchApp(dataDir);
  await showSourceLocations(window);
  expect(await mindIds(window)).toHaveLength(1);
  await expect
    .poll(() => read(window, async (core) => (await core.listLinkedFolders())[0]?.status))
    .toBe("paused");
  await leaveText(window);
  await undo(window);
  await expect(undoMessage(window)).toHaveCount(0);
  expect(await mindIds(window)).toHaveLength(1);
  expect(await deletedMindIds(window)).toHaveLength(0);
  expect((await read(window, (core) => core.listLinkedFolders()))[0]?.status).toBe("paused");
});
