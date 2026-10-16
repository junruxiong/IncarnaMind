import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  enterContent,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/**
 * ⌘K: names, the registry's commands, and the words inside Minds and
 * Documents, by keyboard and from the sidebar's Search.
 */

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

const palette = (window: Page) => window.getByTestId("palette");
const group = (window: Page, id: string) =>
  palette(window).locator(`[data-testid="palette-group"][data-group="${id}"]`);

async function openPalette(window: Page) {
  await window.keyboard.press("ControlOrMeta+k");
  await expect(palette(window)).toBeVisible();
  await expect(window.getByTestId("palette-search")).toBeFocused();
}

/** A new Mind with these lines, named by its first. */
async function writeMind(window: Page, lines: string[]) {
  await newMind(window);
  await enterContent(window);
  for (const [index, line] of lines.entries()) {
    if (index > 0) await window.keyboard.press("Enter");
    await window.keyboard.type(line);
  }
}

/** Whether the focus is in a Mind's text. */
const focusInNote = (window: Page) =>
  window.evaluate(() => document.activeElement?.closest(".ProseMirror") !== null);

test("a word only inside a Mind is found, and so is a Chinese word, and Enter opens the Mind at it", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);
  await writeMind(window, [
    "Supplier review",
    "The pilot boarded near Rotterdam at dawn.",
    "本季度的财务报表显示利润增长。",
  ]);
  await writeMind(window, ["Stand-up notes", "Leeds account handover."]);

  await openPalette(window);
  await window.keyboard.type("Rotterdam");
  const notes = group(window, "notes");
  await expect(notes.getByTestId("palette-item")).toHaveCount(1);
  await expect(notes).toContainText("Supplier review");
  await expect(notes.locator("b")).toHaveText("Rotterdam");
  // Nothing is named so: names come first, and there are none.
  await expect(group(window, "minds")).toHaveCount(0);

  // Opened from the other Mind: Enter shows this one, its title in the front tab.
  await window.keyboard.press("Enter");
  await expect(palette(window)).toBeHidden();
  await expect(window.getByTestId("mind-title").locator("visible=true")).toHaveAttribute(
    "placeholder",
    "Supplier review",
  );
  await expect.poll(() => focusInNote(window)).toBe(true);
  // The cursor is in the line the words were found in.
  await expect
    .poll(() => window.evaluate(() => document.getSelection()?.anchorNode?.textContent ?? ""))
    .toContain("Rotterdam");

  await openPalette(window);
  await window.keyboard.type("财务");
  await expect(group(window, "notes")).toContainText("Supplier review");
  await expect(group(window, "notes")).not.toContainText("Stand-up notes");
});

test("the words inside a Mind are found as soon as they are typed", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);
  await writeMind(window, ["Harbour log", "Tides in Bergen"]);

  await openPalette(window);
  await window.keyboard.type("Bergen");
  await expect(group(window, "notes")).toContainText("Harbour log");
  await window.keyboard.press("Escape");
  await expect(palette(window)).toBeHidden();
  await expect.poll(() => focusInNote(window)).toBe(true);

  // Typing on where the cursor was, then searching at once.
  await window.keyboard.type(" and the moon over Ålesund");
  await openPalette(window);
  await window.keyboard.type("alesund");
  await expect(group(window, "notes")).toContainText("Harbour log");
  await expect(group(window, "notes").locator("b")).toHaveText("Ålesund");
});

test("names, commands and the words in Documents, all by keyboard; Esc closes and gives the focus back", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);
  const sources = join(dataDir, "sources");
  await mkdir(sources);
  const msa = join(sources, "Aldermoor MSA.txt");
  await writeFile(msa, "Either party may end this contract on ninety days' written notice.");
  await addDocuments(window, [msa]);
  await writeMind(window, ["Contract checklist"]);

  // From the sidebar's Search, by keyboard: the row takes the focus back.
  const search = window.getByTestId("sidebar-search");
  await expect(search).toContainText(/⌘K|Ctrl\+K/);
  await search.focus();
  await window.keyboard.press("Enter");
  await expect(palette(window)).toBeVisible();
  // Nothing typed: the Minds edited last, then every action.
  await expect(group(window, "recent")).toContainText("Contract checklist");
  await expect(group(window, "actions")).toBeVisible();
  await window.keyboard.press("Escape");
  await expect(palette(window)).toBeHidden();
  await expect(search).toBeFocused();

  // ⌘K again closes it.
  await openPalette(window);
  await window.keyboard.press("ControlOrMeta+k");
  await expect(palette(window)).toBeHidden();

  // ⌘↵ opens a Document in a tab of its own, beside the Mind's in its group.
  await openPalette(window);
  await window.keyboard.type("aldermoor");
  await expect(group(window, "documents")).toContainText("Aldermoor MSA");
  await window.keyboard.press("ControlOrMeta+Enter");
  await expect(palette(window)).toBeHidden();
  const groups = window.getByTestId("tab-group");
  await expect(groups).toHaveCount(1);
  await expect(groups.first().getByTestId("document-tab")).toHaveAttribute("aria-selected", "true");

  await openPalette(window);
  await window.keyboard.type("contract");
  await expect(group(window, "minds")).toContainText("Contract checklist");
  await expect(group(window, "documentText")).toContainText("Aldermoor MSA");
  await expect(group(window, "documentText").locator("b")).toHaveText("contract");
  // ↓ moves through every group in order, and wraps.
  const items = palette(window).getByTestId("palette-item");
  await expect(items.first()).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("ArrowUp");
  await expect(items.last()).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  await expect(palette(window)).toBeHidden();
  await expect(window.getByTestId("viewer-title")).toHaveText("Aldermoor MSA");

  // ">" for actions only, as in VS Code: a command from the registry, Settings at a page.
  await openPalette(window);
  await window.keyboard.type("> settings models");
  await expect(palette(window).getByTestId("palette-group")).toHaveCount(1);
  const action = group(window, "actions").getByTestId("palette-item").first();
  await expect(action).toContainText("Settings: Models");
  await window.keyboard.press("Enter");
  await expect(window.getByTestId("settings-page-title")).toHaveText("Models");
});

test("an empty Library offers its actions, and with nothing found it says so, in English and in Chinese, at a narrow and a wide window", async () => {
  const { app, window } = await launchApp(dataDir);
  await dismissChatSetup(window);
  await openPalette(window);
  await expect(group(window, "recent")).toHaveCount(0);
  await expect(group(window, "actions").getByTestId("palette-item").first()).toBeVisible();
  await window.keyboard.type("tides");
  await expect(window.getByTestId("palette-empty")).toHaveText("No matches in names or text");
  await window.keyboard.press("Escape");

  await writeMind(window, ["Tides in Bergen"]);

  for (const width of [900, 1600]) {
    await setWindowSize(app, window, width, 660);
    await openPalette(window);
    await window.keyboard.type("zqxjv");
    await expect(window.getByTestId("palette-empty")).toHaveText("No matches in names or text");
    const box = await palette(window).boundingBox();
    expect(box && box.x >= 0 && box.x + box.width <= width).toBe(true);
    await window.keyboard.press("Escape");
  }

  await window.evaluate(async () => {
    const bridge = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await bridge.updateSettings({ user: { language: "zh-CN" } });
  });
  await expect(window.getByTestId("sidebar-search")).toContainText("搜索");
  await openPalette(window);
  await window.keyboard.type("zqxjv");
  await expect(window.getByTestId("palette-empty")).toHaveText("名称和内容中都没有匹配项");
});

test("an action run from ⌘K is taken back with Undo", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);
  await writeMind(window, ["Tides in Bergen"]);

  await openPalette(window);
  await window.keyboard.type("citation style");
  const action = group(window, "actions").getByTestId("palette-item").first();
  await expect(action).toContainText("Citation style…");
  await window.keyboard.press("Enter");
  const picker = window.getByTestId("citation-style-picker");
  await expect(picker.getByTestId("citation-style-search")).toBeFocused();
  await window.keyboard.type("ieee");
  await expect(picker.getByTestId("citation-style-option").first()).toHaveText("IEEE");
  await window.keyboard.press("Enter");
  await expect(picker).toBeHidden();
  await expect(window.getByTestId("undo-message")).toContainText("IEEE");

  await pointAndClick(window, window.getByTestId("undo-action"));
  await openPalette(window);
  await window.keyboard.type("citation style");
  await window.keyboard.press("Enter");
  const current = window.locator('[data-testid="citation-style-option"][data-current="true"]');
  await expect(current).toHaveCount(1);
  await expect(current).not.toHaveText("IEEE");
});

/** Where the caret is: the text of the node it is in, and its offset there. */
const caret = (window: Page) =>
  window.evaluate(() => {
    const selection = document.getSelection();
    return {
      text: selection?.anchorNode?.textContent ?? "",
      offset: selection?.anchorOffset ?? -1,
      collapsed: selection?.isCollapsed ?? false,
    };
  });

test("⌘K in the note: with the caret it opens, and Esc puts the caret back; with text selected it edits the link", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 660);
  await dismissChatSetup(window);
  await writeMind(window, ["Harbour log", "Tides in Bergen at dawn"]);
  for (let step = 0; step < 8; step++) await window.keyboard.press("ArrowLeft");
  // ProseMirror takes in a caret the arrows moved on selectionchange, after the keys.
  await window.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );
  const before = await caret(window);
  expect(before).toMatchObject({ text: "Tides in Bergen at dawn", collapsed: true });

  await window.keyboard.press("ControlOrMeta+k");
  await expect(palette(window)).toBeVisible();
  await window.keyboard.press("Escape");
  await expect(palette(window)).toBeHidden();
  await expect.poll(() => focusInNote(window)).toBe(true);
  expect(await caret(window)).toEqual(before);
  await window.keyboard.type("X");
  const line = window.locator(".ProseMirror p").filter({ hasText: "Tides in" });
  await expect(line).toHaveText("Tides in BergenX at dawn");

  await slideSelect(window, line, "Bergen");
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("link-field")).toBeFocused();
  await expect(palette(window)).toBeHidden();
});

test("finds every kind by name, a Chinese name by its pinyin, a name with a typo, and the words in a conversation", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 760);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  const sources = join(dataDir, "sources");
  await mkdir(sources);
  const files = [join(sources, "合同.txt"), join(sources, "Rate card.txt")];
  await writeFile(files[0] ?? "", "试用期结束后，合同的通知期为三十天。");
  await writeFile(files[1] ?? "", "Cold-chain rates by carrier.");
  await addDocuments(window, files);
  await window.evaluate(async () => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    await core.createLibraryGroup({ name: "财务", description: "" });
    await core.createTag({ name: "Logistics" });
  });
  await writeMind(window, ["Contract checklist"]);
  await askInAssistant(window, "Which carriers end in 2027?");

  const finds = async (query: string, id: string, name: string) => {
    await openPalette(window);
    await window.keyboard.type(query);
    await expect(group(window, id).getByTestId("palette-item").first()).toContainText(name);
    await window.keyboard.press("Escape");
    await expect(palette(window)).toBeHidden();
  };
  await finds("caiwu", "foldersAndTags", "财务");
  await finds("cw", "foldersAndTags", "财务");
  await finds("hetong", "documents", "合同");
  await finds("contarct", "minds", "Contract checklist");
  await finds("logist", "foldersAndTags", "Logistics");
  await finds("rate card", "documents", "Rate card");
  // Words only in the scripted Answer.
  await finds("streams", "conversationText", "Which carriers end in 2027");

  // A conversation found by its name opens in the assistant.
  await pointAndClick(window, window.getByTestId("assistant-new-chat"));
  await openPalette(window);
  await window.keyboard.type("carriers 2027");
  const found = group(window, "conversations").getByTestId("palette-item").first();
  // Named by its first Question, cut as a title.
  await expect(found).toContainText("Which carriers end in 2027");
  await window.keyboard.press("Enter");
  await expect(
    window.getByTestId("assistant").getByTestId("assistant-question").last(),
  ).toContainText("Which carriers end in 2027?");
});

test("runs a command on the Documents selected in the sidebar, taken back with Undo", async () => {
  const { app, window } = await launchApp(dataDir);
  await setWindowSize(app, window, 1280, 760);
  await dismissChatSetup(window);
  const sources = join(dataDir, "sources");
  await mkdir(sources);
  const files = ["Alpha.txt", "Beta.txt", "Gamma.txt"].map((name) => join(sources, name));
  for (const file of files) await writeFile(file, `Notes in ${file}.`);
  await addDocuments(window, files);
  const rows = window.getByTestId("sidebar").getByTestId("document-list-item");
  await expect(rows).toHaveCount(3);
  await pointAndClick(window, rows.nth(0).getByTestId("open-document"));
  await pointAndClick(window, rows.nth(2).getByTestId("open-document"), {
    modifiers: ["ControlOrMeta"],
  });
  await expect(rows.nth(2).getByRole("treeitem")).toHaveAttribute("aria-selected", "true");

  await openPalette(window);
  await window.keyboard.type("delete");
  const action = group(window, "actions").getByTestId("palette-item").first();
  await expect(action).toContainText("Delete 2 Documents");
  await window.keyboard.press("Enter");
  await expect(rows).toHaveCount(1);
  await pointAndClick(window, window.getByTestId("undo-action"));
  await expect(rows).toHaveCount(3);
});

test("with a sentence selected, ⌘K finds the one check by check, find support, 核对 and 出处, and the checks say what each is for", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 760);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await writeMind(window, ["Harbours publish the times of high water every year."]);
  const line = window.locator(".ProseMirror p").filter({ hasText: "Harbours publish" });
  const row = (key: string) => palette(window).locator(`[data-key="command:${key}"]`);

  // The selection stays in the Mind while the palette opens from the sidebar.
  const search = async (query: string) => {
    await slideSelect(window, line, "Harbours publish");
    await pointAndClick(window, window.getByTestId("sidebar-search"));
    await expect(window.getByTestId("palette-search")).toBeFocused();
    await window.keyboard.type(query);
  };

  for (const query of ["check", "find support"]) {
    await search(query);
    await expect(row("check.againstDocuments")).toHaveText("Check against Documents");
    await expect(palette(window).locator('[data-key^="command:check."]')).toHaveCount(1);
    if (query === "check") {
      await expect(row("mind.checkAll").getByTestId("palette-hint")).toHaveText(
        "Every claim against your Documents",
      );
      await expect(row("mind.checkContradictions").getByTestId("palette-hint")).toHaveText(
        "Where your Documents disagree with this Mind's Citations",
      );
      const verify = row("verify.document");
      await expect(verify.getByTestId("palette-hint")).toHaveText(
        "Check the claims in a PDF someone else wrote",
      );
      // The hint sits under the name and inside the row.
      const rowBox = await verify.boundingBox();
      const hintBox = await verify.getByTestId("palette-hint").boundingBox();
      const nameBox = await verify.locator("span.truncate").first().boundingBox();
      expect(hintBox?.y ?? 0).toBeGreaterThanOrEqual((nameBox?.y ?? 0) + (nameBox?.height ?? 0));
      expect((hintBox?.x ?? 0) + (hintBox?.width ?? 0)).toBeLessThanOrEqual(
        (rowBox?.x ?? 0) + (rowBox?.width ?? 0),
      );
    }
    await window.keyboard.press("Escape");
    await expect(palette(window)).toBeHidden();
  }

  await setLanguage(window, "zh-CN");
  for (const query of ["核对", "出处"]) {
    await search(query);
    await expect(row("check.againstDocuments")).toHaveText("对照文档核对");
    await expect(palette(window).locator('[data-key^="command:check."]')).toHaveCount(1);
    if (query === "核对") {
      await expect(row("mind.checkAll").getByTestId("palette-hint")).toHaveText(
        "每个说法都对照你的文档",
      );
      await expect(row("verify.document").getByTestId("palette-hint")).toHaveText(
        "核对别人写的 PDF 里的说法",
      );
    }
    await window.keyboard.press("Escape");
    await expect(palette(window)).toBeHidden();
  }

  await search("矛盾");
  await expect(row("mind.checkContradictions").getByTestId("palette-hint")).toHaveText(
    "你的文档与这个 Mind 的引用不一致的地方",
  );
});

test("with nothing named so, “Ask about …” asks in the assistant; the assistant's commands are in the list", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await setWindowSize(app, window, 1280, 760);
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await writeMind(window, ["Supplier review"]);

  await openPalette(window);
  await window.keyboard.type("dock left");
  await expect(group(window, "actions")).toContainText("Assistant:");
  await window.keyboard.press("Escape");

  await openPalette(window);
  const question = "how long is the warranty on the chillers";
  await window.keyboard.type(question);
  await expect(window.getByTestId("palette-empty")).toHaveText("No matches in names or text");
  const ask = group(window, "ask").getByTestId("palette-item");
  await expect(ask).toContainText(`Ask about “${question}”`);
  await expect(ask).toHaveAttribute("aria-selected", "true");
  await window.keyboard.press("Enter");
  await expect(palette(window)).toBeHidden();
  const assistant = window.getByTestId("assistant");
  await expect(assistant.getByTestId("assistant-question").last()).toContainText(question);
  await expect(assistant.getByTestId("assistant-answer").last()).toHaveAttribute(
    "data-status",
    "done",
    { timeout: 15_000 },
  );
});
