import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CoreBridge } from "../src/core/api";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  addToNote,
  askInAssistant,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/*
 * Edit with AI (#93): select text, ask for a change from the selection menu
 * or with ⌘J, and review what comes back as changes in the note, each with
 * its own bar, by mouse and keyboard, with Undo. The scripted model (see
 * src/main/fakeChatModel.ts) shortens by dropping "very" and "really" (非常),
 * and does what "replace X with Y" (把 X 换成 Y) says. Set
 * INCARNAMIND_SCREENSHOTS to a folder to also save screenshots.
 *
 * A change whose words the User edits goes out of date (#94): it can't be
 * accepted, and Regenerate makes it again from their words.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;
const screenshot = async (window: Page, name: string) => {
  if (SCREENSHOTS) await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
};

const TIDES = buildPdf([
  { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
  { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
]);

const COMMAND = process.platform === "darwin" ? "Command" : "Control";

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

const setLanguage = (window: Page, language: "en" | "zh-CN") =>
  window.evaluate(
    (value) =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.updateSettings({
        user: { language: value },
      }),
    language,
  );

/** Selects a line's `words` with the mouse, then `more` characters after them with the keyboard. */
async function selectWords(window: Page, line: Locator, words: string, more = 0) {
  await slideSelect(window, line, words);
  for (let index = 0; index < more; index++) await window.keyboard.press("Shift+ArrowRight");
}

test("Shorten comes back as changes with their own bars: one accepted by mouse, one rejected by keyboard, Undo, and the Citation is still checked", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  // An Answer with a checked Citation, added to the note under a line of the User's.
  await window.keyboard.type("Tides at the harbour.");
  await window.keyboard.press("Enter");
  const answer = await askInAssistant(window, "When do spring tides happen?");
  await addToNote(window, answer);
  const note = editor.locator(":scope > p").nth(1);
  await expect(note.getByTestId("citation")).toHaveAttribute("data-check", "found");
  await slideSelect(window, note, "Your Documents answer this");
  await window.keyboard.type("The tides here are very strong and really regular");
  await expect(note).toContainText("The tides here are very strong and really regular");

  // The Answer's last line, under it, taken out as a person would: then nothing follows the line.
  const last = editor.locator(":scope > p").last();
  await expect(last).toHaveText("That is all.");
  await slideSelect(window, last, "That is all.");
  await window.keyboard.press("Backspace");
  await window.keyboard.press("Backspace");
  await expect(editor.locator(":scope > p")).toHaveCount(2);

  // Selected, the menu offers the AI's actions; Shorten comes back as two changes.
  // The words, then the space, the Citation and the full stop after them.
  await selectWords(window, note, "The tides here are very strong and really regular", 3);
  const menu = window.getByTestId("selection-menu");
  await expect(menu.getByTestId("selection-action-shorten")).toBeVisible();
  await expect(menu.getByTestId("selection-action-edit")).toContainText("Edit with AI…");
  await screenshot(window, "edit-menu-en");
  await pointAndClick(window, menu.getByTestId("selection-action-shorten"));

  const bar = window.getByTestId("proposals-bar");
  await expect(bar.getByTestId("proposals-count")).toHaveText("2 changes from the assistant");
  const removed = note.locator("del.proposal-delete");
  await expect(removed).toHaveText(["very ", "really "]);
  await expect(removed.first()).toHaveCSS("text-decoration-line", "line-through");
  // Nothing is applied yet: the words are still there, struck.
  await expect(note).toContainText("very strong and really regular");
  // The cursor went into the first change: its bar, and what a screen reader hears.
  const changeBar = window.getByTestId("change-bar");
  await expect(changeBar).toBeVisible();
  await expect(changeBar).toContainText("Change 1 of 2");
  await expect(window.getByTestId("proposal-spoken")).toHaveText(
    `Suggested change: removes “very”. Accept with ${COMMAND} Enter, reject with ${COMMAND} Backspace.`,
  );
  await screenshot(window, "edit-changes-en");
  // Its bar sits over its line.
  const barBox = await changeBar.boundingBox();
  const wordBox = await removed.first().boundingBox();
  expect((barBox?.y ?? 0) + (barBox?.height ?? 0)).toBeLessThanOrEqual((wordBox?.y ?? 0) + 1);

  // Accept by mouse.
  await pointAndClick(window, changeBar.getByTestId("change-accept"));
  await expect(removed).toHaveText(["really "]);
  await expect(note).toContainText("The tides here are strong and really regular");
  await expect(bar.getByTestId("proposals-count")).toHaveText("1 change from the assistant");
  await expect(window.getByTestId("undo-message")).toHaveText("Accepted 1 change");

  // Reject by keyboard, with the cursor in the other change.
  await pointAndClick(window, removed.first());
  await expect(changeBar).toContainText("Change 1 of 1");
  // The change the cursor is in is washed.
  await expect(note.locator(".proposal-current")).toHaveText("really ");
  await window.keyboard.press("ControlOrMeta+Backspace");
  await expect(removed).toHaveCount(0);
  await expect(bar).toHaveCount(0);
  await expect(note).toContainText("The tides here are strong and really regular");

  // Undo brings the rejected change back; Accept all takes it.
  await window.keyboard.press("ControlOrMeta+z");
  await expect(removed).toHaveText(["really "]);
  await expect(bar.getByTestId("proposals-count")).toHaveText("1 change from the assistant");
  await expect(changeBar).toContainText("Change 1 of 1");
  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(bar).toHaveCount(0);
  await expect(note).toContainText("The tides here are strong and regular");
  // The Citation in the sentence is kept, and its quote still found.
  await expect(note.getByTestId("citation")).toHaveCount(1);
  await expect(note.getByTestId("citation")).toHaveAttribute("data-check", "found");
});

test("a whole Mind restructured with ⌘A and ⌘J keeps its headings, lists and Citation through Accept all, and Undo takes it back in one step (#703)", async () => {
  await writeFile(join(sources, "Tides.pdf"), TIDES);
  const { window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await addDocuments(window, [join(sources, "Tides.pdf")]);
  const tides = await window.evaluate(async () => {
    const documents = await (
      globalThis as unknown as { incarnamind: CoreBridge }
    ).incarnamind.listDocuments();
    const found = documents.find((each) => each.name.startsWith("Tides"));
    return found && { id: found.id, name: found.name, contentHash: found.contentHash };
  });
  if (!tides) throw new Error("Tides.pdf wasn't added.");
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();

  // Findings as bullets, one with a Citation; Decisions as numbered items; a closing line.
  const paragraphOf = (...content: object[]) => ({ type: "paragraph", content });
  const words = (text: string) => ({ type: "text", text });
  const items = (...lines: object[][]) =>
    lines.map((content) => ({ type: "listItem", content: [paragraphOf(...content)] }));
  await editor.evaluate(
    (dom, blocks) => {
      const { editor: tiptap } = dom as unknown as { editor: Editor };
      tiptap.commands.setContent(blocks);
    },
    [
      { type: "heading", attrs: { level: 2 }, content: [words("Findings")] },
      {
        type: "bulletList",
        content: items(
          [
            words("Spring tides happen at new moon. "),
            {
              type: "citation",
              attrs: {
                passageId: `passage-${tides.id}`,
                documentId: tides.id,
                documentName: tides.name,
                contentHash: tides.contentHash,
                pageFrom: 2,
                pageTo: 2,
                quote: "Spring tides happen at new moon and at full moon",
                check: "found",
              },
            },
          ],
          [words("Neap tides are the weakest.")],
        ),
      },
      { type: "heading", attrs: { level: 2 }, content: [words("Decisions")] },
      {
        type: "orderedList",
        content: items(
          [words("Publish the tide tables in March.")],
          [words("Check the harbour gauges weekly.")],
        ),
      },
      paragraphOf(words("Harbours publish the times of high water every year.")),
    ],
  );
  const headings = editor.locator(":scope > h2");
  const bullets = editor.locator(":scope > ul > li");
  const numbered = editor.locator(":scope > ol > li");
  await expect(headings).toHaveText(["Findings", "Decisions"]);

  // The whole Mind, then what to change.
  await editor.click();
  await window.keyboard.press("ControlOrMeta+a");
  await window.keyboard.press("ControlOrMeta+j");
  await expect(window.getByTestId("composer-input")).toBeFocused();
  await window.keyboard.type("Restructure so the decisions come first. Cut the last paragraph.");
  await window.keyboard.press("Enter");
  const bar = window.getByTestId("proposals-bar");
  await expect(bar.getByTestId("proposals-count")).toContainText("changes from the assistant");
  await screenshot(window, "edit-restructure-changes-en");

  // Accept all: decisions first, each Block still the kind it was, the Citation still found.
  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(bar).toHaveCount(0);
  await expect(headings).toHaveText(["Decisions", "Findings"]);
  await expect(numbered).toHaveText([
    "Publish the tide tables in March.",
    "Check the harbour gauges weekly.",
  ]);
  await expect(bullets).toHaveCount(2);
  await expect(bullets.first()).toContainText("Spring tides happen at new moon.");
  await expect(bullets.first().getByTestId("citation")).toHaveAttribute("data-check", "found");
  await expect(bullets.last()).toHaveText("Neap tides are the weakest.");
  await expect(editor.locator(":scope > p")).not.toContainText(["Harbours publish"]);
  const order = await editor.evaluate((dom) =>
    [...dom.children].map((child) => `${child.tagName}:${child.textContent?.slice(0, 12)}`),
  );
  // An empty paragraph the editor keeps at the end doesn't count.
  expect(order.filter((each) => each !== "P:")).toEqual([
    "H2:Decisions",
    "OL:Publish the ",
    "H2:Findings",
    "UL:Spring tides",
  ]);
  await screenshot(window, "edit-restructure-accepted-en");

  // One Undo brings every change back; Reject all leaves the Mind as it was.
  await window.keyboard.press("ControlOrMeta+z");
  await expect(bar.getByTestId("proposals-count")).toContainText("changes from the assistant");
  await pointAndClick(window, bar.getByTestId("proposals-reject-all"));
  await expect(bar).toHaveCount(0);
  await expect(headings).toHaveText(["Findings", "Decisions"]);
  await expect(bullets).toHaveCount(2);
  await expect(bullets.first().getByTestId("citation")).toHaveAttribute("data-check", "found");
  await expect(editor.locator(":scope > p").last()).toHaveText(
    "Harbours publish the times of high water every year.",
  );
});

test("⌘J with text selected says what to change in the composer, in Chinese; ⌘↵ accepts, Undo, and Reject all", async () => {
  const { app, window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await setLanguage(window, "zh-CN");
  await setWindowSize(app, window, 900, 660);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("潮水每天非常规律地涨落两次。");
  const note = editor.locator(":scope > p").first();

  // ⌘J with the line selected: the composer quotes it and asks what to change.
  await selectWords(window, note, "潮水每天非常规律地涨落两次。");
  await window.keyboard.press("ControlOrMeta+j");
  const composer = window.getByTestId("composer");
  const input = composer.getByTestId("composer-input");
  await expect(input).toBeFocused();
  await expect(composer.getByTestId("composer-edit-quote")).toContainText("正在编辑“潮水每天");
  await expect(input).toHaveAttribute("placeholder", "说说要怎么改…");
  await expect(note.locator(".proposal-target")).toHaveText("潮水每天非常规律地涨落两次。");
  await window.keyboard.type("把规律换成准时");
  await screenshot(window, "edit-composer-zh");
  await window.keyboard.press("Enter");

  const bar = window.getByTestId("proposals-bar");
  await expect(bar.getByTestId("proposals-count")).toHaveText("来自助手的 1 处修订");
  await expect(note.locator("del.proposal-delete")).toHaveText("规律");
  await expect(note.locator("ins.proposal-insert")).toHaveText("准时");
  await expect(note.locator("ins.proposal-insert")).toHaveCSS("text-decoration-line", "underline");
  const changeBar = window.getByTestId("change-bar");
  await expect(changeBar).toContainText("第 1 处修订，共 1 处");
  await expect(changeBar.getByTestId("change-accept")).toHaveText("接受");
  await expect(window.getByTestId("proposal-spoken")).toContainText(
    "建议的修订：将“规律”改为“准时”",
  );
  await screenshot(window, "edit-changes-zh");
  // The bar fits in the narrow window.
  const box = await changeBar.boundingBox();
  expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual(900);

  // ⌘↵ in the change accepts it; Undo brings it back; Reject all leaves the text as it was.
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+Enter");
  await expect(note).toHaveText("潮水每天非常准时地涨落两次。");
  await expect(bar).toHaveCount(0);
  await window.keyboard.press("ControlOrMeta+z");
  await expect(note.locator("ins.proposal-insert")).toHaveText("准时");
  await pointAndClick(window, bar.getByTestId("proposals-reject-all"));
  await expect(note).toHaveText("潮水每天非常规律地涨落两次。");
  await expect(window.getByTestId("undo-message")).toHaveText("已拒绝 1 处修订");

  // Shorten from the menu, in Chinese: 非常 goes.
  await selectWords(window, note, "潮水每天非常规律地涨落两次。");
  await pointAndClick(window, window.getByTestId("selection-action-shorten"));
  await expect(note.locator("del.proposal-delete")).toHaveText("非常");
  await window.keyboard.press("ControlOrMeta+Enter");
  await expect(note).toHaveText("潮水每天规律地涨落两次。");

  // Translate… asks for the language beside it. A text the model gives back as it was (the
  // scripted model only capitalises) says so in the composer, and changes nothing.
  await selectWords(window, note, "潮水每天规律地涨落两次。");
  await pointAndClick(window, window.getByTestId("selection-action-translate"));
  const languages = window.getByTestId("selection-choices-translate");
  await expect(languages).toBeVisible();
  await pointAndClick(window, languages.getByTestId("selection-choice-en"));
  const problem = window.getByTestId("edit-problem");
  await expect(problem).toContainText("没有要修改的地方");
  await expect(bar).toHaveCount(0);
  await pointAndClick(window, problem.getByTestId("edit-problem-close"));
  await expect(problem).toHaveCount(0);
});

test("a double click accepts a change once; a change whose words the User edits is out of date until Regenerate makes it again", async () => {
  const { window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  await window.keyboard.type("The tides are really very strong. They are very high.");
  const note = editor.locator(":scope > p").first();

  await selectWords(window, note, "The tides are really very strong. They are very high.");
  await pointAndClick(window, window.getByTestId("selection-action-shorten"));
  const bar = window.getByTestId("proposals-bar");
  const count = bar.getByTestId("proposals-count");
  await expect(count).toHaveText("2 changes from the assistant");
  const removed = note.locator("del.proposal-delete");
  await expect(removed).toHaveText(["really very ", "very "]);

  // Double-clicking Accept applies the change once: one step, which one Undo takes back.
  await pointAndClick(window, removed.nth(1));
  const changeBar = window.getByTestId("change-bar");
  await expect(changeBar).toContainText("Change 2 of 2");
  const accept = changeBar.getByTestId("change-accept");
  const box = await accept.boundingBox();
  if (!box) throw new Error("Accept isn't visible.");
  await window.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 8 });
  await window.mouse.dblclick(box.x + box.width / 2, box.y + box.height / 2);
  await expect(count).toHaveText("1 change from the assistant");
  await expect(removed).toHaveText(["really very "]);
  await expect(note).toContainText("They are high.");
  await expect(window.getByTestId("undo-message")).toHaveText("Accepted 1 change");
  await editor.focus();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(count).toHaveText("2 changes from the assistant");
  await expect(removed).toHaveText(["really very ", "very "]);

  // The User writes among the words the first change removes: they are theirs now.
  await editor.evaluate((dom) => {
    const { view, commands } = (dom as unknown as { editor: Editor }).editor;
    let at = -1;
    view.state.doc.descendants((node, pos) => {
      if (at < 0 && node.isText && node.text === "really very ") at = pos + "really ".length;
    });
    commands.setTextSelection(at);
  });
  await window.keyboard.type("quite ");
  await expect(note).toContainText("really quite very strong");
  // "Out of date · Regenerate", on a line under its paragraph.
  const outOfDate = window.getByTestId("change-out-of-date");
  await expect(outOfDate).toHaveCount(1);
  await expect(outOfDate).toContainText("Out of date");
  await expect(outOfDate.getByTestId("change-regenerate")).toHaveText("Regenerate");
  const lineBox = await outOfDate.boundingBox();
  const noteBox = await note.boundingBox();
  expect(lineBox?.y ?? 0).toBeGreaterThanOrEqual((noteBox?.y ?? 0) + (noteBox?.height ?? 0) - 1);
  // It can't be accepted: its line is its only bar, and ⌘↵ leaves it, saying why.
  await expect(changeBar).toHaveCount(0);
  await window.keyboard.press("ControlOrMeta+Enter");
  await expect(window.getByTestId("proposal-spoken")).toHaveText(
    "This change is out of date: regenerate it or reject it.",
  );
  await expect(count).toHaveText("2 changes from the assistant");
  await screenshot(window, "edit-out-of-date-en");

  // Accept all takes the other one, and leaves it.
  await expect(bar.getByTestId("proposals-accept-all")).toHaveAttribute(
    "title",
    "Changes out of date are left as they are",
  );
  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(count).toHaveText("1 change from the assistant");
  await expect(note).toContainText("They are high.");

  // Reject on its line takes it away, and Undo brings it back.
  await pointAndClick(window, outOfDate.getByTestId("change-reject"));
  await expect(bar).toHaveCount(0);
  await expect(note).toHaveText("The tides are really quite very strong. They are high.");
  await expect(window.getByTestId("undo-message")).toHaveText("Rejected 1 change");
  await window.keyboard.press("ControlOrMeta+z");
  await expect(outOfDate).toHaveCount(1);

  // Regenerate makes it again from the words as they are now, the User's among them.
  await pointAndClick(window, outOfDate.getByTestId("change-regenerate"));
  await expect(outOfDate).toHaveCount(0);
  await expect(removed).toHaveText(["really quite "]);
  await expect(count).toHaveText("1 change from the assistant");
  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(bar).toHaveCount(0);
  await expect(note).toHaveText("The tides are very strong. They are high.");
});

test("copying a paragraph with changes waiting gives the text as it stands, in plain text and rich text; accepting them changes the copy", async () => {
  const { window } = await launchApp(dataDir, { fakeChat: true });
  await dismissChatSetup(window);
  await useLocalChatModel(window);
  await newMind(window);
  const editor = window.getByTestId("mind-editor");
  await editor.click();
  const sentence = "The tides here are strong and really regular.";
  await window.keyboard.type(sentence);
  const note = editor.locator(":scope > p").first();

  // What Copy would put on the clipboard for the whole paragraph, by type.
  const copyParagraph = async () => {
    await editor.focus();
    await window.keyboard.press("ControlOrMeta+a");
    return editor.evaluate((dom) => {
      const data = new DataTransfer();
      dom.dispatchEvent(
        new ClipboardEvent("copy", { clipboardData: data, bubbles: true, cancelable: true }),
      );
      return { plain: data.getData("text/plain"), html: data.getData("text/html") };
    });
  };

  // A change that replaces "strong" with "powerful", waiting: the copy is the sentence as it was.
  await selectWords(window, note, sentence);
  await window.keyboard.press("ControlOrMeta+j");
  await window.keyboard.type("replace strong with powerful");
  await window.keyboard.press("Enter");
  await expect(note.locator("ins.proposal-insert")).toHaveText("powerful");
  const bar = window.getByTestId("proposals-bar");
  const replacing = await copyParagraph();
  expect(replacing.plain).toBe(sentence);
  expect(replacing.html).toContain(sentence);
  expect(replacing.html).not.toContain("powerful");

  // Accepted, the copy has the new word.
  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(bar).toHaveCount(0);
  await expect(note).toHaveText("The tides here are powerful and really regular.");

  // Shorten drops "really": waiting, the copy still has it; accepted, it doesn't.
  await editor.focus();
  await window.keyboard.press("ArrowRight");
  await selectWords(window, note, "The tides here are powerful and really regular.");
  await pointAndClick(window, window.getByTestId("selection-action-shorten"));
  await expect(note.locator("del.proposal-delete")).toHaveText("really ");
  const shortening = await copyParagraph();
  expect(shortening.plain).toBe("The tides here are powerful and really regular.");
  expect(shortening.html).toContain("really");
  await pointAndClick(window, bar.getByTestId("proposals-accept-all"));
  await expect(bar).toHaveCount(0);
  const shortened = await copyParagraph();
  expect(shortened.plain).toBe("The tides here are powerful and regular.");
  expect(shortened.html).not.toContain("really");
});
