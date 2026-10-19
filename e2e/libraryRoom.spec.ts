import { expect, type Locator, test } from "@playwright/test";
import {
  createDataFolder,
  launchApp,
  placeAssistant,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";
import { openFullLibrary } from "./libraryFixture";

/*
 * The Library gives names their width and keeps its sentences in tooltips (#1192): Added goes first,
 * then Format, before a name is cut; the Organize line, a Folder's description, the Recently deleted
 * sentence and the Tag picker's legend are tooltips, and a Folder's description is in its Get info.
 */

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

/** What `useFittingColumns` asks for: a name that fits in this, whole; a longer one, this much of it. */
const WORTH_ROOM = 320;
const LEAST_ROOM = 160;

/** The names of the rows on screen that have less room than they are asked for. */
function namesCut(library: Locator) {
  return library.evaluate(
    (root, [worth, least]) =>
      [...root.querySelectorAll('[data-testid="library-document-name"]')]
        .filter((name) => {
          const box = name.getBoundingClientRect();
          return (
            box.width > 0 &&
            box.height > 0 &&
            name.clientWidth + 1 <
              (name.scrollWidth <= (worth ?? 0) ? name.scrollWidth : (least ?? 0))
          );
        })
        .map((name) => name.textContent),
    [WORTH_ROOM, LEAST_ROOM],
  );
}

/** The words a User can read in the Library as it stands: the text in view, not what scrolls below. */
function wordsOnScreen(library: Locator, language: string) {
  return library.evaluate((root, locale) => {
    const view = root.getBoundingClientRect();
    const segmenter = new Intl.Segmenter(locale, { granularity: "word" });
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let words = 0;
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.textContent ?? "";
      if (!text.trim()) continue;
      const range = document.createRange();
      range.selectNodeContents(node);
      const seen = [...range.getClientRects()].some(
        (box) =>
          box.width > 0 &&
          box.height > 0 &&
          box.bottom > view.top &&
          box.top < view.bottom &&
          box.right > view.left &&
          box.left < view.right,
      );
      if (seen) words += [...segmenter.segment(text)].filter((part) => part.isWordLike).length;
    }
    return words;
  }, language);
}

for (const language of ["en", "zh-CN"] as const) {
  test(`at 1,100 px no name is cut for want of room, Added and Format go first, and their words stay in the tooltip (${language})`, async () => {
    const { app, window } = await launchLibrary(language);
    const library = window.getByTestId("library");
    await setWindowSize(app, window, 1100, 900);

    // With the assistant beside the page, and with it folded: the narrow page gives names the room.
    for (const folded of [false, true]) {
      await placeAssistant(window, { folded });
      await expect.poll(() => namesCut(library), { message: `folded: ${folded}` }).toEqual([]);
    }

    // Wide, Added and Format are there.
    const first = library.getByTestId("library-document").first();
    await setWindowSize(app, window, 1600, 900);
    await expect(first.getByTestId("library-document-format")).toBeVisible();
    await expect(first.getByTestId("library-document-added")).toBeVisible();

    // With a Document open beside the page, in the narrowest room, every column has gone first:
    // Added, Format and the Tags' chips, and the names that are cut take all the room there is.
    await setWindowSize(app, window, 1100, 900);
    await pointAndClick(window, library.getByTestId("library-document-name").first());
    await expect(window.getByTestId("viewer")).toBeVisible();
    await expect(first.getByTestId("library-document-added")).toBeHidden();
    await expect(first.getByTestId("library-document-format")).toBeHidden();
    await expect(library.locator('[data-testid="tag-chip"]:visible')).toHaveCount(0);

    // What they said is the name's tooltip.
    const tip = await first
      .getByTestId("library-document-name")
      .locator("xpath=..")
      .getAttribute("title");
    expect(tip).toContain(
      ((await first.getByTestId("library-document-format").textContent()) ?? "").trim(),
    );
    expect(tip).toContain(
      ((await first.getByTestId("library-document-added").textContent()) ?? "").trim(),
    );
    await app.close();
  });

  test(`the Library at 1,440 px shows about 200 words at most (${language})`, async () => {
    const { app, window } = await launchLibrary(language);
    await setWindowSize(app, window, 1440, 900);
    const library = window.getByTestId("library");
    expect(await wordsOnScreen(library, language)).toBeLessThanOrEqual(200);
    await app.close();
  });

  test(`the Organize line, a Folder's description and the Recently deleted sentence are tooltips, and the description is in the Folder's Get info (${language})`, async () => {
    const { app, window } = await launchLibrary(language);
    await setWindowSize(app, window, 1440, 900);
    const library = window.getByTestId("library");
    const words = TIPS[language];

    // Organize is a button; what it does is its tooltip.
    await expect(library.getByTestId("library-organize-explain")).toHaveCount(0);
    await expect(library.getByTestId("library-organize")).toHaveAttribute("title", words.organize);

    // A Folder's description is not a line on the page: it shows pointing at the head, and in Get info.
    const head = library.getByTestId("library-section-head").filter({
      has: window.getByTestId("library-section-name").getByText("Board", { exact: true }),
    });
    await expect(head).not.toContainText(words.board);
    await expect(head).toHaveAttribute("title", words.board);
    await pointAndClick(window, head, { button: "right" });
    await pointAndClick(window, window.getByTestId("file.info").filter({ visible: true }));
    const info = window.getByTestId("info-panel");
    await expect(info.getByTestId("info-name")).toHaveText("Board");
    await expect(info.getByTestId("info-description")).toHaveText(words.board);
    await expect(info.getByTestId("info-documents")).toHaveText(words.documents);
    await window.keyboard.press("Escape");
    await expect(info).toHaveCount(0);

    // From the keyboard: ⌘I on the focused head.
    await head.getByTestId("library-section-name").focus();
    await window.keyboard.press("ControlOrMeta+i");
    await expect(window.getByTestId("info-panel").getByTestId("info-description")).toHaveText(
      words.board,
    );
    await window.keyboard.press("Escape");

    // Recently deleted: its sentence is its head's tooltip, and Empty is in the head.
    const deleted = window.getByTestId("recently-deleted");
    await deleted.scrollIntoViewIfNeeded();
    const deletedHead = deleted.getByTestId("deleted-section-head");
    await pointAndClick(window, deletedHead.getByTestId("deleted-heading"));
    await expect(deletedHead).toHaveAttribute("title", words.deleted);
    await expect(deleted).not.toContainText(words.deleted);
    await expect(deletedHead.getByTestId("deleted-empty")).toBeVisible();
    await app.close();
  });

  test(`the Tag picker has no legend of keys, and each mark says what it is (${language})`, async () => {
    const { app, window } = await launchLibrary(language);
    await setWindowSize(app, window, 1440, 900);
    const library = window.getByTestId("library");
    const register = library.getByTestId("library-document").filter({
      has: window.getByTestId("library-document-name").getByText("Risk register", { exact: true }),
    });
    await pointAndClick(window, register.getByTestId("tag-chip").first());
    const picker = register.getByTestId("document-tags-popover");
    await expect(picker.getByTestId("tag-picker-input")).toBeFocused();
    await expect(picker.getByTestId("tag-picker-keys")).toHaveCount(0);
    // A Tag on the Document has its check, which explains itself; no description line is cut under a name.
    const on = picker.locator('[data-testid="tag-option"][data-coverage="all"]').first();
    await expect(on.locator("[title]").first()).toHaveAttribute("title", TIPS[language].markOn);
    await expect(on.locator(".truncate")).toHaveCount(1);
    await app.close();
  });
}

const TIPS = {
  en: {
    organize: "Suggests a Folder and Tags for each Document. You decide.",
    board: "Minutes, board packs and resolutions",
    deleted: "Kept here for 30 days, then deleted for good. Files on disk aren’t touched.",
    documents: "9 Documents",
    markOn: "on this Document",
  },
  "zh-CN": {
    organize: "为每个文档建议文件夹和标签，由你决定。",
    board: "Minutes, board packs and resolutions",
    deleted: "删除的内容保留 30 天，之后彻底删除。磁盘上的文件不受影响。",
    documents: "9 个文档",
    markOn: "已添加到此文档",
  },
} as const;

async function launchLibrary(language: "en" | "zh-CN") {
  const launched = await launchApp(dataDir);
  await openFullLibrary(launched.window, sources);
  await setLanguage(launched.window, language);
  return launched;
}
