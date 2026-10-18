import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import type { Editor } from "@tiptap/core";
import type { CoreBridge } from "../src/core/api";
import {
  addDocuments,
  clickEndOf,
  createDataFolder,
  dismissChatSetup,
  focusSettled,
  launchApp,
  motionSettled,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  slideSelect,
  useLocalChatModel,
} from "./app";

/**
 * A cited sentence found unsupported offers three fixes in its card (#1111):
 * Cut the unsupported part (an Edit with AI applied at once), Find support
 * in every Document, and Keep as my view; each is also in ⌘K, and the ones
 * that change the Mind take one ⌘Z. A sentence with no Citation keeps its
 * card. The scripted model (src/main/fakeChat/checks.ts) finds the owner's
 * sentence partly supported and cuts it as `PARTLY` says.
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

const QUOTE_ONE = "QN methods are much faster than GD algorithms";
const QUOTE_TWO = "First-order methods converge more slowly";
const UNCITED = "Penguins nest on the harbour cliffs.";

const LANGUAGES = {
  en: {
    owner:
      "The tutorial discusses the trade-off between training speed, robustness and memory needs: experiments usually show quasi-Newton methods are faster than gradient descent; first-order methods are usually more robust but slower.",
    cut: "Experiments usually show quasi-Newton methods are faster than gradient descent, and first-order methods are slower.",
    first: "The tutorial discusses",
    cutAction: "Cut the unsupported part",
    findSupport: "Find support",
    keep: "Keep as my view",
    keepCommand: "Keep as my view",
    nothing: "Nothing in your Documents says this.",
  },
  "zh-CN": {
    owner:
      "教程讨论了训练速度、鲁棒性和内存需求之间的权衡：已有实验通常显示，拟牛顿方法比梯度下降更快；一阶优化方法通常更加稳健，但速度较慢。",
    cut: "已有实验通常显示，拟牛顿方法比梯度下降更快，一阶优化方法速度较慢。",
    first: "教程讨论了",
    cutAction: "删去无依据的部分",
    findSupport: "查找出处",
    keep: "保留为我的观点",
    keepCommand: "保留为我的观点",
    nothing: "你的文档中没有内容提到这一点。",
  },
} as const;

/** Whether the first box sits inside the second. */
const inside = async (inner: Locator, outer: Locator) => {
  const [a, b] = [await inner.boundingBox(), await outer.boundingBox()];
  if (!a || !b) return false;
  return (
    a.x >= b.x - 1 &&
    a.x + a.width <= b.x + b.width + 1 &&
    a.y >= b.y - 1 &&
    a.y + a.height <= b.y + b.height + 1
  );
};

/** A line's words, without its Citations' numbers. */
const wordsOf = (line: Locator) =>
  line.evaluate((element) => {
    const copy = element.cloneNode(true) as HTMLElement;
    for (const citation of copy.querySelectorAll('[data-testid="citation"]')) citation.remove();
    return (copy.textContent ?? "").trim();
  });

for (const language of ["en", "zh-CN"] as const) {
  const words = LANGUAGES[language];
  test(`a cited unsupported sentence's card cuts the part, finds support and keeps it as the User's view, each with Undo and from ⌘K; one with no Citation keeps its card (${language})`, async () => {
    test.setTimeout(180_000);
    await writeFile(join(sources, "Tutorial one.md"), `# One\n\n${QUOTE_ONE} in practice.\n`);
    await writeFile(join(sources, "Tutorial two.md"), `# Two\n\n${QUOTE_TWO} than QN methods.\n`);
    const { app, window } = await launchApp(dataDir, { fakeChat: true });
    await dismissChatSetup(window);
    await useLocalChatModel(window);
    await setLanguage(window, language);
    await setWindowSize(app, window, 1600, 900);
    await addDocuments(window, [
      join(sources, "Tutorial one.md"),
      join(sources, "Tutorial two.md"),
    ]);
    const documents = await window.evaluate(() =>
      (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind.listDocuments(),
    );
    const cite = (name: string, quote: string) => {
      const document = documents.find((each) => each.name === name);
      return {
        type: "citation",
        attrs: {
          passageId: "passage-1",
          documentId: document?.id,
          documentName: name,
          contentHash: document?.contentHash,
          pageFrom: 1,
          pageTo: 1,
          location: null,
          quote,
          check: "found",
          checkReason: null,
        },
      };
    };
    const text = (value: string) => ({ type: "text", text: value });

    await newMind(window);
    const editor = window.getByTestId("mind-editor");
    await editor.evaluate(
      (dom, content) => {
        (dom as unknown as { editor: Editor }).editor.commands.insertContentAt(0, content);
      },
      [
        {
          type: "paragraph",
          content: [
            text(words.owner),
            cite("Tutorial one", QUOTE_ONE),
            cite("Tutorial two", QUOTE_TWO),
          ],
        },
        { type: "paragraph", content: [text(UNCITED)] },
      ],
    );
    const lines = editor.locator(":scope > p");
    const owner = lines.nth(0);
    const marks = window.getByTestId("claim-mark");
    const card = window.getByTestId("claim-card");
    const ownerMark = () => marks.first();
    const openCard = async (mark: Locator) => {
      await motionSettled(window);
      await pointAndClick(window, mark);
      await expect(card).toBeVisible();
    };
    // ⌘Z in the Mind, where each fix leaves the focus.
    const undo = async () => {
      await focusSettled(window, editor);
      await window.keyboard.press("ControlOrMeta+z");
    };

    await slideSelect(window, owner, words.first, { line: lines.nth(1), words: "cliffs." });
    await pointAndClick(
      window,
      window.getByTestId("selection-menu").getByTestId("selection-action-check-against-documents"),
    );
    await expect(marks).toHaveCount(2);
    await expect(marks.nth(0)).toHaveAttribute("data-check", "unsupported");
    await expect(marks.nth(1)).toHaveAttribute("data-check", "unsupported");

    // No Citation: today's card, with no fixes.
    await openCard(marks.nth(1));
    await expect(card.getByTestId("claim-nothing")).toHaveText(words.nothing);
    await expect(card.getByTestId("claim-find-sources")).toBeVisible();
    await expect(card.getByTestId("claim-remove")).toBeVisible();
    for (const fix of ["claim-cut", "claim-find-support", "claim-keep"]) {
      await expect(card.getByTestId(fix)).toHaveCount(0);
    }
    await window.keyboard.press("Escape");

    // With Citations: the three fixes, the first the primary one, inside the card at both widths.
    for (const width of [1600, 1100]) {
      await setWindowSize(app, window, width, 900);
      await openCard(ownerMark());
      await expect(card.getByTestId("claim-covers")).toBeVisible();
      const fixes = [
        card.getByTestId("claim-cut"),
        card.getByTestId("claim-find-support"),
        card.getByTestId("claim-keep"),
      ];
      await expect(fixes[0] as Locator).toHaveText(words.cutAction);
      await expect(fixes[1] as Locator).toHaveText(words.findSupport);
      await expect(fixes[2] as Locator).toHaveText(words.keep);
      await expect(fixes[0] as Locator).toHaveClass(/citation-card-action--primary/);
      for (const fix of fixes) expect(await inside(fix, card)).toBe(true);
      for (const gone of ["claim-find-sources", "claim-search-web", "claim-remove"]) {
        await expect(card.getByTestId(gone)).toHaveCount(0);
      }
      await window.keyboard.press("Escape");
    }

    // Cut the unsupported part: applied at once, the Citations kept; ⌘Z puts the sentence back.
    // Wide, so the Mind keeps the focus once a card closes (one group gives it up, #963).
    await setWindowSize(app, window, 1600, 900);
    await openCard(ownerMark());
    await pointAndClick(window, card.getByTestId("claim-cut"));
    await expect.poll(() => wordsOf(owner)).toBe(words.cut);
    await expect(owner.getByTestId("citation")).toHaveCount(2);
    await expect(ownerMark()).toHaveAttribute("data-check", "needs-recheck");
    await undo();
    await expect.poll(() => wordsOf(owner)).toBe(words.owner);
    await expect(ownerMark()).toHaveAttribute("data-check", "unsupported");

    // Keep as my view: the mark goes, and ⌘Z brings it back.
    await openCard(ownerMark());
    await pointAndClick(window, card.getByTestId("claim-keep"));
    await expect(marks).toHaveCount(1);
    await undo();
    await expect(marks).toHaveCount(2);
    await expect(ownerMark()).toHaveAttribute("data-check", "unsupported");

    // Find support: the sentence is checked again, in every Document.
    const before = await ownerMark().getAttribute("data-claim");
    await openCard(ownerMark());
    await pointAndClick(window, card.getByTestId("claim-find-support"));
    await expect(ownerMark()).not.toHaveAttribute("data-claim", before ?? "");
    await expect(ownerMark()).toHaveAttribute("data-check", "unsupported");

    // From ⌘K, with the cursor in the sentence.
    await clickEndOf(window, owner);
    await runCommand(window, "check.keepAsView", words.keepCommand);
    await expect(marks).toHaveCount(1);
    await undo();
    await expect(marks).toHaveCount(2);
    await clickEndOf(window, owner);
    await runCommand(window, "check.cutUnsupported", words.cutAction);
    await expect.poll(() => wordsOf(owner)).toBe(words.cut);
    await app.close();
  });
}

/** Runs a command from ⌘K, as a person does. */
async function runCommand(window: Page, id: string, label: string): Promise<void> {
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type(label);
  const item = window.getByTestId("palette").locator(`[data-key="command:${id}"]`);
  await expect(item).toContainText(label);
  await pointAndClick(window, item);
}
