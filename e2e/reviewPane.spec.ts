import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, type Page, test } from "@playwright/test";
import { supplyRedline } from "../tests/helpers/redline";
import { paragraph, wordFile } from "../tests/helpers/word";
import {
  createDataFolder,
  dismissChatSetup,
  focusSettled,
  importWordFromMenu,
  interceptOpenDialog,
  launchApp,
  motionSettled,
  openPageMenu,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/*
 * The reviewing pane of someone else's changes in a Word file (#697; the
 * board Final-10-Redlines): the line under the title opens it beside the
 * Mind, with a card for each change and comment in the Mind's order; a card
 * selected scrolls to its change and washes it, a change clicked in the text
 * selects its card; All · Unresolved and the author menu filter the cards; a
 * decided change stays in All; Esc closes it; and under a 760px Mind it folds
 * to "Changes ({n})". Set INCARNAMIND_SCREENSHOTS to a folder to also save
 * screenshots there.
 */

const SCREENSHOTS = process.env.INCARNAMIND_SCREENSHOTS;

async function screenshot(window: Page, name: string): Promise<void> {
  if (!SCREENSHOTS) return;
  await motionSettled(window);
  await window.screenshot({ path: join(SCREENSHOTS, `${name}.png`) });
}

let dataDir: string;
let home: string;
let files: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  home = await createDataFolder();
  files = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
  await removeDataFolder(files);
});

async function importFile(
  window: Page,
  app: Parameters<typeof interceptOpenDialog>[0],
  name: string,
  bytes: Buffer,
) {
  const path = join(files, name);
  await writeFile(path, bytes);
  await interceptOpenDialog(app, path);
  const dialog = await importWordFromMenu(window);
  await dialog.getByTestId("word-import-confirm").click();
  await expect(dialog).toBeHidden();
}

/** Each card as it reads: its clause and status, or the comment's author, and its words. */
const cardsOf = (pane: Locator) =>
  pane.locator('[data-testid="review-change"], [data-testid="review-comment"]');

/** Whether a box is inside what the Mind's scroller shows. */
async function inView(window: Page, target: Locator): Promise<boolean> {
  const box = await target.boundingBox();
  const scroller = await window.getByTestId("mind-scroller").boundingBox();
  return (
    !!box && !!scroller && box.y >= scroller.y && box.y + box.height <= scroller.y + scroller.height
  );
}

test("the reviewing pane lists their changes and comments in order, follows the text both ways, filters by status and author, and folds under a narrow Mind", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1600, 900);
  await dismissChatSetup(window);
  await importFile(window, app, "Supply agreement v5.docx", supplyRedline());
  // As on the board: the sidebar hidden, the assistant beside the tabs.
  await window.keyboard.press("ControlOrMeta+Backslash");
  await expect(window.getByTestId("sidebar-panel")).toHaveAttribute("data-hidden", "true");
  await motionSettled(window);

  // The line under the title says whose changes wait, and Review opens the pane.
  const line = window.getByTestId("their-changes-line");
  await expect(line.getByTestId("their-changes-count")).toHaveText(
    "7 changes from Dana Kim and Luis Ortega",
  );
  await pointAndClick(window, line.getByTestId("their-changes-review"));
  const pane = window.getByTestId("review-pane");
  await expect(pane).toHaveAttribute("data-placement", "beside");
  await expect(line).toHaveCount(0);

  // Every change and comment, in the Mind's order; the first card has the focus.
  const cards = cardsOf(pane);
  await expect(cards).toHaveCount(9);
  await expect(pane.getByTestId("review-change").getByTestId("review-clause")).toHaveText([
    "1.2",
    "1.3",
    "2.1",
    "3.1",
    "4.2",
    "6.2",
    "7.1",
  ]);
  await expect(cards.nth(3)).toHaveAttribute("data-testid", "review-comment");
  await expect(cards.nth(3).getByTestId("review-comment-text")).toHaveText(
    "Resin is up 11% since June. Our letter of 19 September has the figures.",
  );
  const notice = cards.nth(1);
  await expect(notice.locator("del")).toHaveText("90");
  await expect(notice.locator("ins")).toHaveText("60");
  await expect(notice.getByRole("button")).toHaveAttribute("title", "Luis Ortega, 5 Oct 2026");
  await expect(cards.first().getByRole("button")).toBeFocused();
  await expect(pane.getByTestId("review-filter-all")).toBeChecked();

  // The pane sits beside the page, which keeps its text clear of it.
  const paneBox = await pane.boundingBox();
  const page = await window.getByTestId("mind-scroller").boundingBox();
  expect(paneBox && page && paneBox.x >= page.x + page.width - 1).toBe(true);
  await screenshot(window, "review-pane");

  // A card selected scrolls the Mind to its change and washes it there; ↓ goes to the next.
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, cards.nth(6).getByRole("button"));
  await expect(cards.nth(6)).toHaveAttribute("data-selected", "true");
  const picked = editor.locator(".review-picked");
  await expect(picked.first()).toContainText("Northwind’s total liability");
  await expect.poll(() => inView(window, picked.first())).toBe(true);
  await window.keyboard.press("ArrowDown");
  await expect(cards.nth(7)).toHaveAttribute("data-selected", "true");
  await expect(cards.nth(7).getByRole("button")).toBeFocused();
  await window.keyboard.press("ArrowUp");
  await window.keyboard.press("ArrowUp");
  await expect(cards.nth(5)).toHaveAttribute("data-selected", "true");
  await expect(picked).toHaveText([", for orders of at least 40 pallets"]);

  // A change clicked in the text selects its card, and so does a comment's mark.
  await pointAndClick(window, editor.locator("ins.proposal-theirs", { hasText: "and Gdańsk" }));
  await expect(cards.first()).toHaveAttribute("data-selected", "true");
  await pointAndClick(window, editor.locator(".word-atom-comment").nth(1));
  await expect(cards.nth(7)).toHaveAttribute("data-selected", "true");

  // Only one person's.
  await pointAndClick(window, pane.getByTestId("review-author-menu"));
  const luis = window.getByTestId("review-author").filter({ hasText: "Luis Ortega (2)" });
  await expect(window.getByTestId("review-author")).toHaveText(["Dana Kim (7)", "Luis Ortega (2)"]);
  await pointAndClick(window, luis);
  await expect(cards).toHaveCount(2);
  await expect(pane.getByTestId("review-author-menu")).toHaveText("Luis Ortega");
  await pointAndClick(window, pane.getByTestId("review-author-menu"));
  await pointAndClick(window, window.getByTestId("review-author-everyone"));
  await expect(cards).toHaveCount(9);

  // Accepted in the text, the change stays in All as Accepted, and leaves Unresolved.
  await editor.locator("ins.proposal-theirs", { hasText: "60" }).hover();
  await pointAndClick(window, window.getByTestId("change-bar").getByTestId("change-accept"));
  await expect(notice).toHaveAttribute("data-state", "accepted");
  await expect(notice.getByTestId("review-status")).toHaveText("Accepted");
  await expect(cards).toHaveCount(9);
  await pointAndClick(window, pane.getByTestId("review-filter-unresolved").locator(".."));
  await expect(cards).toHaveCount(8);
  await expect(pane.getByTestId("review-change").getByTestId("review-clause")).not.toContainText([
    "1.3",
  ]);

  // Esc closes it; the line comes back with what still waits.
  await cards.first().getByRole("button").focus();
  await window.keyboard.press("Escape");
  await expect(pane).toHaveCount(0);
  await expect(line.getByTestId("their-changes-count")).toHaveText(
    "6 changes from Dana Kim and Luis Ortega",
  );

  // The ⋯ menu reopens it, on Unresolved now that something is decided.
  await pointAndClick(window, (await openPageMenu(window)).getByTestId("page-review"));
  await expect(pane).toBeVisible();
  await expect(pane.getByTestId("review-filter-unresolved")).toBeChecked();
  await expect(pane.locator("[data-review-id]").first()).toBeFocused();

  // In Chinese.
  await setLanguage(window, "zh-CN");
  await expect(pane.getByTestId("review-filter-all").locator("..")).toHaveText("全部");
  await expect(pane.getByTestId("review-filter-unresolved").locator("..")).toHaveText("未处理");
  await expect(pane.getByTestId("review-author-menu")).toHaveText("所有人");
  await pointAndClick(window, pane.getByTestId("review-filter-all").locator(".."));
  await expect(notice.getByTestId("review-status")).toHaveText("已接受");
  await screenshot(window, "review-pane-zh");

  // Under a 760px Mind it folds to "Changes ({n})", which opens it over the page's right edge.
  for (const width of [1100, 900]) {
    await setWindowSize(app, window, width, 800);
    const fold = window.getByTestId("review-fold");
    await expect(fold).toHaveText("修订（6）");
    await expect(pane).toHaveCount(0);
    await pointAndClick(window, fold);
    await expect(pane).toHaveAttribute("data-placement", "over");
    const over = await pane.boundingBox();
    const scroller = await window.getByTestId("mind-scroller").boundingBox();
    expect(
      over &&
        scroller &&
        over.x >= scroller.x &&
        over.x + over.width <= scroller.x + scroller.width + 1,
    ).toBe(true);
    await screenshot(window, `review-pane-zh-${width}`);
    await pointAndClick(window, fold);
    await expect(pane).toHaveCount(0);
  }
  await setLanguage(window, "en");
  await expect(window.getByTestId("review-fold")).toHaveText("Changes (6)");
  await screenshot(window, "review-pane-folded-900");
  await pointAndClick(window, window.getByTestId("review-fold"));
  await expect(pane).toHaveAttribute("data-placement", "over");
  await screenshot(window, "review-pane-900");
  await pane.getByTestId("review-close").click();
  await expect(pane).toHaveCount(0);
  await expect(window.getByTestId("review-fold")).toHaveCount(0);

  // Wide again, ⌘K opens it beside the page, from the cursor in the text.
  await setWindowSize(app, window, 1600, 900);
  await pointAndClick(window, editor.locator("p", { hasText: "Nothing in this agreement" }));
  await focusSettled(window, editor);
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type("Review changes");
  await pointAndClick(
    window,
    window.getByTestId("palette-item").filter({ hasText: "Review changes" }).first(),
  );
  await expect(pane).toHaveAttribute("data-placement", "beside");
  await app.close();
});

test("a Word file without anyone's changes or comments offers no review", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await importFile(
    window,
    app,
    "Plain memo.docx",
    wordFile({ body: paragraph("1. Nobody changed this clause.") }),
  );
  await expect(window.getByTestId("mind-editor")).toContainText("Nobody changed this clause.");
  await expect(window.getByTestId("their-changes-line")).toHaveCount(0);
  const menu = await openPageMenu(window);
  await expect(menu.getByTestId("page-from")).toBeVisible();
  await expect(menu.getByTestId("page-review")).toHaveCount(0);
  await window.keyboard.press("Escape");
  await window.keyboard.press("ControlOrMeta+k");
  await expect(window.getByTestId("palette-search")).toBeFocused();
  await window.keyboard.type("Review changes");
  await expect(
    window.getByTestId("palette-item").filter({ hasText: /^Review changes$/ }),
  ).toHaveCount(0);
  await app.close();
});
