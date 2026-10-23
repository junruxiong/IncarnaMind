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

/** Nothing of the page is under the pane: the scroller ends where it starts, and so does `target`. */
async function expectClearOf(window: Page, pane: Locator, target?: Locator): Promise<void> {
  const box = await pane.boundingBox();
  const page = await (target ?? window.getByTestId("mind-scroller")).boundingBox();
  expect(box && page && page.x + page.width <= box.x + 1).toBe(true);
}

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
  // The sidebar and the assistant both shown.
  await setWindowSize(app, window, 1440, 900);
  await dismissChatSetup(window);
  await importFile(window, app, "Supply agreement v5.docx", supplyRedline());
  await motionSettled(window);

  // The line under the title says whose changes wait, and Review opens the pane.
  const line = window.getByTestId("their-changes-line");
  await expect(line.getByTestId("their-changes-count")).toHaveText(
    "7 changes from Dana Kim and Luis Ortega",
  );
  await screenshot(window, "review-line");
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
  await expectClearOf(window, pane);
  await screenshot(window, "review-pane");

  // A card selected scrolls the Mind to its change and washes it there; ↓ goes to the next.
  const editor = window.getByTestId("mind-editor");
  await pointAndClick(window, cards.nth(6).getByRole("button"));
  await expect(cards.nth(6)).toHaveAttribute("data-selected", "true");
  const picked = editor.locator(".review-picked");
  await expect(picked.first()).toContainText("Northwind’s total liability");
  await expect.poll(() => inView(window, picked.first())).toBe(true);
  await expectClearOf(window, pane, picked.first());
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

  // Under a 760px Mind it folds to "Changes ({n})", which opens it at the page's right edge,
  // the page narrowing so a card's change shows beside it.
  await setWindowSize(app, window, 1100, 800);
  const fold = window.getByTestId("review-fold");
  await expect(fold).toHaveText("修订（6）");
  await expect(pane).toHaveCount(0);
  await pointAndClick(window, fold);
  await expect(pane).toHaveAttribute("data-placement", "over");
  await expectClearOf(window, pane);
  await pointAndClick(window, cards.nth(2).getByRole("button"));
  await expect(picked).toHaveText(["8%, on 30 days’ notice", /^6%, on 45 days’ notice/]);
  await expect.poll(() => inView(window, picked.last())).toBe(true);
  await expectClearOf(window, pane, picked.last());
  await screenshot(window, "review-pane-zh-1100");
  await pointAndClick(window, fold);
  await expect(pane).toHaveCount(0);
  await setLanguage(window, "en");
  await expect(fold).toHaveText("Changes (6)");
  await screenshot(window, "review-pane-folded-1100");

  // At 900 px the assistant goes under the tabs; the Mind is still under 760 px, so the pane stays folded.
  await setWindowSize(app, window, 900, 800);
  await expect(fold).toHaveText("Changes (6)");
  await pointAndClick(window, fold);
  await expect(pane).toHaveAttribute("data-placement", "over");
  await expectClearOf(window, pane);
  await screenshot(window, "review-pane-900");
  await pane.getByTestId("review-close").click();
  await expect(pane).toHaveCount(0);
  await expect(fold).toHaveCount(0);

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

test("Accept and Reject on the selected card act at once with Undo, from the mouse, ⌘↵ and ⌘⌫ in the pane and ⌘K, one by one or all of one person's", async () => {
  const { app, window } = await launchApp(dataDir, { home });
  await setWindowSize(app, window, 1600, 900);
  await dismissChatSetup(window);
  await importFile(window, app, "Supply agreement v5.docx", supplyRedline());
  await window.keyboard.press("ControlOrMeta+Backslash");
  await expect(window.getByTestId("sidebar-panel")).toHaveAttribute("data-hidden", "true");
  await pointAndClick(window, window.getByTestId("their-changes-review"));
  const pane = window.getByTestId("review-pane");
  const editor = window.getByTestId("mind-editor");
  const cards = cardsOf(pane);
  const supply = cards.nth(0);
  const notice = cards.nth(1);

  // Only the selected card has Accept and Reject.
  await pointAndClick(window, supply.getByRole("button").first());
  await expect(pane.getByTestId("review-accept")).toHaveCount(1);
  await expect(supply.getByTestId("review-accept")).toHaveText("Accept");
  await expect(supply.getByTestId("review-reject")).toHaveText("Reject");
  await screenshot(window, "review-card-selected");

  // Accept acts at once: the words are the Mind's, the card says so, and the next change is selected.
  await pointAndClick(window, supply.getByTestId("review-accept"));
  await expect(supply).toHaveAttribute("data-state", "accepted");
  await expect(supply.getByTestId("review-status")).toHaveText("Accepted");
  await expect(editor.locator("ins.proposal-theirs", { hasText: "and Gdańsk" })).toHaveCount(0);
  await expect(editor.locator("p", { hasText: "1.2 Northwind" })).toContainText(
    "Leeds and Gdańsk and delivers",
  );
  await expect(notice).toHaveAttribute("data-selected", "true");
  await expect(notice.locator("[data-review-id]")).toBeFocused();
  // ⌘Z, with the focus in the pane, puts it back, waiting.
  await expect(window.getByTestId("undo-message")).toHaveText("Accepted 1 change");
  await window.keyboard.press("ControlOrMeta+z");
  await expect(supply).toHaveAttribute("data-state", "waiting");
  await expect(editor.locator("ins.proposal-theirs", { hasText: "and Gdańsk" })).toHaveCount(1);

  // The change Undo brought back is selected again. ↓, and ⌘⌫ rejects the next; ⌘Z takes it back.
  await expect(supply).toHaveAttribute("data-selected", "true");
  await window.keyboard.press("ArrowDown");
  await expect(notice).toHaveAttribute("data-selected", "true");
  await window.keyboard.press("ControlOrMeta+Backspace");
  await expect(notice).toHaveAttribute("data-state", "rejected");
  await expect(editor.locator("p", { hasText: "1.3 Northwind" })).toContainText(
    "Northwind gives 90 days’ notice",
  );
  await expect(cards.nth(2)).toHaveAttribute("data-selected", "true");
  await expect(cards.nth(2).locator("[data-review-id]")).toBeFocused();
  await window.keyboard.press("ControlOrMeta+z");
  await expect(notice).toHaveAttribute("data-state", "waiting");
  // The change Undo brought back is selected again, so ⌘↵ accepts that one.
  await expect(notice).toHaveAttribute("data-selected", "true");
  await expect(notice.locator("[data-review-id]")).toBeFocused();
  await window.keyboard.press("ControlOrMeta+Enter");
  await expect(notice).toHaveAttribute("data-state", "accepted");

  // From ⌘K: Next change, then all of Dana Kim's at once.
  const palette = async (label: string) => {
    await window.keyboard.press("ControlOrMeta+k");
    await expect(window.getByTestId("palette-search")).toBeFocused();
    await window.keyboard.type(label);
    await pointAndClick(
      window,
      window.getByTestId("palette-item").filter({ hasText: label }).first(),
    );
  };
  await palette("Next change");
  await expect(cards.nth(4)).toHaveAttribute("data-selected", "true");
  await palette("Accept all changes from Luis Ortega");
  await expect(cards.nth(4)).toHaveAttribute("data-state", "accepted");
  await expect(pane.locator('[data-testid="review-change"][data-state="waiting"]')).toHaveCount(5);

  // In Unresolved, a card decided leaves; the last leaves nothing to review.
  await pointAndClick(window, pane.getByTestId("review-filter-unresolved").locator(".."));
  await expect(pane.locator('[data-testid="review-change"]')).toHaveCount(5);
  await pointAndClick(window, pane.getByTestId("review-author-menu"));
  await pointAndClick(window, window.getByTestId("review-author").filter({ hasText: "Dana Kim" }));
  await setLanguage(window, "zh-CN");
  await pointAndClick(window, cards.first().locator("[data-review-id]"));
  await expect(cards.first().getByTestId("review-accept")).toHaveText("接受");
  await expect(cards.first().getByTestId("review-reject")).toHaveText("拒绝");
  await screenshot(window, "review-card-selected-zh");
  for (let left = 5; left > 0; left--) {
    await expect(pane.locator('[data-testid="review-change"]')).toHaveCount(left);
    await window.keyboard.press("ControlOrMeta+Enter");
  }
  // Dana's comments stay, and the first takes the focus.
  await expect(pane.locator('[data-testid="review-change"]')).toHaveCount(0);
  await expect(pane.getByTestId("review-comment")).toHaveCount(2);
  await expect(
    pane.getByTestId("review-comment").first().locator("[data-review-id]"),
  ).toBeFocused();
  await window.keyboard.press("Escape");
  await expect(pane).toHaveCount(0);
  await expect(window.getByTestId("their-changes-line")).toHaveCount(0);
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
