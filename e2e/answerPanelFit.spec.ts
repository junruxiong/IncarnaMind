import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, type Locator, test } from "@playwright/test";
import { buildPdf } from "../tests/helpers/pdf";
import {
  addDocuments,
  askInAssistant,
  useLocalChatModel as connectLocalChatModel,
  createDataFolder,
  dismissChatSetup,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
} from "./app";

/**
 * What an Answer in the assistant holds fits the Answer (#898), at the sizes the assistant is
 * narrowest in, in English and Chinese: a wide table scrolls inside it, first column in view,
 * and opens as a table in a tab; Citation badges wrap with their sentence and never pass its
 * text. The checks are relations (inside, on one line, beside), never pixels.
 */
const SIZES = [
  { width: 1100, height: 760 },
  { width: 1440, height: 900 },
] as const;

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

async function start(language: "en" | "zh-CN", width: number, height: number) {
  const { app, window } = await launchApp(dataDir, { fakeChat: true, home: sources });
  await setWindowSize(app, window, width, height);
  await dismissChatSetup(window);
  await connectLocalChatModel(window);
  await setLanguage(window, language);
  const paths: string[] = [];
  for (const name of ["Tides", "Moon", "Harbour", "Weather"]) {
    const path = join(sources, `${name}.pdf`);
    await writeFile(
      path,
      buildPdf([
        { lines: [`${name} and tides`, `Every ${name} document says tides rise and fall.`] },
      ]),
    );
    paths.push(path);
  }
  await addDocuments(window, paths);
  await newMind(window);
  return window;
}

const QUESTIONS = {
  en: {
    table: "What do tides do in nine rows",
    forty: "What do tides do with forty citations",
    row: "What do tides do, twelve in a row",
  },
  "zh-CN": {
    table: "九行 tides",
    forty: "四十条引用 tides",
    row: "连续十二条 tides",
  },
} as const;

/** What in an Answer's text is wider than the text, or reaches past its right edge; and whether the thread scrolls sideways. */
function reachesPast(answer: Locator) {
  return answer.getByTestId("assistant-answer-text").evaluate((text) => {
    const box = text.getBoundingClientRect();
    const past: string[] = [];
    for (const element of text.querySelectorAll("*")) {
      // A table's cells are in its own frame, which scrolls: the frame is what must fit.
      const inFrame = element.closest(".tableWrapper");
      if (inFrame && inFrame !== element) continue;
      const rect = element.getBoundingClientRect();
      if (rect.width === 0) continue;
      if (rect.width > box.width + 0.5 || rect.right > box.right + 0.5) {
        past.push(`${element.tagName.toLowerCase()}.${element.className}`);
      }
    }
    const thread = text.closest('[role="log"]')?.parentElement;
    return { past, threadScrollsSideways: thread ? thread.scrollWidth > thread.clientWidth : null };
  });
}

for (const { width, height } of SIZES) {
  for (const language of ["en", "zh-CN"] as const) {
    test(`at ${width} × ${height}, in ${language}: a wide table scrolls to every column and opens as a table, and a run of Citations stays in the Answer`, async () => {
      test.setTimeout(240_000);
      const window = await start(language, width, height);
      const asked = QUESTIONS[language];

      await test.step("the table scrolls inside the Answer", async () => {
        const answer = await askInAssistant(window, asked.table);
        const text = answer.getByTestId("assistant-answer-text");
        const frame = text.locator(".tableWrapper");
        await expect(text.locator("table tr")).toHaveCount(10);
        await expect(frame.locator("tr").first().locator("th")).toHaveCount(4);

        expect(await reachesPast(answer)).toEqual({ past: [], threadScrollsSideways: false });

        // It is wider than the Answer, so it scrolls, and the keyboard reaches it.
        const sizes = await frame.evaluate((each) => ({
          scrolls: each.scrollWidth > each.clientWidth,
          overflow: getComputedStyle(each).overflowX,
        }));
        expect(sizes).toEqual({ scrolls: true, overflow: "auto" });
        await expect(frame).toHaveAttribute("tabindex", "0");
        await expect(frame).toHaveAttribute("role", "region");

        // Numbers are never cut: no cell wraps, and none holds more than it shows.
        const cut = await frame.evaluate((each) => {
          const cells = [...each.querySelectorAll("td:not(:first-child), th:not(:first-child)")];
          return cells
            .filter((cell) => {
              const words = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
              const lines = new Set<number>();
              for (let node = words.nextNode(); node; node = words.nextNode()) {
                if (node.parentElement?.closest(".citation")) continue;
                const range = document.createRange();
                range.selectNodeContents(node);
                for (const rect of range.getClientRects()) lines.add(Math.round(rect.top));
              }
              return cell.scrollWidth > cell.clientWidth || lines.size > 1;
            })
            .map((cell) => cell.textContent);
        });
        expect(cut).toEqual([]);

        // Scrolled with the keyboard to the end, each column comes into view in turn, while the
        // first column stays where it is, in front of what slides under it.
        await frame.focus();
        const seen = new Set<number>();
        for (let press = 0; press < 40; press++) {
          const state = await frame.evaluate((each) => {
            const frameBox = each.getBoundingClientRect();
            const headings = [...each.querySelectorAll("tr:first-child > *")] as HTMLElement[];
            const first = headings[0] as HTMLElement;
            const firstBox = first.getBoundingClientRect();
            const visible = headings.flatMap((heading, column) => {
              const box = heading.getBoundingClientRect();
              return column > 0 &&
                box.left >= firstBox.right - 0.5 &&
                box.right <= frameBox.right + 1.5
                ? [column]
                : [];
            });
            const middle = document.elementFromPoint(
              firstBox.left + firstBox.width / 2,
              firstBox.top + firstBox.height / 2,
            );
            return {
              visible,
              firstAtEdge: Math.abs(firstBox.left - frameBox.left) <= 1,
              firstInFront: first.contains(middle),
              atEnd: each.scrollLeft + each.clientWidth >= each.scrollWidth - 1,
            };
          });
          for (const column of state.visible) seen.add(column);
          expect(state.firstAtEdge).toBe(true);
          expect(state.firstInFront).toBe(true);
          if (state.atEnd) break;
          await window.keyboard.press("ArrowRight");
        }
        expect([...seen].sort()).toEqual([1, 2, 3]);
        await expect(frame).not.toHaveAttribute("data-more-after", "");
        await expect(frame).toHaveAttribute("data-more-before", "");
      });

      await test.step("Open as a table shows all of it in a tab beside the Mind", async () => {
        const answer = window.getByTestId("assistant-answer").last();
        const tabs = window.getByTestId("mind-tab");
        await expect(tabs).toHaveCount(1);
        const original = await tabs.first().getAttribute("title");
        const note = window.getByTestId("mind-editor");
        const before = await note.textContent();

        const open = answer.getByTestId("answer-open-table");
        await expect(open).toHaveText(language === "en" ? "Open as a table" : "以表格打开");
        // Under the table, on the Answer's edge.
        const opener = await open.boundingBox();
        const table = await answer.locator(".tableWrapper").boundingBox();
        if (!opener || !table) throw new Error("The table or its button isn't visible.");
        expect(opener.y).toBeGreaterThanOrEqual(table.y + table.height);
        await pointAndClick(window, open);

        await expect(tabs).toHaveCount(2);
        await expect(tabs.nth(1)).toHaveAttribute("aria-selected", "true");
        // The Mind the User was in is still there, as it was.
        await expect(tabs.first()).toHaveAttribute("title", original ?? "");
        await tabs.first().click();
        await expect(note).toHaveText(before ?? "");
        await tabs.nth(1).click();

        const opened = window.getByTestId("mind-editor").locator("table");
        await expect(opened).toHaveCount(1);
        await expect(opened.locator("tr")).toHaveCount(10);
        const cells = await opened
          .locator("tr")
          .evaluateAll((rows) =>
            rows.map((row) => [...row.children].map((cell) => (cell.textContent ?? "").trim())),
          );
        expect(cells.every((row) => row.length === 4)).toBe(true);
        expect(cells[0]).toEqual(
          language === "en"
            ? ["Figure", "Harrow Foods", "Brightmart", "Calder Grocers"]
            : ["指标", "哈罗食品", "明智超市", "考尔德杂货"],
        );
        expect(cells[9]?.slice(1)).toEqual(["1.1×", "0.1×", "0.65×"]);

        const said = answer.getByTestId("answer-added");
        await expect(said).toContainText(
          language === "en" ? "from this table" : "已用这个表格新建 Mind",
        );
        await pointAndClick(window, answer.getByTestId("answer-added-undo"));
        await expect(tabs).toHaveCount(1);
      });

      await test.step("twelve Citations in a row wrap and stay in the Answer", async () => {
        const answer = await askInAssistant(window, asked.row);
        const chips = answer.getByTestId("citation-chip");
        await expect(chips).toHaveCount(12);
        expect(await reachesPast(answer)).toEqual({ past: [], threadScrollsSideways: false });
        for (const chip of await chips.all())
          await expect(chip.locator(".citation-marker-icon")).toBeVisible();
      });

      if (width === SIZES[0].width) {
        await test.step("forty Citations, each beside its sentence", async () => {
          const answer = await askInAssistant(window, asked.forty, { timeout: 120_000 });
          const text = answer.getByTestId("assistant-answer-text");
          await expect(text.getByTestId("citation-chip")).toHaveCount(40);
          expect(await reachesPast(answer)).toEqual({ past: [], threadScrollsSideways: false });

          const strays = await text.evaluate((root) => {
            const stray: number[] = [];
            for (const paragraph of root.querySelectorAll(":scope > p")) {
              let last: DOMRect | null = null;
              const walker = document.createTreeWalker(
                paragraph,
                NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
              );
              for (let node = walker.nextNode(); node; node = walker.nextNode()) {
                if (node instanceof Text) {
                  if (node.parentElement?.closest(".citation")) continue;
                  const value = node.nodeValue ?? "";
                  const end = value.trimEnd().length;
                  if (end === 0) continue;
                  const range = document.createRange();
                  range.setStart(node, end - 1);
                  range.setEnd(node, end);
                  last = range.getBoundingClientRect();
                } else if ((node as Element).matches(".citation-marker")) {
                  const chip = (node as Element).getBoundingClientRect();
                  const middle = (chip.top + chip.bottom) / 2;
                  // On the line of the last word before it, and after it.
                  const beside =
                    last !== null &&
                    middle > last.top &&
                    middle < last.bottom &&
                    chip.left >= last.right - 1;
                  if (!beside) stray.push(Number((node as Element).textContent));
                }
              }
            }
            return stray;
          });
          expect(strays).toEqual([]);
        });
      }
    });
  }
}

const COMMAND = {
  en: { open: "Open as a table", nth: (n: number) => `Open table ${n} as a table`, find: "table" },
  "zh-CN": { open: "以表格打开", nth: (n: number) => `以表格打开第 ${n} 个表格`, find: "表格" },
} as const;

for (const language of ["en", "zh-CN"] as const) {
  test(`⌘K opens a table of the Answer as a table: the one the cursor is in, or one listed, in ${language}`, async () => {
    test.setTimeout(240_000);
    const window = await start(language, 1440, 900);
    const words = COMMAND[language];
    const asked = language === "en" ? "What do tides do in two tables" : "两个表 tides";
    const answer = await askInAssistant(window, asked);
    const frames = answer.locator(".tableWrapper");
    await expect(frames).toHaveCount(2);
    const tabs = window.getByTestId("mind-tab");
    await expect(tabs).toHaveCount(1);
    const items = window.getByTestId("palette").getByTestId("palette-item");

    const ask = async () => {
      await window.keyboard.press("ControlOrMeta+k");
      await expect(window.getByTestId("palette-search")).toBeFocused();
      await window.keyboard.type(words.find);
    };
    const opened = async (header: string[]) => {
      await expect(tabs).toHaveCount(2);
      await expect(tabs.nth(1)).toHaveAttribute("aria-selected", "true");
      const table = window.getByTestId("mind-editor").locator("table");
      await expect(table.locator("tr")).toHaveCount(3);
      await expect(table.locator("th")).toHaveText(header);
      await pointAndClick(window, answer.getByTestId("answer-added-undo"));
      await expect(tabs).toHaveCount(1);
    };

    await test.step("with the cursor in no table, both are listed, and the second is chosen", async () => {
      await ask();
      await expect(items.filter({ hasText: words.nth(1) })).toHaveCount(1);
      await expect(items.filter({ hasText: words.nth(2) })).toHaveCount(1);
      await items.filter({ hasText: words.nth(2) }).click();
      await opened(language === "en" ? ["Region", "Share"] : ["地区", "占比"]);
    });

    await test.step("with the cursor in a table, only that one is offered, and it opens", async () => {
      const second = frames.nth(1);
      await pointAndClick(window, second.locator("td").first());
      await ask();
      await expect(items.filter({ hasText: words.open })).toHaveCount(1);
      await expect(items.filter({ hasText: words.nth(1) })).toHaveCount(0);
      await items.filter({ hasText: words.open }).click();
      await opened(language === "en" ? ["Region", "Share"] : ["地区", "占比"]);
    });
  });

  test(`an Answer with one table is offered "Open as a table" from ⌘K, and its tab is open before the table is in, in ${language}`, async () => {
    test.setTimeout(240_000);
    const window = await start(language, 1440, 900);
    const words = COMMAND[language];
    const answer = await askInAssistant(
      window,
      language === "en" ? "What do tides do in nine rows" : "九行 tides",
    );
    const tabs = window.getByTestId("mind-tab");
    await expect(tabs).toHaveCount(1);

    await window.keyboard.press("ControlOrMeta+k");
    await expect(window.getByTestId("palette-search")).toBeFocused();
    await window.keyboard.type(words.find);
    const item = window.getByTestId("palette").getByTestId("palette-item");
    await expect(item.filter({ hasText: words.open })).toHaveCount(1);

    // The Mind it opens holds no table the first moment its tab shows (openTable.test.ts pins
    // that the table is put in after the tab opens).
    await window.evaluate(() => {
      const seen: number[] = [];
      Object.assign(window, { tableWhenTabShown: seen });
      new MutationObserver(() => {
        const tabsNow = document.querySelectorAll('[data-testid="mind-tab"]').length;
        if (tabsNow === 2 && seen.length === 0) {
          seen.push(document.querySelectorAll('[data-testid="mind-editor"] table').length);
        }
      }).observe(document.body, { subtree: true, childList: true, attributes: true });
    });
    await item.filter({ hasText: words.open }).click();

    await expect(tabs).toHaveCount(2);
    const table = window.getByTestId("mind-editor").locator("table");
    await expect(table.locator("tr")).toHaveCount(10);
    expect(
      await window.evaluate(
        () => (window as unknown as { tableWhenTabShown: number[] }).tableWhenTabShown,
      ),
    ).toEqual([0]);
    await expect(answer.getByTestId("answer-added")).toContainText(
      language === "en" ? "from this table" : "已用这个表格新建 Mind",
    );
  });
}
