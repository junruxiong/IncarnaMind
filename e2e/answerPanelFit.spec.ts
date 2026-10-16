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
  focusRingShows,
  launchApp,
  newMind,
  pointAndClick,
  removeDataFolder,
  setLanguage,
  setWindowSize,
  tabTo,
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
    two: "What do tides do in two tables",
  },
  "zh-CN": {
    table: "九行 tides",
    forty: "四十条引用 tides",
    row: "连续十二条 tides",
    two: "两个表 tides",
  },
} as const;

const WORDS = {
  en: {
    open: "Open as a table",
    nth: (n: number) => `Open table ${n} as a table`,
    find: "table",
    made: "from this table",
    header: ["Figure", "Harrow Foods", "Brightmart", "Calder Grocers"],
    second: ["Region", "Share"],
  },
  "zh-CN": {
    open: "以表格打开",
    nth: (n: number) => `以表格打开第 ${n} 个表格`,
    find: "表格",
    made: "已用这个表格新建 Mind",
    header: ["指标", "哈罗食品", "明智超市", "考尔德杂货"],
    second: ["地区", "占比"],
  },
} as const;

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

const leftFadeShows = (frame: Locator) =>
  frame.evaluate((each) => {
    const cell = each.querySelector("tr > :first-child");
    if (!cell) return null;
    const { content } = getComputedStyle(cell, "::after");
    return content !== "none" && content !== "normal";
  });

for (const { width, height } of SIZES) {
  for (const language of ["en", "zh-CN"] as const) {
    test(`at ${width} × ${height}, in ${language}: a wide table scrolls to every column and opens as a table, and a run of Citations stays in the Answer`, async () => {
      test.setTimeout(240_000);
      const window = await start(language, width, height);
      const asked = QUESTIONS[language];
      const words = WORDS[language];

      await test.step("the table scrolls inside the Answer", async () => {
        const answer = await askInAssistant(window, asked.table);
        const text = answer.getByTestId("assistant-answer-text");
        const frame = text.locator(".tableWrapper");
        await expect(text.locator("table tr")).toHaveCount(10);
        await expect(frame.locator("tr").first().locator("th")).toHaveCount(4);

        expect(await reachesPast(answer)).toEqual({ past: [], threadScrollsSideways: false });

        const sizes = await frame.evaluate((each) => ({
          scrolls: each.scrollWidth > each.clientWidth,
          overflow: getComputedStyle(each).overflowX,
        }));
        expect(sizes).toEqual({ scrolls: true, overflow: "auto" });
        await expect(frame).toHaveAttribute("tabindex", "0");
        await expect(frame).toHaveAttribute("role", "region");

        const cut = await frame.evaluate((each) => {
          const cells = [...each.querySelectorAll("td:not(:first-child), th:not(:first-child)")];
          return cells
            .filter((cell) => {
              const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
              const lines = new Set<number>();
              for (let node = walker.nextNode(); node; node = walker.nextNode()) {
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

        await expect(frame).toHaveAttribute("data-more-after", "");
        expect(await leftFadeShows(frame)).toBe(false);
        await tabTo(window, frame);
        expect(await focusRingShows(window)).toBe(true);
        const ring = await frame.evaluate((each) => {
          const style = getComputedStyle(each);
          return {
            offset: Number.parseFloat(style.outlineOffset),
            masked: style.maskImage !== "none",
          };
        });
        expect(ring.offset).toBeLessThanOrEqual(-2);
        expect(ring.masked).toBe(false);

        const seen = new Set<number>();
        for (let press = 0; press < 40; press++) {
          const state = await frame.evaluate((each) => {
            const frameBox = each.getBoundingClientRect();
            const headings = [...each.querySelectorAll<HTMLElement>("tr:first-child > *")];
            const first = headings[0];
            if (!first) throw new Error("The table has no heading row.");
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
        expect(await leftFadeShows(frame)).toBe(true);
        expect(await focusRingShows(window)).toBe(true);
      });

      await test.step("Open as a table shows all of it in a tab beside the Mind", async () => {
        const answer = window.getByTestId("assistant-answer").last();
        const tabs = window.getByTestId("mind-tab");
        await expect(tabs).toHaveCount(1);
        const original = await tabs.first().getAttribute("title");
        const note = window.getByTestId("mind-editor");
        const before = await note.textContent();

        const open = answer.getByTestId("answer-open-table");
        await expect(open).toHaveText(words.open);
        const opener = await open.boundingBox();
        const table = await answer.locator(".tableWrapper").boundingBox();
        if (!opener || !table) throw new Error("The table or its button isn't visible.");
        expect(opener.y).toBeGreaterThanOrEqual(table.y + table.height);
        await pointAndClick(window, open);

        await expect(tabs).toHaveCount(2);
        await expect(tabs.nth(1)).toHaveAttribute("aria-selected", "true");
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
        expect(cells[0]).toEqual(words.header);
        expect(cells[9]?.slice(1)).toEqual(["1.1×", "0.1×", "0.65×"]);

        await expect(answer.getByTestId("answer-added")).toContainText(words.made);
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
                  const end = (node.nodeValue ?? "").trimEnd().length;
                  if (end === 0) continue;
                  const range = document.createRange();
                  range.setStart(node, end - 1);
                  range.setEnd(node, end);
                  last = range.getBoundingClientRect();
                } else if (node instanceof Element && node.matches(".citation-marker")) {
                  const chip = node.getBoundingClientRect();
                  const middle = (chip.top + chip.bottom) / 2;
                  const beside =
                    last !== null &&
                    middle > last.top &&
                    middle < last.bottom &&
                    chip.left >= last.right - 1;
                  if (!beside) stray.push(Number(node.textContent));
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

for (const language of ["en", "zh-CN"] as const) {
  test(`⌘K opens a table of the Answer as a table: the one the cursor is in, or one listed, in ${language}`, async () => {
    test.setTimeout(240_000);
    const window = await start(language, 1440, 900);
    const words = WORDS[language];
    const answer = await askInAssistant(window, QUESTIONS[language].two);
    const frames = answer.locator(".tableWrapper");
    await expect(frames).toHaveCount(2);
    const tabs = window.getByTestId("mind-tab");
    await expect(tabs).toHaveCount(1);
    const items = window.getByTestId("palette").getByTestId("palette-item");

    for (const frame of await frames.all()) {
      await expect(frame).not.toHaveAttribute("tabindex");
      await expect(frame).not.toHaveAttribute("data-more-after");
      expect(await leftFadeShows(frame)).toBe(false);
    }

    const ask = async () => {
      await window.keyboard.press("ControlOrMeta+k");
      await expect(window.getByTestId("palette-search")).toBeFocused();
      await window.keyboard.type(words.find);
    };
    const opened = async () => {
      await expect(tabs).toHaveCount(2);
      await expect(tabs.nth(1)).toHaveAttribute("aria-selected", "true");
      const table = window.getByTestId("mind-editor").locator("table");
      await expect(table.locator("tr")).toHaveCount(3);
      await expect(table.locator("th")).toHaveText(words.second);
      await pointAndClick(window, answer.getByTestId("answer-added-undo"));
      await expect(tabs).toHaveCount(1);
    };

    await test.step("with the cursor in no table, both are listed, and the second is chosen", async () => {
      await ask();
      await expect(items.filter({ hasText: words.nth(1) })).toHaveCount(1);
      await expect(items.filter({ hasText: words.nth(2) })).toHaveCount(1);
      await items.filter({ hasText: words.nth(2) }).click();
      await opened();
    });

    await test.step("with the cursor in a table, only that one is offered, and it opens", async () => {
      await pointAndClick(window, frames.nth(1).locator("td").first());
      await ask();
      await expect(items.filter({ hasText: words.open })).toHaveCount(1);
      await expect(items.filter({ hasText: words.nth(1) })).toHaveCount(0);
      await items.filter({ hasText: words.open }).click();
      await opened();
    });
  });

  test(`an Answer with one table: Open as a table from ⌘K and from the keyboard, ⌘Z takes it back, and its tab is open before the table is in, in ${language}`, async () => {
    test.setTimeout(240_000);
    const window = await start(language, 1440, 900);
    const words = WORDS[language];
    const answer = await askInAssistant(window, QUESTIONS[language].table);
    const tabs = window.getByTestId("mind-tab");
    const table = window.getByTestId("mind-editor").locator("table");
    await expect(tabs).toHaveCount(1);

    await test.step("from ⌘K", async () => {
      await window.keyboard.press("ControlOrMeta+k");
      await expect(window.getByTestId("palette-search")).toBeFocused();
      await window.keyboard.type(words.find);
      const item = window.getByTestId("palette").getByTestId("palette-item");
      await expect(item.filter({ hasText: words.open })).toHaveCount(1);

      const tableWhenTabShown = await window.evaluateHandle(() => {
        const seen: number[] = [];
        new MutationObserver(() => {
          const shown = document.querySelectorAll('[data-testid="mind-tab"]').length;
          if (shown === 2 && seen.length === 0) {
            seen.push(document.querySelectorAll('[data-testid="mind-editor"] table').length);
          }
        }).observe(document.body, { subtree: true, childList: true, attributes: true });
        return seen;
      });
      await item.filter({ hasText: words.open }).click();

      await expect(tabs).toHaveCount(2);
      await expect(table.locator("tr")).toHaveCount(10);
      expect(await tableWhenTabShown.jsonValue()).toEqual([0]);
      await expect(answer.getByTestId("answer-added")).toContainText(words.made);
      await pointAndClick(window, answer.getByTestId("answer-added-undo"));
      await expect(tabs).toHaveCount(1);
    });

    await test.step("from the keyboard: Tab to the button, Enter, then ⌘Z", async () => {
      const open = answer.getByTestId("answer-open-table");
      await tabTo(window, open);
      expect(await focusRingShows(window)).toBe(true);
      await window.keyboard.press("Enter");

      await expect(tabs).toHaveCount(2);
      await expect(table.locator("tr")).toHaveCount(10);
      await expect(answer.getByTestId("answer-added")).toContainText(words.made);

      await window.keyboard.press("ControlOrMeta+z");
      await expect(tabs).toHaveCount(1);
      await expect(answer.getByTestId("answer-added")).toHaveCount(0);
      await expect(open).toBeVisible();
    });
  });
}
