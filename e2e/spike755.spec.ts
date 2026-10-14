import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { readSpikeLook, type SpikeLook } from "../spikes/755-page-view/wordLayout";
import {
  createDataFolder,
  dismissChatSetup,
  interceptOpenDialog,
  launchApp,
  removeDataFolder,
} from "./app";

/*
 * Spike for #755, not for merging: measures Page view's approaches on Word files.
 * SPIKE755=1 SPIKE755_FIXTURES=<dir> SPIKE755_FILES=a.docx,b.docx SPIKE755_OUT=<dir>
 * [SPIKE755_MODES=look,plus,custom] npx playwright test e2e/spike755.spec.ts
 */

const RUN = process.env.SPIKE755 === "1";
const FIXTURES = process.env.SPIKE755_FIXTURES ?? "";
const FILES = (process.env.SPIKE755_FILES ?? "").split(",").filter(Boolean);
const MODES = (process.env.SPIKE755_MODES ?? "look,plus,custom").split(",");
const OUT = process.env.SPIKE755_OUT ?? "";
const KEYS = Number(process.env.SPIKE755_KEYS ?? 8);

/** The faces LibreOffice drew each file's fonts with (its PDFs' font lists), so both sides use the same. */
const FONT_MAP: Record<string, string> = {
  Calibri: "Carlito",
  SimSun: "Arial Unicode MS",
  宋体: "Songti SC",
  仿宋_GB2312: "Songti SC",
  仿宋: "Songti SC",
  黑体: "Heiti SC",
  楷体: "Kaiti SC",
  楷体_GB2312: "Kaiti SC",
  標楷體: "Kaiti TC",
  等线: "PingFang SC",
  Batang: "Times New Roman",
};

test.describe.configure({ timeout: 60 * 60_000 });

let dataDir: string;
let home: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
  home = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
  await removeDataFolder(home);
});

interface Api {
  apply(): Promise<number>;
  pages(): number;
  pageStart(page: number): number | null;
  firstLines(length?: number): { page: number; text: string }[];
  positionOnPage(page: number, where: "start" | "middle"): number;
  select(pos: number): void;
  caret(): { top: number; left: number; scrollTop: number; head: number };
  footnotes(): { label: string; page: number | null; notePage: number | null }[];
  stats(): {
    runs: {
      reason: string;
      totalMs: number;
      measureMs: number;
      layoutMs: number;
      pagesLaidOut: number;
      converged: boolean;
      background: boolean;
      at: number;
    }[];
    keydownAt: number;
    settledAt: number;
    backgroundDoneAt: number;
    composingSkips: number;
  };
  frames(): { keydownAt: number; frameAt: number }[];
  resetStats(): void;
  textAround(pos: number): string;
  docSize(): number;
  blockCount(): number;
  zoom(factor: number, how?: "zoom" | "transform"): void;
}

const api = (window: Page) =>
  window.evaluate(() => Boolean((window as unknown as { __spike755Api?: Api }).__spike755Api));

async function openMind(window: Page, title: string) {
  const editor = window.getByTestId("mind-editor");
  if (!(await editor.isVisible().catch(() => false))) {
    await window.getByText(title, { exact: true }).first().click();
  }
  await expect(editor).toHaveAttribute("contenteditable", "true", { timeout: 10 * 60_000 });
  await expect.poll(() => api(window), { timeout: 60_000 }).toBe(true);
}

/** Waits until the pages stop changing: the custom pass says it's done; for the plugin, no change for 40 frames. */
async function settle(window: Page, mode: string, since: number): Promise<number> {
  if (mode === "custom") {
    await expect
      .poll(
        () =>
          window.evaluate(
            (t) =>
              ((window as unknown as { __spike755Api: Api }).__spike755Api.stats()
                .backgroundDoneAt ?? 0) >= t,
            since,
          ),
        { timeout: 10 * 60_000, intervals: [20] },
      )
      .toBe(true);
    return window.evaluate(
      () => (window as unknown as { __spike755Api: Api }).__spike755Api.stats().backgroundDoneAt,
    );
  }
  return window.evaluate(
    () =>
      new Promise<number>((resolve) => {
        const spike = (window as unknown as { __spike755Api: Api }).__spike755Api;
        let last = spike.pages();
        let changedAt = performance.now();
        let still = 0;
        const tick = () => {
          const now = spike.pages();
          if (now !== last) {
            last = now;
            changedAt = performance.now();
            still = 0;
          } else still++;
          if (still >= 40) resolve(changedAt);
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }),
  );
}

const now = (window: Page) => window.evaluate(() => performance.now());

/** Types `count` characters one by one at a position, measuring each keystroke. */
async function typeAt(window: Page, mode: string, pos: number, count: number) {
  await window.evaluate(
    (p) => (window as unknown as { __spike755Api: Api }).__spike755Api.select(p),
    pos,
  );
  await window.waitForTimeout(300);
  const before = await window.evaluate(() =>
    (window as unknown as { __spike755Api: Api }).__spike755Api.caret(),
  );
  const keys: Record<string, number | boolean | null>[] = [];
  for (let index = 0; index < count; index++) {
    await window.evaluate(() =>
      (window as unknown as { __spike755Api: Api }).__spike755Api.resetStats(),
    );
    const caretBefore = await window.evaluate(() =>
      (window as unknown as { __spike755Api: Api }).__spike755Api.caret(),
    );
    await window.keyboard.type("x");
    await expect
      .poll(
        () =>
          window.evaluate(
            () => (window as unknown as { __spike755Api: Api }).__spike755Api.frames().length,
          ),
        {
          intervals: [10],
        },
      )
      .toBeGreaterThan(0);
    if (mode === "custom") {
      await expect
        .poll(
          () =>
            window.evaluate(() => {
              const stats = (window as unknown as { __spike755Api: Api }).__spike755Api.stats();
              return stats.backgroundDoneAt >= stats.keydownAt && stats.keydownAt > 0;
            }),
          { timeout: 120_000, intervals: [10] },
        )
        .toBe(true);
    } else {
      await window.waitForTimeout(120);
    }
    const measured = await window.evaluate(() => {
      const spike = (window as unknown as { __spike755Api: Api }).__spike755Api;
      const stats = spike.stats();
      const frame = spike.frames()[0];
      const sync = stats.runs.filter((run) => !run.background);
      const background = stats.runs.filter((run) => run.background);
      return {
        frameMs: frame ? frame.frameAt - frame.keydownAt : null,
        syncLayoutMs: sync.reduce((sum, run) => sum + run.totalMs, 0),
        syncPages: sync.reduce((sum, run) => sum + run.pagesLaidOut, 0),
        measureMs: sync.reduce((sum, run) => sum + run.measureMs, 0),
        backgroundRuns: background.length,
        backgroundPages: background.reduce((sum, run) => sum + run.pagesLaidOut, 0),
        settledMs: stats.settledAt && stats.keydownAt ? stats.settledAt - stats.keydownAt : null,
        allDoneMs:
          stats.backgroundDoneAt && stats.keydownAt
            ? stats.backgroundDoneAt - stats.keydownAt
            : null,
        converged: sync.some((run) => run.converged),
        caret: spike.caret(),
      };
    });
    keys.push({
      frameMs: measured.frameMs,
      syncLayoutMs: measured.syncLayoutMs,
      syncPages: measured.syncPages,
      measureMs: measured.measureMs,
      backgroundRuns: measured.backgroundRuns,
      backgroundPages: measured.backgroundPages,
      settledMs: measured.settledMs,
      allDoneMs: measured.allDoneMs,
      converged: measured.converged,
      caretDy: measured.caret.top - caretBefore.top,
      scrollDy: measured.caret.scrollTop - caretBefore.scrollTop,
    });
  }
  return { before, keys };
}

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? (sorted[Math.floor(sorted.length / 2)] as number) : null;
};
const max = (values: number[]) => (values.length ? Math.max(...values) : null);

function summary(keys: Record<string, number | boolean | null>[]) {
  const pick = (name: string) =>
    keys.map((key) => key[name]).filter((value): value is number => typeof value === "number");
  return {
    frameMedian: median(pick("frameMs")),
    frameMax: max(pick("frameMs")),
    syncMedian: median(pick("syncLayoutMs")),
    syncMax: max(pick("syncLayoutMs")),
    allDoneMedian: median(pick("allDoneMs")),
    allDoneMax: max(pick("allDoneMs")),
    caretMoves: keys.filter((key) => Math.abs(Number(key.caretDy)) > 1).length,
    scrollMoves: keys.filter((key) => Math.abs(Number(key.scrollDy)) > 1).length,
    keys,
  };
}

/** Scrolls through the Mind a frame at a time, returning the frame times. */
function scrollFrames(window: Page) {
  return window.evaluate(
    () =>
      new Promise<{ median: number; p95: number; frames: number }>((resolve) => {
        const scroller = document.querySelector(".mind-scroller") as HTMLElement | null;
        if (!scroller) return resolve({ median: 0, p95: 0, frames: 0 });
        scroller.scrollTop = 0;
        const times: number[] = [];
        let last = performance.now();
        let count = 0;
        const step = () => {
          const at = performance.now();
          times.push(at - last);
          last = at;
          scroller.scrollTop += 400;
          if (++count < 120) requestAnimationFrame(step);
          else {
            times.sort((a, b) => a - b);
            resolve({
              median: times[Math.floor(times.length / 2)] ?? 0,
              p95: times[Math.floor(times.length * 0.95)] ?? 0,
              frames: times.length,
            });
          }
        };
        requestAnimationFrame(step);
      }),
  );
}

for (const file of FILES) {
  test(`spike 755: ${file}`, async () => {
    test.skip(!RUN, "the #755 spike runs only when asked");
    const path = join(FIXTURES, file);
    const look: SpikeLook = await readSpikeLook(await readFile(path));
    const title = basename(file, ".docx");
    const { app, window } = await launchApp(dataDir, { home });
    await window.setViewportSize({ width: 1600, height: 1000 }).catch(() => undefined);
    await dismissChatSetup(window);
    await interceptOpenDialog(app, path);
    await window.getByTestId("plus-menu").click();
    await window.getByTestId("plus-import-word").click();
    const dialog = window.getByTestId("word-import-dialog");
    await expect(dialog).toBeVisible();
    const importStarted = Date.now();
    await dialog.getByTestId("word-import-confirm").click();
    await expect(window.getByTestId("mind-editor")).toHaveAttribute("contenteditable", "true", {
      timeout: 10 * 60_000,
    });
    const importMs = Date.now() - importStarted;

    const results: Record<string, unknown> = { file, importMs, sections: look.sections.length };
    for (const mode of MODES) {
      await window.addInitScript(
        (config) => {
          (window as unknown as { __spike755: unknown }).__spike755 = config;
        },
        {
          mode,
          look,
          fontMap: FONT_MAP,
          syncPages: Number(process.env.SPIKE755_SYNC ?? 2),
          stats: {
            runs: [],
            keydownAt: 0,
            updateAt: 0,
            settledAt: 0,
            backgroundDoneAt: 0,
            composingSkips: 0,
          },
        },
      );
      await window.reload();
      await openMind(window, title);
      await window.evaluate(() => document.fonts.ready.then(() => undefined));
      const blocks = await window.evaluate(() =>
        (window as unknown as { __spike755Api: Api }).__spike755Api.blockCount(),
      );
      const started = await window.evaluate(() =>
        (window as unknown as { __spike755Api: Api }).__spike755Api.apply(),
      );
      const settledAt = await settle(window, mode, started);
      const layoutMs = mode === "look" ? 0 : settledAt - started;
      const runs = await window.evaluate(
        () => (window as unknown as { __spike755Api: Api }).__spike755Api.stats().runs,
      );
      const pages = await window.evaluate(() =>
        (window as unknown as { __spike755Api: Api }).__spike755Api.pages(),
      );
      const firstLines =
        mode === "look"
          ? []
          : await window.evaluate(() =>
              (window as unknown as { __spike755Api: Api }).__spike755Api.firstLines(),
            );
      const footnotes =
        mode === "custom"
          ? await window.evaluate(() =>
              (window as unknown as { __spike755Api: Api }).__spike755Api.footnotes(),
            )
          : [];
      if (OUT && process.env.SPIKE755_SHOTS === "1") {
        await window.screenshot({ path: join(OUT, `${title}-${mode}-top.png`) });
      }
      let detail: unknown = null;
      if (mode === "custom" && process.env.SPIKE755_DETAIL) {
        detail = await window.evaluate(
          (list) =>
            list.map((page) =>
              (
                window as unknown as { __spike755Api: { pageDetail(page: number): unknown } }
              ).__spike755Api.pageDetail(page),
            ),
          process.env.SPIKE755_DETAIL.split(",").map(Number),
        );
      }
      if (OUT && process.env.SPIKE755_SHOTPAGES && mode !== "look") {
        for (const page of process.env.SPIKE755_SHOTPAGES.split(",").map(Number)) {
          await window.evaluate(
            (p) =>
              (
                window as unknown as { __spike755Api: { scrollToPage(page: number): void } }
              ).__spike755Api.scrollToPage(p),
            page,
          );
          await window.waitForTimeout(200);
          await window.screenshot({ path: join(OUT, `${title}-${mode}-page${page}.png`) });
        }
      }
      const scrolling = await scrollFrames(window);
      const debug = await window.evaluate(() =>
        (window as unknown as { __spike755Api: { debug(): unknown } }).__spike755Api.debug(),
      );
      const entry: Record<string, unknown> = {
        detail,
        debug,
        blocks,
        layoutMs,
        firstRun: runs[0] ?? null,
        runs: runs.length,
        pages,
        firstLines,
        footnotes,
        scrolling,
      };

      if (mode === "custom" && OUT && process.env.SPIKE755_PDF === "1") {
        const printAt = await window.evaluate(() =>
          (
            window as unknown as { __spike755Api: { printMode(on: boolean): number } }
          ).__spike755Api.printMode(true),
        );
        await settle(window, mode, printAt);
        const pdfLines = await window.evaluate(() =>
          (window as unknown as { __spike755Api: Api }).__spike755Api.firstLines(),
        );
        const pdf = await app.evaluate(async ({ BrowserWindow }) => {
          const target =
            BrowserWindow.getAllWindows().find((each) => each.isVisible()) ??
            BrowserWindow.getAllWindows()[0];
          const data = await target?.webContents.printToPDF({
            preferCSSPageSize: true,
            printBackground: true,
            margins: { top: 0, bottom: 0, left: 0, right: 0 },
          });
          return data ? data.toString("base64") : "";
        });
        await writeFile(join(OUT, `${title}-pages.pdf`), Buffer.from(pdf, "base64"));
        await writeFile(join(OUT, `${title}-print-firstlines.json`), JSON.stringify(pdfLines));
        const backAt = await window.evaluate(() =>
          (
            window as unknown as { __spike755Api: { printMode(on: boolean): number } }
          ).__spike755Api.printMode(false),
        );
        await settle(window, mode, backAt);
      }
      const textBeforeEdits = await window.evaluate(() =>
        (window as unknown as { __spike755Api: { docText(): string } }).__spike755Api.docText(),
      );
      if (mode === "custom") {
        entry.alignment = await window.evaluate(() =>
          (
            window as unknown as { __spike755Api: { alignment(): unknown } }
          ).__spike755Api.alignment(),
        );
      }
      // Keystrokes on page 1, page 200 (or the last but one), and page 3.
      const targets = [1, Math.max(1, Math.min(200, pages - 1)), Math.min(3, Math.max(1, pages))];
      const typing: Record<string, unknown> = {};
      for (const page of [...new Set(targets)]) {
        const usePage = mode === "look" ? 1 : page;
        let pos: number;
        if (mode === "look") {
          pos = await window.evaluate(
            (fraction) => {
              const spike = (window as unknown as { __spike755Api: Api }).__spike755Api;
              return Math.max(2, Math.floor(spike.docSize() * fraction));
            },
            page === 1 ? 0.002 : page / Math.max(pages, 300),
          );
          pos = await window.evaluate((p) => {
            const spike = (window as unknown as { __spike755Api: Api }).__spike755Api;
            spike.select(p);
            return spike.caret().head;
          }, pos);
        } else {
          pos = await window.evaluate(
            (p) =>
              (window as unknown as { __spike755Api: Api }).__spike755Api.positionOnPage(
                p,
                "middle",
              ),
            usePage,
          );
        }
        typing[`page${page}`] = summary((await typeAt(window, mode, pos, KEYS)).keys);
      }
      entry.typing = typing;

      // Enter on page 3, then Undo of everything typed.
      if (mode !== "look") {
        const pos = await window.evaluate(
          (p) =>
            (window as unknown as { __spike755Api: Api }).__spike755Api.positionOnPage(p, "middle"),
          Math.min(3, pages),
        );
        await window.evaluate(
          (p) => (window as unknown as { __spike755Api: Api }).__spike755Api.select(p),
          pos,
        );
        await window.evaluate(() =>
          (window as unknown as { __spike755Api: Api }).__spike755Api.resetStats(),
        );
        const enterAt = await now(window);
        await window.keyboard.press("Enter");
        const enterDone = await settle(window, mode, enterAt);
        entry.enter = {
          ms: enterDone - enterAt,
          pages: await window.evaluate(() =>
            (window as unknown as { __spike755Api: Api }).__spike755Api.pages(),
          ),
        };
        const undoAt = await now(window);
        let undos = 0;
        while (
          undos < 80 &&
          (await window.evaluate(() =>
            (window as unknown as { __spike755Api: { docText(): string } }).__spike755Api.docText(),
          )) !== textBeforeEdits
        ) {
          await window.keyboard.press("Meta+z");
          undos++;
          await window.waitForTimeout(60);
        }
        const undoDone = await settle(window, mode, undoAt);
        const restored =
          (await window.evaluate(() =>
            (window as unknown as { __spike755Api: { docText(): string } }).__spike755Api.docText(),
          )) === textBeforeEdits;
        const afterUndo = await window.evaluate(() =>
          (window as unknown as { __spike755Api: Api }).__spike755Api.firstLines(),
        );
        entry.undo = {
          undos,
          restored,
          ms: undoDone - undoAt,
          pages: await window.evaluate(() =>
            (window as unknown as { __spike755Api: Api }).__spike755Api.pages(),
          ),
          firstLinesSame: JSON.stringify(afterUndo) === JSON.stringify(firstLines),
          firstLinesDiffer: afterUndo.filter((line, index) => line.text !== firstLines[index]?.text)
            .length,
        };
      }

      // Chinese IME at the start of a page that begins inside a paragraph (custom), or of page 2.
      if (mode !== "look" && pages > 1) {
        let target: number | null = null;
        if (mode === "custom") {
          target = await window.evaluate(() => {
            const spike = (window as unknown as { __spike755Api: Api }).__spike755Api;
            for (let page = 2; page <= spike.pages(); page++) {
              const start = spike.pageStart(page);
              if (start !== null) return start;
            }
            return null;
          });
        }
        target ??= await window.evaluate(() =>
          (window as unknown as { __spike755Api: Api }).__spike755Api.positionOnPage(2, "start"),
        );
        await window.evaluate(
          (p) => (window as unknown as { __spike755Api: Api }).__spike755Api.select(p),
          target,
        );
        await window.evaluate(() =>
          (window as unknown as { __spike755Api: Api }).__spike755Api.resetStats(),
        );
        const textBefore = await window.evaluate(
          (p) => (window as unknown as { __spike755Api: Api }).__spike755Api.textAround(p),
          target,
        );
        const cdp = await window.context().newCDPSession(window);
        const imeAt = await now(window);
        await cdp.send("Input.imeSetComposition", {
          text: "zhong",
          selectionStart: 5,
          selectionEnd: 5,
        });
        await window.waitForTimeout(150);
        await cdp.send("Input.imeSetComposition", {
          text: "zhongwen",
          selectionStart: 8,
          selectionEnd: 8,
        });
        await window.waitForTimeout(150);
        await cdp.send("Input.insertText", { text: "中文" });
        const imeDone = await settle(window, mode, imeAt);
        const textAfter = await window.evaluate(
          (p) => (window as unknown as { __spike755Api: Api }).__spike755Api.textAround(p),
          target,
        );
        const stats = await window.evaluate(() =>
          (window as unknown as { __spike755Api: Api }).__spike755Api.stats(),
        );
        entry.ime = {
          ms: imeDone - imeAt,
          textBefore,
          textAfter,
          committedOnce:
            (textAfter.match(/中文/g) ?? []).length === 1 && !textAfter.includes("zhong"),
          composingSkips: stats.composingSkips,
        };
        await cdp.detach();
      }

      // Narrow: the same pages at 60%, by CSS zoom and by a transform, against the same document at 100%.
      if (mode === "custom") {
        const unzoomed = await window.evaluate(() =>
          (window as unknown as { __spike755Api: Api }).__spike755Api.firstLines(),
        );
        const narrow: Record<string, unknown> = {};
        for (const how of ["zoom", "transform"] as const) {
          await window.evaluate(
            (h) => (window as unknown as { __spike755Api: Api }).__spike755Api.zoom(0.6, h),
            how,
          );
          const zoomAt = await now(window);
          await window.evaluate(() =>
            (window as unknown as { __spike755Api: Api }).__spike755Api.apply(),
          );
          await settle(window, mode, zoomAt);
          const zoomed = await window.evaluate(() =>
            (window as unknown as { __spike755Api: Api }).__spike755Api.firstLines(),
          );
          if (OUT && process.env.SPIKE755_SHOTS === "1") {
            await window.screenshot({ path: join(OUT, `${title}-${mode}-${how}60.png`) });
          }
          narrow[how] = {
            ms: (await now(window)) - zoomAt,
            pages: zoomed.length,
            pagesAt100: unzoomed.length,
            differentFirstLines: zoomed.filter((line, index) => line.text !== unzoomed[index]?.text)
              .length,
          };
          await window.evaluate(() =>
            (window as unknown as { __spike755Api: Api }).__spike755Api.zoom(1),
          );
        }
        entry.zoom60 = narrow;
      }
      results[mode] = entry;
      if (OUT) {
        await mkdir(OUT, { recursive: true });
        await writeFile(join(OUT, `${title}.json`), JSON.stringify(results, null, 1));
      }
    }
    await app.close();
  });
}
