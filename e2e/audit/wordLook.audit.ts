/**
 * The Word look's cost (#746), measured, not asserted: CI runners are too
 * starved for wall-clock limits, so each part of #746 runs this and puts the
 * numbers in its pull request. Typing in the 304-page contract
 * (tests/helpers/longContract.ts, 2,881 Blocks), on its first page and its
 * last, keydown to the next frame; and how long the contract and the Word
 * corpus's largest file take to open, from the tab opening to their last
 * Block drawn. Results go to AUDIT_OUT/results.json under "wordLook.*".
 *
 *   npx electron-vite build --mode test
 *   AUDIT_OUT=/tmp/word-look npx playwright test -c e2e/audit/playwright.audit.config.ts wordLook.audit.ts
 */
import { readdirSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import type { CoreBridge } from "../../src/core/api";
import { longContract } from "../../tests/helpers/longContract";
import {
  close,
  installKeyLatency,
  launch,
  median,
  record,
  removeDir,
  takeKeyLatency,
  tempDir,
} from "./harness";

/** Imports the file and resolves with its Mind once every Block is in it. */
async function importWord(window: Page, path: string): Promise<string> {
  return window.evaluate(async (file) => {
    const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
    const result = await core.importWordFile({ path: file, origin: "own" });
    if (!result.ok) throw new Error(`Refused: ${result.refusal}`);
    return result.mind.id;
  }, path);
}

/** Opens the Mind in place and resolves with the ms until `last` (a selector) is drawn. */
async function openTimed(window: Page, mindId: string, done: string, count: number) {
  return window.evaluate(
    async ({ id, selector, wanted }) => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const start = performance.now();
      await core.changeLayout("main", {
        type: "open",
        item: { kind: "mind", mindId: id },
        where: { in: "place" },
      });
      for (;;) {
        if (document.querySelectorAll(selector).length >= wanted) break;
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
      await new Promise((resolve) => requestAnimationFrame(resolve));
      return Math.round(performance.now() - start);
    },
    { id: mindId, selector: done, wanted: count },
  );
}

/** The largest .docx in the Word corpus's committed set. */
function largestCorpusFile(): string {
  const root = join(__dirname, "../../eval/word");
  let largest = { path: "", size: 0 };
  const visit = (folder: string) => {
    for (const name of readdirSync(folder)) {
      const path = join(folder, name);
      const stat = statSync(path);
      if (stat.isDirectory()) visit(path);
      else if (name.endsWith(".docx") && stat.size > largest.size)
        largest = { path, size: stat.size };
    }
  };
  visit(root);
  return largest.path;
}

test("typing in the 304-page contract, and opening it and the corpus's largest file", async () => {
  const dataDir = tempDir("data");
  const files = tempDir("files");
  const contract = join(files, "Framework Services Agreement.docx");
  writeFileSync(contract, longContract());
  const launched = await launch(dataDir);
  const { window } = launched;
  const elsewhere = () =>
    window.evaluate(async () => {
      const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
      const mind = await core.createMind({ title: "Elsewhere" });
      await core.changeLayout("main", {
        type: "open",
        item: { kind: "mind", mindId: mind.id },
        where: { in: "place" },
      });
    });
  /** Opens the Mind once until it is all there, then times three opens of it. */
  const timedOpens = async (mindId: string, selector: string, count: () => Promise<number>) => {
    await openTimed(window, mindId, selector, 1);
    const blocks = await count();
    const runs: number[] = [];
    for (let run = 0; run < 3; run++) {
      await elsewhere();
      runs.push(await openTimed(window, mindId, selector, blocks));
    }
    return { blocks, runs, median: median(runs) };
  };
  try {
    const contractId = await importWord(window, contract);
    const contractOpen = await timedOpens(contractId, ".ProseMirror h1", async () => {
      // Every clause's heading in: the import has written every Block.
      await expect(window.locator(".ProseMirror h1")).toHaveCount(240, { timeout: 120_000 });
      return 240;
    });

    await installKeyLatency(window);
    const paragraphs = window.locator(".ProseMirror > p");
    const typeIn = async (index: number) => {
      const target = paragraphs.nth(index);
      await target.scrollIntoViewIfNeeded();
      await target.click({ position: { x: 20, y: 6 } });
      await window.keyboard.press("End");
      for (let key = 0; key < 30; key++) {
        await window.keyboard.type("x");
        await window.waitForTimeout(40);
      }
      return takeKeyLatency(window);
    };
    const first = await typeIn(2);
    const last = await typeIn((await paragraphs.count()) - 2);

    const corpus = largestCorpusFile();
    const corpusId = await importWord(window, corpus);
    const corpusOpen = await timedOpens(corpusId, ".ProseMirror > *", async () => {
      // The import is done once its Blocks stop coming.
      let blocks = -1;
      for (;;) {
        await window.waitForTimeout(500);
        const now = await window.locator(".ProseMirror > *").count();
        if (now === blocks) return now;
        blocks = now;
      }
    });
    const result = {
      contractOpenMs: contractOpen,
      typingFirstPage: first,
      typingLastPage: last,
      corpusFile: corpus.slice(corpus.indexOf("eval/word")),
      corpusOpenMs: corpusOpen,
    };
    record("wordLook", result);
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await close(launched);
    removeDir(dataDir);
    removeDir(files);
  }
});
