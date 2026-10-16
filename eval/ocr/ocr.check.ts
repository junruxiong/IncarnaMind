/**
 * `npm run eval:ocr` (eval/ocr/README.md): reads the scanned-page test set
 * (tests/fixtures/scans) with each OCR engine that can run here, and reports
 * each one's character error rate, the share of quotes the Citation check
 * finds in what it read, and its time a page.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";
import { evalCache } from "../lib/cache";
import { commitOf } from "../lib/commit";
import { createLog } from "../lib/log";
import { type Engine, engines } from "./lib/engines";
import { type PageScore, type SetPage, scorePage } from "./lib/score";

const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const setDir = join(root, "tests/fixtures/scans");
const cacheDir = evalCache();
/** How many times each page is read; the fastest counts, as the machine may be busy. */
const RUNS = Number(process.env.INCARNAMIND_OCR_RUNS ?? 2);

interface PageResult extends PageScore {
  page: string;
  ms: number;
  text: string;
}

interface EngineResult {
  engine: Pick<Engine, "id" | "name" | "size" | "licence">;
  skipped: string | null;
  pages: PageResult[];
}

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function summary(pages: readonly PageResult[], which: (page: SetPage) => boolean, set: SetPage[]) {
  const chosen = pages.filter((each) => {
    const page = set.find((item) => item.id === each.page);
    return page !== undefined && which(page);
  });
  const cer = chosen.reduce((sum, each) => sum + each.cer, 0) / Math.max(1, chosen.length);
  const found = chosen.reduce((sum, each) => sum + each.found, 0);
  const quotes = chosen.reduce((sum, each) => sum + each.quotes, 0);
  return { cer, found, quotes };
}

const wanted = process.env.INCARNAMIND_OCR_ENGINES?.split(",").map((each) => each.trim());

test.runIf(!wanted || engines(root).some((engine) => wanted.includes(engine.id)))(
  "OCR engines on the scanned-page set",
  async () => {
    const log = createLog();
    const set = (
      JSON.parse(await readFile(join(setDir, "set.json"), "utf8")) as { pages: SetPage[] }
    ).pages;
    const results: EngineResult[] = [];
    for (const engine of engines(root)) {
      if (wanted && !wanted.includes(engine.id)) continue;
      const { id, name, size, licence } = engine;
      const skipped = await engine.unavailable();
      const result: EngineResult = { engine: { id, name, size, licence }, skipped, pages: [] };
      results.push(result);
      if (skipped) {
        log(`${name}: skipped (${skipped})`);
        continue;
      }
      await engine.open(cacheDir, log);
      // A first read loads what the engine loads lazily; it isn't timed.
      await engine.read(join(setDir, (set[0] as SetPage).file));
      for (const page of set) {
        let best: { text: string; ms: number } | null = null;
        for (let run = 0; run < RUNS; run++) {
          const read = await engine.read(join(setDir, page.file));
          if (!best || read.ms < best.ms) best = read;
        }
        if (!best) continue;
        const score = scorePage(page, best.text);
        result.pages.push({ page: page.id, ms: best.ms, text: best.text, ...score });
        log(
          `${name} · ${page.id}: CER ${percent(score.cer)}, quotes ${score.found}/${score.quotes}, ${Math.round(best.ms)} ms`,
        );
      }
      engine.close();
    }

    const groups: [string, (page: SetPage) => boolean][] = [
      ["All", () => true],
      ["English", (page) => page.language === "en"],
      ["Chinese", (page) => page.language.startsWith("zh")],
      ["Clean", (page) => page.condition === "clean"],
      ["Noisy", (page) => page.condition === "noisy"],
      ["Table", (page) => page.kind === "table"],
      ["Held out", (page) => page.heldout === true],
    ];
    const lines: string[] = [
      `# OCR comparison, ${new Date().toISOString().slice(0, 10)}`,
      "",
      `Commit ${commitOf(root)}, ${process.platform} ${process.arch}, fastest of ${RUNS} reads a page.`,
      "",
      "## Character error rate (lower is better) and quotes found",
      "",
      `| Engine | ${groups.map(([group]) => group).join(" | ")} | Quotes found | Time a page (median) |`,
      `|---|${groups.map(() => "---").join("|")}|---|---|`,
    ];
    for (const { engine, skipped, pages } of results) {
      if (skipped) {
        lines.push(
          `| ${engine.name} | ${groups.map(() => "–").join(" | ")} | not run: ${skipped} | – |`,
        );
        continue;
      }
      const all = summary(pages, () => true, set);
      const times = pages.map((each) => each.ms).sort((a, b) => a - b);
      const median = times[Math.floor(times.length / 2)] ?? 0;
      lines.push(
        `| ${engine.name} | ${groups.map(([, which]) => percent(summary(pages, which, set).cer)).join(" | ")} | ${all.found}/${all.quotes} | ${(median / 1000).toFixed(2)} s |`,
      );
    }
    lines.push("", "## Per page", "");
    lines.push(`| Engine | ${set.map((page) => page.id).join(" | ")} |`);
    lines.push(`|---|${set.map(() => "---").join("|")}|`);
    for (const { engine, skipped, pages } of results) {
      if (skipped) continue;
      lines.push(
        `| ${engine.name} | ${set
          .map((page) => {
            const each = pages.find((item) => item.page === page.id);
            return each
              ? `${percent(each.cer)}, ${each.found}/${each.quotes}, ${(each.ms / 1000).toFixed(2)} s`
              : "–";
          })
          .join(" | ")} |`,
      );
    }
    lines.push("", "## Sizes and licences", "");
    lines.push("| Engine | What it adds | Licence |", "|---|---|---|");
    for (const { engine } of results)
      lines.push(`| ${engine.name} | ${engine.size} | ${engine.licence} |`);
    lines.push("", "## What each read", "");
    for (const { engine, pages } of results) {
      for (const page of pages) {
        lines.push(`### ${engine.name} · ${page.page}`, "", "```", page.text, "```", "");
      }
    }
    const out = join(root, "eval/results/ocr", new Date().toISOString().replace(/[:.]/g, "-"));
    await mkdir(out, { recursive: true });
    await writeFile(join(out, "report.md"), `${lines.join("\n")}\n`);
    await writeFile(join(out, "results.json"), `${JSON.stringify(results, null, 2)}\n`);
    log(`Report: ${join(out, "report.md")}`);
    console.log(
      lines.slice(0, lines.indexOf("## Sizes and licences") + 2 + results.length + 2).join("\n"),
    );
    expect(results.some((each) => each.pages.length > 0)).toBe(true);
  },
);
