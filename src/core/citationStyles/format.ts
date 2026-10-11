/**
 * What the binding adds to Hayagriva's output (ADR-0016), for one request: English references
 * in a Chinese style in English terms, and no italics on Chinese text. Pure: it only calls
 * `render`, so tests run it on the engine directly and the worker runs it on its own instance.
 */
import { CJK } from "../../shared/text";
import { type CslItem, isChineseItem } from "./csl";
import type { EngineRun } from "./types";
import type { EngineCitation, EngineEntry, EngineOutput, EngineRequest } from "./wasm";

export interface FormatJob {
  items: CslItem[];
  citations: EngineCitation[];
  /**
   * The style is Chinese (GB/T 7714): a reference that isn't Chinese is written with English
   * terms ("et al.", "4th ed."), as the standard asks, rather than the style's ("等", "4 版").
   */
  englishTermsForOthers: boolean;
}

const hasCjk = new RegExp(`[${CJK}]`, "u");

/**
 * Chinese text is never italic, as Chinese typography has it and Zotero's citeproc-js does:
 * an italic run with Chinese in it (a journal's or a book's title) comes out upright.
 */
export function uprightChinese<T extends EngineRun>(runs: readonly T[]): T[] {
  return runs.map((run) => {
    if (!run.italic || !hasCjk.test(run.text)) return run;
    const { italic: _, ...upright } = run;
    return upright as T;
  });
}

/**
 * A citation from the Chinese render with each non-Chinese reference's runs taken from the
 * English one: the delimiters and the order stay the style's.
 */
function spliceCitation(
  chinese: readonly EngineRun[],
  english: readonly EngineRun[],
  inEnglish: (item: number) => boolean,
): EngineRun[] {
  const out: EngineRun[] = [];
  const done = new Set<number>();
  for (const run of chinese) {
    const { item } = run;
    if (item === undefined || !inEnglish(item)) {
      out.push(run);
      continue;
    }
    if (done.has(item)) continue;
    done.add(item);
    const replacement = english.filter((each) => each.item === item);
    out.push(
      ...(replacement.length > 0 ? replacement : chinese.filter((each) => each.item === item)),
    );
  }
  return out;
}

/**
 * Formats a job: once in the style's own locale, and for a Chinese style with references in
 * other languages once more in English, keeping each citation's reference and each entry in
 * its own language. The bibliography keeps the style's order and numbers.
 */
export function formatJob(
  job: FormatJob,
  render: (request: EngineRequest) => EngineOutput,
): EngineOutput {
  const request: EngineRequest = { items: job.items, citations: job.citations };
  let output = render(request);
  const others = new Set(job.items.filter((item) => !isChineseItem(item)).map((item) => item.id));
  if (job.englishTermsForOthers && others.size > 0) {
    const english = render({ ...request, locale: "en-US" });
    const englishEntries = new Map(english.bibliography.map((entry) => [entry.id, entry]));
    output = {
      ...output,
      citations: output.citations.map((runs, index) => {
        const items = job.citations[index]?.items ?? [];
        return spliceCitation(runs, english.citations[index] ?? [], (item) =>
          others.has(items[item]?.id ?? ""),
        );
      }),
      bibliography: output.bibliography.map((entry): EngineEntry => {
        const inEnglish = others.has(entry.id) ? englishEntries.get(entry.id) : undefined;
        return inEnglish ? { ...entry, runs: inEnglish.runs } : entry;
      }),
    };
  }
  return {
    ...output,
    citations: output.citations.map(uprightChinese),
    bibliography: output.bibliography.map((entry) => ({
      ...entry,
      firstField: entry.firstField && uprightChinese(entry.firstField),
      runs: uprightChinese(entry.runs),
    })),
  };
}
