/**
 * What the binding adds to Hayagriva's output (ADR-0016), for one request: English references
 * in a Chinese style in English terms, no italics on Chinese text, and no space between a
 * Chinese name and 等, each as Zotero's citeproc-js writes them; a work cited twice in one
 * cluster written once; and Locations raised after the work in a style whose file leaves them
 * out (GB/T 7714). Pure: it only calls `render`, so tests run it on the engine directly and the
 * worker runs it on its own instance.
 */
import { CJK } from "../../shared/text";
import { type CslItem, isChineseItem } from "./csl";
import type { EngineRun } from "./types";
import type { EngineCitation, EngineCite, EngineEntry, EngineOutput, EngineRequest } from "./wasm";

export interface FormatJob {
  items: CslItem[];
  citations: EngineCitation[];
  /**
   * The style is Chinese (GB/T 7714): a reference that isn't Chinese is written with English
   * terms ("et al.", "4th ed."), as the standard asks, rather than the style's ("等", "4 版").
   */
  englishTermsForOthers: boolean;
  /**
   * The style's file leaves Locations out of its citations, and its standard raises them after
   * each work (`CitationStyle.locators`): "[3]⁵⁻⁶", "（钱学森，1990）³⁰".
   */
  raisedLocators: boolean;
}

/**
 * One citation as the engine is asked for it: a job's citation with each work once, or, when
 * Locations are raised, one of its works on its own. `works` maps each of its items to the
 * job citation's; `after` is what follows it, its page raised or another Location in brackets.
 */
interface Part {
  citation: EngineCitation;
  works: number[];
  after: EngineRun[];
}

function mergedLocator(cites: readonly EngineCite[], separator: string): Partial<EngineCite> {
  const given = cites.filter((cite) => cite.locator);
  const [first] = given;
  if (!first) return {};
  const locator = [...new Set(given.map((cite) => cite.locator))].join(separator);
  return first.label === undefined ? { locator } : { locator, label: first.label };
}

/**
 * What follows a work cited on its own in a style that raises Locations: its pages raised, as
 * the standard prints them; a slide, a section or lines, which it has no form for, in brackets
 * on the line, "（第 1 行）".
 */
function locationAfter(cite: EngineCite, work: number): EngineRun[] {
  if (!cite.locator) return [];
  if (cite.label === "page") return [{ text: cite.locator, script: "sup", item: work }];
  const text = hasCjk.test(cite.locator) ? `（${cite.locator}）` : ` (${cite.locator})`;
  return [{ text, item: work }];
}

/**
 * A job's citation as the engine is asked for it: a work it cites twice cited once, so a
 * cluster never reads "[1,1]"; and, with raised Locations, each work on its own once any has
 * a Location, so each is followed by its own.
 */
function partsOf(citation: EngineCitation, raised: boolean): Part[] {
  const firstOf = new Map<string, number>();
  const merged: { cites: [EngineCite, ...EngineCite[]]; work: number }[] = [];
  citation.items.forEach((cite, index) => {
    const at = firstOf.get(cite.id);
    const known = at === undefined ? undefined : merged[at];
    if (known) {
      known.cites.push(cite);
    } else {
      firstOf.set(cite.id, merged.length);
      merged.push({ cites: [cite], work: index });
    }
  });
  const items = merged.map(({ cites: [first, ...more], work }) => {
    const { locator: _, label: __, ...rest } = first;
    return { cite: { ...rest, ...mergedLocator([first, ...more], raised ? "," : ", ") }, work };
  });
  const noteIndex = citation.noteIndex === undefined ? {} : { noteIndex: citation.noteIndex };
  if (!raised || !items.some(({ cite }) => cite.locator)) {
    return [
      {
        citation: { ...noteIndex, items: items.map(({ cite }) => cite) },
        works: items.map(({ work }) => work),
        after: [],
      },
    ];
  }
  return items.map(({ cite, work }) => {
    const { locator: _, label: __, ...bare } = cite;
    return {
      citation: { ...noteIndex, items: [bare] },
      works: [work],
      after: locationAfter(cite, work),
    };
  });
}

/**
 * What a style writes between two works in one citation, "; " in "(Oke, 1982; Kuhn, 2012)", as
 * a render of the whole cluster shows it: the runs between the first work's and the next's.
 */
function delimiterOf(runs: readonly EngineRun[]): string {
  const firstAt = runs.findIndex((run) => run.item !== undefined);
  const first = runs[firstAt]?.item;
  const nextAt = runs.findIndex(
    (run, at) => at > firstAt && run.item !== undefined && run.item !== first,
  );
  if (firstAt < 0 || nextAt < 0) return "";
  let from = nextAt;
  while (from > firstAt && runs[from - 1]?.item === undefined) from--;
  return runs
    .slice(from, nextAt)
    .map((run) => run.text)
    .join("");
}

/** A work cited on its own in brackets, which the next is set apart from by the style's delimiter. */
const parenthesised = (runs: readonly EngineRun[]) => /[)）]$/u.test(runs.at(-1)?.text ?? "");

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

/** A space before 等 that follows Chinese, or full-width punctuation ("钱学森 等", "林衍凯， 等"). */
const spaceBeforeDeng = new RegExp(`(?<=[${CJK}\\u3000-\\u303f\\uff00-\\uffef])\\s+(?=等)`, "gu");

/**
 * No space between a Chinese name and 等 ("et al."), as Zotero's citeproc-js writes GB/T 7714
 * and Chinese academic writing has it: "(钱学森等, 1990)", "林衍凯，等". Hayagriva writes the
 * space the style's own comment asks for. The runs are read as one text, so a space at the end
 * of one run and 等 at the start of the next go too; a Latin name keeps its space.
 */
export function noSpaceBeforeDeng<T extends EngineRun>(runs: readonly T[]): T[] {
  const text = runs.map((run) => run.text).join("");
  const dropped = new Set<number>();
  for (const match of text.matchAll(spaceBeforeDeng)) {
    for (let at = match.index; at < match.index + match[0].length; at++) dropped.add(at);
  }
  if (dropped.size === 0) return [...runs];
  let start = 0;
  return runs
    .map((run) => {
      let kept = "";
      for (let index = 0; index < run.text.length; index++) {
        if (!dropped.has(start + index)) kept += run.text[index];
      }
      start += run.text.length;
      return kept === run.text ? run : { ...run, text: kept };
    })
    .filter((run) => run.text !== "");
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
  const partsByCitation = job.citations.map((citation) => partsOf(citation, job.raisedLocators));
  const parts = partsByCitation.flat();
  // A whole cluster split into its works, rendered once more as one, for its delimiter.
  const probes = partsByCitation.flatMap((each, index) =>
    each.length > 1
      ? [
          {
            index,
            citation: {
              ...(job.citations[index]?.noteIndex === undefined
                ? {}
                : { noteIndex: job.citations[index]?.noteIndex }),
              items: each.flatMap((part) => part.citation.items),
            },
          },
        ]
      : [],
  );
  const request: EngineRequest = {
    items: job.items,
    citations: [...parts.map((part) => part.citation), ...probes.map((probe) => probe.citation)],
  };
  let output = render(request);
  const others = new Set(job.items.filter((item) => !isChineseItem(item)).map((item) => item.id));
  if (job.englishTermsForOthers && others.size > 0) {
    const english = render({ ...request, locale: "en-US" });
    const englishEntries = new Map(english.bibliography.map((entry) => [entry.id, entry]));
    output = {
      ...output,
      citations: output.citations.map((runs, index) => {
        const items = request.citations[index]?.items ?? [];
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
  const delimiters = new Map(
    probes.map((probe, at) => [
      probe.index,
      delimiterOf(output.citations[parts.length + at] ?? []),
    ]),
  );
  let next = 0;
  const citations = partsByCitation.map((each, index) =>
    each.flatMap((part, at) => {
      const runs = (output.citations[next++] ?? []).map(
        ({ item, ...run }): EngineRun =>
          item === undefined ? run : { ...run, item: part.works[item] ?? item },
      );
      const delimiter = delimiters.get(index) ?? "";
      const between =
        at < each.length - 1 && delimiter !== "" && parenthesised(runs)
          ? [{ text: delimiter }]
          : [];
      return [...runs, ...part.after, ...between];
    }),
  );
  const chinese = <T extends EngineRun>(runs: readonly T[]) =>
    noSpaceBeforeDeng(uprightChinese(runs));
  return {
    ...output,
    citations: citations.map(chinese),
    bibliography: output.bibliography.map((entry) => ({
      ...entry,
      firstField: entry.firstField && chinese(entry.firstField),
      runs: chinese(entry.runs),
    })),
  };
}
