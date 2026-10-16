/**
 * What the pages of a PDF call themselves (#901): the names its page labels
 * give them or, when it has none worth the name, the numbers its headers and
 * footers print, when those keep step with the PDF's page index. Pure: it
 * runs in the processing worker and in tests.
 */
import { normaliseText } from "../../shared/text";
import { MAX_PRINTED_PAGE_LENGTH } from "../../shared/units";

const EDGE_LINES = 3;
const MIN_PAGES_AT_ONE_OFFSET = 3;

/** A line that is only a page number: "68", "- 68 -", "Page 68 of 120", "68 / 120", "第 68 页". */
const PAGE_NUMBER_LINES = [
  /^(?:[-|·•([]\s*)?(\d{1,4})(?:\s*[-|·•)\]])?$/,
  /^(?:page|pg|p)\.?\s*(\d{1,4})(?:\s*(?:of|\/)\s*\d{1,4})?$/i,
  /^(\d{1,4})\s*(?:of|\/)\s*\d{1,4}$/i,
  /^第(\d{1,4})页(?:[,/]?共\d{1,4}页)?$/,
];

/** A page's first or last line with its number before other text ("68 | Annual Report"). */
const NUMBER_BEFORE_TEXT = /^(\d{1,3})(?:\s*[|·•-]\s*|\s+)\p{L}.{0,80}$/u;

/** As above, with the number after the text ("Annual Report 2024 | 68"). */
const NUMBER_AFTER_TEXT = /^(?=.*\p{L}).{0,80}[\p{L}\p{N}](?:\s*[|·•-]\s*|\s+)(\d{1,3})$/u;

interface Readings {
  /** From lines that are only a page number. */
  alone: Set<number>;
  /** From a number beside other text on the page's first or last line. */
  beside: Set<number>;
}

function readPage(text: string): Readings {
  const lines = text
    .split("\n")
    .map((line) => normaliseText(line))
    .filter((line) => line !== "");
  const edges =
    lines.length <= EDGE_LINES * 2
      ? lines
      : [...lines.slice(0, EDGE_LINES), ...lines.slice(-EDGE_LINES)];
  const readings: Readings = { alone: new Set(), beside: new Set() };
  for (const line of edges) {
    for (const pattern of PAGE_NUMBER_LINES) {
      const number = Number(pattern.exec(line)?.[1]);
      if (number >= 1) readings.alone.add(number);
    }
  }
  for (const line of [lines[0], lines.at(-1)]) {
    if (line === undefined) continue;
    for (const pattern of [NUMBER_BEFORE_TEXT, NUMBER_AFTER_TEXT]) {
      const number = Number(pattern.exec(line)?.[1]);
      if (number >= 1) readings.beside.add(number);
    }
  }
  return readings;
}

/**
 * The number each page prints, as text, when the offset from the PDF's page
 * index is steady: the same on at least three pages, with no page that prints
 * only a page number saying another. A page between the first and the last
 * that print it gets the number the offset gives it, though it prints none
 * itself. Null for every page when no offset is steady.
 */
export function printedNumbers(texts: readonly string[]): (string | null)[] {
  const none = texts.map(() => null);
  const pages = texts.map(readPage);
  const read = (page: Readings, number: number) =>
    page.alone.has(number) || page.beside.has(number);

  const votes = new Map<number, number>();
  pages.forEach((page, index) => {
    const offsets = new Set([...page.alone, ...page.beside].map((number) => index + 1 - number));
    for (const offset of offsets) votes.set(offset, (votes.get(offset) ?? 0) + 1);
  });
  const ranked = [...votes].sort((a, b) => b[1] - a[1]);
  const [best, runnerUp] = ranked;
  if (!best || best[1] < MIN_PAGES_AT_ONE_OFFSET || best[1] === runnerUp?.[1]) return none;

  const offset = best[0];
  if (pages.some((page, index) => page.alone.size > 0 && !page.alone.has(index + 1 - offset))) {
    return none;
  }
  const printed = pages.flatMap((page, index) => (read(page, index + 1 - offset) ? [index] : []));
  const first = printed[0] as number;
  const last = printed.at(-1) as number;
  return texts.map((_, index) =>
    index >= first && index <= last ? String(index + 1 - offset) : null,
  );
}

function nameOf(label: string | undefined): string | null {
  const name = label?.trim() ?? "";
  return name === "" || name.length > MAX_PRINTED_PAGE_LENGTH || /\p{C}/u.test(name) ? null : name;
}

/**
 * The name of each page, by the PDF's page index: from its page labels when
 * they name any page differently from its index, else from the numbers its
 * pages print. A page that names itself as its index is no different from
 * one with no name, so a PDF where none does has none (a producer's default
 * labels say nothing). `texts` are the pages' text as extracted, running
 * headers and footers still in it.
 */
export function printedPageNames(
  labels: readonly string[] | null,
  texts: readonly string[],
): (string | null)[] {
  const named = labels ? texts.map((_, index) => nameOf(labels[index])) : null;
  const names = named?.some((name, index) => name !== null && name !== String(index + 1))
    ? named
    : printedNumbers(texts);
  return names.some((name, index) => name !== null && name !== String(index + 1))
    ? names
    : texts.map(() => null);
}
