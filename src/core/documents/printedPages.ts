/**
 * The names a PDF's pages give themselves (#901). Nothing is guessed: a page is named by its
 * page label, or by the number its header or footer prints when that holds at one steady offset
 * from the PDF's page index; otherwise it is known by its place in the PDF.
 */
import { normaliseText } from "../../shared/text";
import { isPrintedName } from "../../shared/units";

const ZONE_LINES = 3;
const MIN_PAGES = 3;
const MIN_SHARE = 0.6;

const BARE_NUMBER = [
  /^(?<n>\d{1,4})$/,
  /^([-|·•])\s*(?<n>\d{1,4})\s*\1$/,
  /^\(\s*(?<n>\d{1,4})\s*\)$/,
  /^\[\s*(?<n>\d{1,4})\s*\]$/,
  /^(?:page|pg|p)\.?\s*(?<n>\d{1,4})(?:\s*(?:of|\/)\s*\d{1,4})?$/i,
  /^(?<n>\d{1,4})\s*(?:of|\/)\s*\d{1,4}$/,
  /^第(?<n>\d{1,4})页(?:,?共\d{1,4}页)?$/,
];
const NUMBER_BEFORE_TEXT = /^(?<n>\d{1,3})(?:\s*[|·•]\s*|\s+)(?<rest>\p{L}.{0,80})$/u;
const NUMBER_AFTER_TEXT =
  /^(?<rest>(?=.*\p{L}).{1,80}?[\p{L}\p{N}])(?:\s*[|·•]\s*|\s+)(?<n>\d{1,3})$/u;
const LABEL_OF_ANOTHER_NUMBER =
  /(?:^|\s)(?:figure|fig|table|chart|exhibit|section|chapter|appendix|part|note|plate|equation|article|clause|schedule|annex|step|item|no)\.?$/i;

interface Reading {
  number: number;
  shape: string | null;
}

function readLine(line: string, outermost: boolean): Reading | null {
  for (const pattern of BARE_NUMBER) {
    const number = Number(pattern.exec(line)?.groups?.n);
    if (number >= 1) return { number, shape: null };
  }
  if (!outermost) return null;
  const before = NUMBER_BEFORE_TEXT.exec(line)?.groups;
  const after = NUMBER_AFTER_TEXT.exec(line)?.groups;
  if (Boolean(before) === Boolean(after)) return null;
  if (after && LABEL_OF_ANOTHER_NUMBER.test(after.rest ?? "")) return null;
  const number = Number((before ?? after)?.n);
  return number >= 1 ? { number, shape: line.toLowerCase().replace(/\d+/g, "#") } : null;
}

function zonesOf(text: string): { header: string[]; footer: string[] } {
  const lines = text.split("\n").filter((line) => line.trim() !== "");
  const head = Math.min(ZONE_LINES, Math.floor(lines.length / 2));
  const foot = Math.min(ZONE_LINES, lines.length - head);
  return {
    header: lines.slice(0, head).map(normaliseText),
    footer: lines
      .slice(lines.length - foot)
      .reverse()
      .map(normaliseText),
  };
}

function numbersAt(zones: readonly string[][]): Set<number>[] {
  const readings = zones.map((lines) =>
    lines.flatMap((line, depth) => readLine(line, depth === 0) ?? []),
  );
  const pagesWithShape = new Map<string, number>();
  for (const page of readings) {
    for (const shape of new Set(page.map((reading) => reading.shape))) {
      if (shape !== null) pagesWithShape.set(shape, (pagesWithShape.get(shape) ?? 0) + 1);
    }
  }
  // A running header or footer has the same shape on page after page; a heading, a caption or a
  // footnote doesn't.
  return readings.map(
    (page) =>
      new Set(
        page
          .filter(
            (reading) =>
              reading.shape === null || (pagesWithShape.get(reading.shape) ?? 0) >= MIN_PAGES,
          )
          .map((reading) => reading.number),
      ),
  );
}

function steadyNames(numbers: readonly Set<number>[]): (string | null)[] | null {
  const votes = new Map<number, number>();
  numbers.forEach((page, index) => {
    for (const number of page) {
      const offset = index + 1 - number;
      votes.set(offset, (votes.get(offset) ?? 0) + 1);
    }
  });
  const [best, runnerUp] = [...votes].sort((a, b) => b[1] - a[1]);
  if (!best || best[1] < MIN_PAGES || best[1] === runnerUp?.[1]) return null;

  const offset = best[0];
  // A cover can print a year or a title's number alone, so the first page is let off.
  const disagrees = numbers.some(
    (page, index) => index > 0 && page.size > 0 && !page.has(index + 1 - offset),
  );
  if (disagrees) return null;
  const agreeing = numbers.flatMap((page, index) => (page.has(index + 1 - offset) ? [index] : []));
  const first = agreeing[0];
  const last = agreeing.at(-1);
  if (first === undefined || last === undefined) return null;
  if (agreeing.length < MIN_SHARE * (last - first + 1)) return null;
  return numbers.map((_, index) =>
    index >= first && index <= last ? String(index + 1 - offset) : null,
  );
}

export function printedNumbers(texts: readonly string[]): (string | null)[] {
  const zones = texts.map(zonesOf);
  const header = steadyNames(numbersAt(zones.map((zone) => zone.header)));
  const footer = steadyNames(numbersAt(zones.map((zone) => zone.footer)));
  if (!header || !footer) return header ?? footer ?? texts.map(() => null);
  const agree = header.every(
    (name, index) => name === null || footer[index] === null || name === footer[index],
  );
  return agree ? footer.map((name, index) => name ?? header[index] ?? null) : texts.map(() => null);
}

function nameOf(label: string | undefined): string | null {
  const name = label?.trim();
  return isPrintedName(name) ? name : null;
}

const differsFromIndex = (names: readonly (string | null)[]) =>
  names.some((name, index) => name !== null && name !== String(index + 1));

/**
 * The name of each page, by the PDF's page index: from its page labels when they name any page
 * differently from its index, else from the numbers its pages print. `texts` are the pages' text
 * as extracted, running headers and footers still in it.
 */
export function printedPageNames(
  labels: readonly string[] | null,
  texts: readonly string[],
): (string | null)[] {
  const fromLabels = labels && texts.map((_, index) => nameOf(labels[index]));
  const names = fromLabels && differsFromIndex(fromLabels) ? fromLabels : printedNumbers(texts);
  return differsFromIndex(names) ? names : texts.map(() => null);
}
