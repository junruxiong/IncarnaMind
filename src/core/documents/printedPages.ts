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
// A separator must stand between the number and the text: a footnote's "3 Ibid., p. 4" has only a space.
const NUMBER_BEFORE_TEXT = /^(?<n>\d{1,3})\s*[|·•]\s*(?<rest>\p{L}.{0,80})$/u;
const NUMBER_AFTER_TEXT = /^(?<rest>(?=.*\p{L}).{1,80}?[\p{L}\p{N}])\s*[|·•]\s*(?<n>\d{1,3})$/u;
const ROMAN = "(?=[ivxlcdm])m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})";
const OTHER_NAME = new RegExp(
  `^(?:(?:page|pg|p)\\.?\\s*)?(?:${ROMAN}|[a-z]{1,3}\\s*-\\s*\\d{1,4}|\\d{1,3}\\s*-\\s*\\d{1,4})$`,
  "i",
);
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

type End = (string | null)[] | "unsteady" | "absent";

// A cover can print a year or a title's number alone, so the first page is let off.
function steadyNames(numbers: readonly Set<number>[], named: readonly boolean[]): End {
  if (!numbers.some((page, index) => index > 0 && page.size > 0)) return "absent";
  const votes = new Map<number, number>();
  numbers.forEach((page, index) => {
    for (const number of page) {
      const offset = index + 1 - number;
      votes.set(offset, (votes.get(offset) ?? 0) + 1);
    }
  });
  const [best, runnerUp] = [...votes].sort((a, b) => b[1] - a[1]);
  if (!best || best[1] < MIN_PAGES || best[1] === runnerUp?.[1]) return "unsteady";

  const offset = best[0];
  const disagrees = numbers.some(
    (page, index) => index > 0 && page.size > 0 && !page.has(index + 1 - offset),
  );
  if (disagrees) return "unsteady";
  const agreeing = numbers.flatMap((page, index) => (page.has(index + 1 - offset) ? [index] : []));
  const first = agreeing[0];
  const last = agreeing.at(-1);
  if (first === undefined || last === undefined) return "unsteady";
  if (agreeing.length < MIN_SHARE * (last - first + 1)) return "unsteady";
  return numbers.map((page, index) =>
    index >= first && index <= last && (page.has(index + 1 - offset) || !named[index])
      ? String(index + 1 - offset)
      : null,
  );
}

export function printedNumbers(texts: readonly string[]): (string | null)[] {
  const none = texts.map(() => null);
  const zones = texts.map(zonesOf);
  // A page in a gap that prints "A-1" or "ii" has a name of its own, not one from the offset.
  const named = zones.map((zone) =>
    [...zone.header, ...zone.footer].some((line) => OTHER_NAME.test(line)),
  );
  const header = steadyNames(numbersAt(zones.map((zone) => zone.header)), named);
  const footer = steadyNames(numbersAt(zones.map((zone) => zone.footer)), named);
  if (header === "unsteady" || footer === "unsteady") return none;
  if (header === "absent") return footer === "absent" ? none : footer;
  if (footer === "absent") return header;
  const agree = header.every(
    (name, index) => name === null || footer[index] === null || name === footer[index],
  );
  return agree ? footer.map((name, index) => name ?? header[index] ?? null) : none;
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
