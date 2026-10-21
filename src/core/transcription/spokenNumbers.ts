/**
 * Numbers in a transcript are written one way (#1508): as digits for figures, percentages, money
 * and dates, so a check of "6.8%" finds the line that says "six point eight percent". An engine
 * writes some numbers as digits and others as words, whichever it heard; this makes them one.
 *
 * A small number is left as the word it is in running speech ("one of the", "two weeks"), unless a
 * unit says it is a figure: a percent, money, a month or a date.
 */

const UNITS: ReadonlyMap<string, number> = new Map(
  [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
  ].map((word, value) => [word, value]),
);
const TENS: ReadonlyMap<string, number> = new Map(
  ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"].map((word, at) => [
    word,
    (at + 2) * 10,
  ]),
);
const SCALES: ReadonlyMap<string, number> = new Map([
  ["thousand", 1e3],
  ["million", 1e6],
  ["billion", 1e9],
  ["trillion", 1e12],
]);
const ORDINALS: ReadonlyMap<string, number> = new Map([
  ...[
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
  ].map((word, at): [string, number] => [word, at + 1]),
  ["twentieth", 20],
  ["thirtieth", 30],
]);
const MONTHS =
  "january|february|march|april|may|june|july|august|september|october|november|december";
const MONEY = /^(?:dollars?|euros?|pounds?|yuan|cents?|pence|rupees?|yen|bucks)$/;

const word = (token: string) => token.toLowerCase();

const SUFFIX: ReadonlyMap<number, string> = new Map([
  [1, "st"],
  [2, "nd"],
  [3, "rd"],
]);
const ordinalSuffix = (value: number): string =>
  value % 100 >= 11 && value % 100 <= 13 ? "th" : (SUFFIX.get(value % 10) ?? "th");

const grouped = (value: number): string =>
  value >= 1000 ? String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ",") : String(value);

const DIGIT_WORD = "zero|one|two|three|four|five|six|seven|eight|nine";
const NUMBER_WORD = `${[...UNITS.keys()].join("|")}|${[...TENS.keys()].join("|")}|hundred|${[...SCALES.keys()].join("|")}`;
// "a hundred", "one hundred and twenty-eight thousand", "nine point three", "six percent".
const RUN = new RegExp(
  `\\b(?:a )?(?:${NUMBER_WORD})(?:[ -](?:${NUMBER_WORD})|(?<=hundred|thousand|million|billion|trillion) and (?:${NUMBER_WORD}))*(?: point(?: (?:${DIGIT_WORD}))+(?: (?:${[...SCALES.keys()].join("|")}))?)?(?: (?:percent|percentage|per cent))?(?![A-Za-z-])`,
  "gi",
);

type Kind = "unit" | "teen" | "tens" | "hundred" | "scale";
const kindOf = (each: string): Kind =>
  each === "hundred"
    ? "hundred"
    : SCALES.has(each)
      ? "scale"
      : TENS.has(each)
        ? "tens"
        : (UNITS.get(each) ?? 0) >= 10
          ? "teen"
          : "unit";

/** Whether `next` goes on the number `last` ended with, as "twenty" goes before "eight" and "eight" before "hundred". */
function goesOn(last: Kind | null, next: Kind): boolean {
  if (last === null || last === "scale") return true;
  if (last === "tens") return next === "unit" || next === "hundred" || next === "scale";
  if (last === "unit" || last === "teen") return next === "hundred" || next === "scale";
  return next !== "hundred";
}

interface Figure {
  value: number;
  /** The one million, billion or trillion the number ends with, kept as a word: "18.4 million". */
  word: string | null;
  scaled: boolean;
  /** The number words are exactly "one" to "nine", which running speech uses as words. */
  small: boolean;
}

/** The numbers a run of number words says: "four hundred twelve" is one, "four five" is two. */
function figuresOf(words: readonly string[]): Figure[] {
  const out: Figure[] = [];
  let total = 0;
  let current = 0;
  let last: Kind | null = null;
  let scaled = false;
  let count = 0;
  let scales = 0;
  let lastScale = "";
  const close = () => {
    if (count > 0) {
      out.push({
        value: total + current,
        word:
          scales === 1 && last === "scale" && (SCALES.get(lastScale) ?? 0) >= 1e6
            ? lastScale
            : null,
        scaled,
        small: count === 1 && last === "unit" && total + current < 10,
      });
    }
    total = 0;
    current = 0;
    last = null;
    scaled = false;
    count = 0;
    scales = 0;
  };
  for (const each of words) {
    const kind = kindOf(each);
    if (!goesOn(last, kind)) close();
    if (kind === "hundred") current = (current || 1) * 100;
    else if (kind === "scale") {
      total += (current || 1) * (SCALES.get(each) ?? 1);
      current = 0;
      scaled = true;
      scales++;
      lastScale = each;
    } else current += TENS.get(each) ?? UNITS.get(each) ?? 0;
    last = kind;
    count++;
  }
  close();
  return out;
}

/** A year said as two numbers: "twenty twenty-five", "nineteen ninety-nine", "twenty oh five". */
const YEAR = new RegExp(
  `\\b(nineteen|twenty)[ -]((?:${[...TENS.keys()].join("|")})(?:[- ](?:${DIGIT_WORD}))?|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|oh[ -](?:${DIGIT_WORD}))(?![A-Za-z-])`,
  "gi",
);

function yearOf(century: string, rest: string): string {
  const [head = "", tail] = word(rest).split(/[ -]/);
  const second =
    head === "oh"
      ? (UNITS.get(tail ?? "") ?? 0)
      : (TENS.get(head) ?? UNITS.get(head) ?? 0) + (UNITS.get(tail ?? "") ?? 0);
  return String((word(century) === "nineteen" ? 19 : 20) * 100 + second);
}

const ORDINAL_WORD = `(?:${[...ORDINALS.keys()].join("|")}|twenty[- ](?:${[...ORDINALS.keys()].slice(0, 9).join("|")})|thirty[- ]first)`;
/** An ordinal that names a day: "the thirtieth of November", "November thirtieth". */
const DAY = new RegExp(
  `\\b${ORDINAL_WORD}(?= of (?:${MONTHS})\\b)|(?<=\\b(?:${MONTHS}) )${ORDINAL_WORD}(?![A-Za-z-])`,
  "gi",
);

function dayOf(match: string): string {
  const lower = word(match);
  const [first = "", second] = lower.split(/[- ]/);
  const value =
    second === undefined
      ? ORDINALS.get(first)
      : (first === "thirty" ? 30 : 20) + (ORDINALS.get(second) ?? 0);
  return value === undefined ? match : `${value}${ordinalSuffix(value)}`;
}

function runToDigits(run: string, after: string): string {
  let text = run;
  let percent = false;
  const percentAt = /(?: (?:percent|percentage|per cent))$/i.exec(text);
  if (percentAt) {
    percent = true;
    text = text.slice(0, percentAt.index);
  }
  const [integer = "", places] = text.split(/ point /i);
  const all = integer.toLowerCase().replace(/^a /, "");
  const figures = figuresOf(all.split(/[ -]+/).filter((each) => each !== "and"));
  const scaleWord = places?.split(" ").find((each) => SCALES.has(word(each)));
  const decimals =
    places === undefined
      ? ""
      : places
          .split(" ")
          .filter((each) => UNITS.has(word(each)))
          .map((each) => UNITS.get(word(each)) ?? 0)
          .join("");
  const money = MONEY.test(word(/^ +([A-Za-z]+)/.exec(after)?.[1] ?? ""));
  const written = figures.map((figure, at) => {
    const last = at === figures.length - 1;
    const places = last && decimals !== "" ? `.${decimals}` : "";
    const isFigure =
      figure.value >= 10 || places !== "" || figure.scaled || (last && percent) || (last && money);
    if (!isFigure) return null;
    if (last && scaleWord !== undefined) {
      return `${figure.value}${places} ${word(scaleWord)}`;
    }
    if (figure.word !== null) {
      return `${figure.value / (SCALES.get(figure.word) ?? 1)}${places} ${figure.word}`;
    }
    return `${grouped(figure.value)}${places}`;
  });
  // A number said only in words: left as it is, unless every number of the run is a figure.
  if (written.some((each) => each === null)) return run;
  return `${written.join(" ")}${percent ? "%" : ""}`;
}

/** English: number words as digits; see the module's note on which. */
function englishNumbers(text: string): string {
  const dated = text
    .replace(YEAR, (_whole, century: string, rest: string) => yearOf(century, rest))
    .replace(DAY, dayOf);
  return dated.replace(RUN, (run: string, offset: number) =>
    runToDigits(run, dated.slice(offset + run.length, offset + run.length + 24)),
  );
}

const ZH_DIGITS: ReadonlyMap<string, number> = new Map([
  ["零", 0],
  ["〇", 0],
  ["一", 1],
  ["二", 2],
  ["两", 2],
  ["三", 3],
  ["四", 4],
  ["五", 5],
  ["六", 6],
  ["七", 7],
  ["八", 8],
  ["九", 9],
]);
const ZH_SMALL: ReadonlyMap<string, number> = new Map([
  ["十", 10],
  ["百", 100],
  ["千", 1000],
]);
const ZH_BIG: ReadonlyMap<string, number> = new Map([
  ["万", 1e4],
  ["亿", 1e8],
]);
const ZH_CHAR = "零〇一二三四五六七八九两十百千万亿";

/** A Chinese numeral's value: "一百二十八万四千" is 1,284,000. */
function zhValue(text: string): number {
  let total = 0;
  let section = 0;
  let digit = 0;
  for (const char of text) {
    const each = ZH_DIGITS.get(char);
    if (each !== undefined) {
      digit = each;
    } else if (ZH_SMALL.has(char)) {
      section += (digit || (char === "十" ? 1 : 0)) * (ZH_SMALL.get(char) ?? 1);
      digit = 0;
    } else if (ZH_BIG.has(char)) {
      total += (section + digit) * (ZH_BIG.get(char) ?? 1);
      section = 0;
      digit = 0;
    }
  }
  return total + section + digit;
}

const ZH_MEASURES = "年月日号元块%";

/** Chinese: numerals as digits, where a unit, a date or a size says they are a figure. */
function chineseNumbers(text: string): string {
  const whole = `[${ZH_CHAR}]+(?:点[零〇一二三四五六七八九]+[万亿]?)?`;
  const pattern = new RegExp(`(百分之)?(${whole})`, "gu");
  return text.replace(
    pattern,
    (match: string, percent: string | undefined, number: string, offset: number) => {
      const after = text.charAt(offset + match.length);
      const [integer = "", fraction] = number.split("点");
      const scale = fraction === undefined ? "" : (/[万亿]$/.exec(fraction)?.[0] ?? "");
      const decimal = fraction?.slice(0, fraction.length - scale.length);
      if (percent) {
        const value = zhValue(integer);
        return `${grouped(value)}${decimal === undefined ? "" : `.${[...decimal].map((each) => ZH_DIGITS.get(each) ?? 0).join("")}`}%`;
      }
      const hasUnit = [...integer].some((char) => ZH_SMALL.has(char) || ZH_BIG.has(char));
      // A year said digit by digit: "二零二五年".
      if (!hasUnit && integer.length >= 4 && after === "年") {
        return `${[...integer].map((each) => ZH_DIGITS.get(each) ?? 0).join("")}${decimal ?? ""}`;
      }
      const value = zhValue(integer);
      const figure =
        (hasUnit && !(integer === "十" || integer === "百" || integer === "千")) ||
        value > 10 ||
        (value > 0 && ZH_MEASURES.includes(after));
      if (
        !figure ||
        (integer.length === 1 && ZH_DIGITS.has(integer) && !ZH_MEASURES.includes(after))
      ) {
        return match;
      }
      const places =
        decimal === undefined
          ? ""
          : `.${[...decimal].map((each) => ZH_DIGITS.get(each) ?? 0).join("")}`;
      if (scale !== "") return `${zhValue(integer)}${places}${scale}`;
      // "三百万" keeps its 万 or 亿, as written numbers do.
      const big = integer.at(-1);
      if (big !== undefined && ZH_BIG.has(big)) {
        const before = zhValue(integer.slice(0, -1));
        if (!/[万亿]/.test(integer.slice(0, -1))) return `${before}${places}${big}`;
      }
      return `${grouped(value)}${places}`;
    },
  );
}

/** What an engine said, with its numbers written as digits where they are figures. */
export function writtenNumbers(text: string): string {
  return chineseNumbers(englishNumbers(text));
}
