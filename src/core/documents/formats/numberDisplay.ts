/**
 * A cell's value as Excel shows it, for the sheet preview (ADR-0011). What is
 * indexed and quoted stays `formatNumber`'s (./numbers: ISO dates, the plain
 * number for formats it doesn't know), so Citations keep matching; this is
 * for the eye: dates in their own pattern, times and elapsed times,
 * scientific notation, fractions, digits between literals ("000-00-0000"),
 * accounting formats with their fill, colours ("[Red]"), conditions
 * ("[>=1000]") and a format's text section. Pure.
 */
import { generalNumber, group, splitSections } from "./numbers";

/** How a value is shown: its text, a colour its format gives it, and where a fill goes. */
export interface DisplayValue {
  text: string;
  /** A colour from the format ("[Red]"), as #RRGGBB. */
  color?: string;
  /**
   * Where the format repeats a character to fill the cell ("* " in accounting
   * formats): the text before this index sits at the cell's left edge, the
   * rest at its right.
   */
  fill?: number;
}

export interface DisplayOptions {
  date1904?: boolean;
  /** The format's built-in id, when it has one: 14 and 22 follow the system's short date. */
  builtIn?: number;
  /** The system's short date (built-in format 14), e.g. "m/d/yyyy" (see `shortDatePattern`). */
  shortDate?: string;
  /** How many characters a General number may take: about 8 in a standard column. */
  generalWidth?: number;
}

/** Excel's legacy palette: indexed colours 0–63 (a workbook's styles may replace it). */
export const INDEXED_COLORS: readonly string[] = [
  ...["000000", "FFFFFF", "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF"],
  ...["000000", "FFFFFF", "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF"],
  ...["800000", "008000", "000080", "808000", "800080", "008080", "C0C0C0", "808080"],
  ...["9999FF", "993366", "FFFFCC", "CCFFFF", "660066", "FF8080", "0066CC", "CCCCFF"],
  ...["000080", "FF00FF", "FFFF00", "00FFFF", "800080", "800000", "008080", "0000FF"],
  ...["00CCFF", "CCFFFF", "CCFFCC", "FFFF99", "99CCFF", "FF99CC", "CC99FF", "FFCC99"],
  ...["3366FF", "33CCCC", "99CC00", "FFCC00", "FF9900", "FF6600", "666699", "969696"],
  ...["003366", "339966", "003300", "333300", "993300", "993366", "333399", "333333"],
];

/** The colours a format may name. */
const NAMED_COLORS: Readonly<Record<string, string>> = {
  black: "#000000",
  white: "#FFFFFF",
  red: "#FF0000",
  green: "#00FF00",
  blue: "#0000FF",
  yellow: "#FFFF00",
  magenta: "#FF00FF",
  cyan: "#00FFFF",
};

interface Condition {
  op: string;
  value: number;
}

interface Section {
  body: string;
  color?: string;
  condition?: Condition;
}

/** A section's colour and condition, taken out of its brackets; what is left is its body. */
function parseSection(code: string): Section {
  const section: Section = { body: "" };
  let body = "";
  for (let at = 0; at < code.length; at++) {
    const char = code[at] as string;
    if (char === '"') {
      const end = code.indexOf('"', at + 1);
      const stop = end < 0 ? code.length : end;
      body += code.slice(at, stop + 1);
      at = stop;
    } else if (char === "\\" || char === "_" || char === "*") {
      body += code.slice(at, at + 2);
      at++;
    } else if (char === "[") {
      const end = code.indexOf("]", at);
      const stop = end < 0 ? code.length : end;
      const inside = code.slice(at + 1, stop);
      const condition = /^(<=|>=|<>|<|>|=)\s*(-?[\d.]+)$/.exec(inside);
      const indexed = /^color\s*(\d{1,2})$/i.exec(inside);
      if (condition) {
        section.condition = { op: condition[1] as string, value: Number(condition[2]) };
      } else if (NAMED_COLORS[inside.toLowerCase()]) {
        section.color = NAMED_COLORS[inside.toLowerCase()];
      } else if (indexed) {
        const color = INDEXED_COLORS[Number(indexed[1]) + 7];
        if (color) section.color = `#${color}`;
      } else body += code.slice(at, stop + 1); // a currency or an elapsed time: the body reads it
      at = stop;
    } else body += char;
  }
  section.body = body;
  return section;
}

function meets(value: number, { op, value: limit }: Condition): boolean {
  switch (op) {
    case "<":
      return value < limit;
    case "<=":
      return value <= limit;
    case ">":
      return value > limit;
    case ">=":
      return value >= limit;
    case "=":
      return value === limit;
    default:
      return value !== limit;
  }
}

/**
 * The section a number is shown with, and whether it shows its own minus
 * sign: not in the section for negative numbers, which shows it its own way.
 */
function chooseSection(sections: readonly Section[], value: number): [Section, boolean] {
  const [first, second, third] = sections.slice(0, 3) as [
    Section,
    Section | undefined,
    Section | undefined,
  ];
  if (first.condition || second?.condition) {
    if (first.condition && meets(value, first.condition)) return [first, true];
    if (second?.condition && meets(value, second.condition)) return [second, true];
    return [(second?.condition ? third : second) ?? first, true];
  }
  if (value < 0 && second) return [second, false];
  if (value === 0 && third) return [third, true];
  return [first, true];
}

/** One piece of a section's body. */
type Token =
  | { kind: "literal"; text: string }
  | { kind: "fill" }
  | { kind: "digit"; char: "0" | "#" | "?" }
  | { kind: "point" }
  | { kind: "comma" }
  | { kind: "percent" }
  | { kind: "exponent"; sign: "+" | "-" }
  | { kind: "slash" }
  | { kind: "text" }
  | { kind: "general" }
  | { kind: "date"; part: string };

const DATE_PART = /^(?:\[(?:h+|m+|s+)\]|AM\/PM|A\/P|y+|m+|d+|h+|s+)/i;

/** A section's body as tokens. */
function tokenize(body: string): Token[] {
  const tokens: Token[] = [];
  for (let at = 0; at < body.length; at++) {
    const char = body[at] as string;
    const rest = body.slice(at);
    const previous = tokens.at(-1);
    if (char === '"') {
      const end = body.indexOf('"', at + 1);
      const stop = end < 0 ? body.length : end;
      tokens.push({ kind: "literal", text: body.slice(at + 1, stop) });
      at = stop;
    } else if (char === "\\") {
      tokens.push({ kind: "literal", text: body[at + 1] ?? "" });
      at++;
    } else if (char === "_") {
      tokens.push({ kind: "literal", text: " " });
      at++;
    } else if (char === "*") {
      tokens.push({ kind: "fill" });
      at++;
    } else if (char === "[") {
      const end = body.indexOf("]", at);
      const stop = end < 0 ? body.length : end;
      const inside = body.slice(at + 1, stop);
      const currency = /^\$([^-]*)/.exec(inside);
      if (currency) tokens.push({ kind: "literal", text: currency[1] ?? "" });
      else if (/^(?:h+|m+|s+)$/i.test(inside)) {
        tokens.push({ kind: "date", part: `[${inside.toLowerCase()}]` });
      }
      at = stop;
    } else if (/^general/i.test(rest)) {
      tokens.push({ kind: "general" });
      at += 6;
    } else if (char === "0" && previous?.kind === "date" && previous.part.startsWith(".")) {
      previous.part += "0"; // more places of a fraction of a second
    } else if (char === "0" || char === "#" || char === "?") {
      tokens.push({ kind: "digit", char });
    } else if (char === ".") {
      // ".0" right after seconds is a fraction of a second.
      if (previous?.kind === "date" && /s/i.test(previous.part) && body[at + 1] === "0") {
        tokens.push({ kind: "date", part: "." });
      } else tokens.push({ kind: "point" });
    } else if (char === ",") tokens.push({ kind: "comma" });
    else if (char === "%") tokens.push({ kind: "percent" });
    else if ((char === "E" || char === "e") && (body[at + 1] === "+" || body[at + 1] === "-")) {
      tokens.push({ kind: "exponent", sign: body[at + 1] as "+" | "-" });
      at++;
    } else if (char === "/") tokens.push({ kind: "slash" });
    else if (char === "@") tokens.push({ kind: "text" });
    else {
      const date = DATE_PART.exec(rest);
      if (date) {
        tokens.push({ kind: "date", part: date[0] });
        at += date[0].length - 1;
      } else tokens.push({ kind: "literal", text: char });
    }
  }
  return tokens;
}

type Part = string | { fill: true };

/** Joins parts into a DisplayValue, noting where the fill goes. */
function assemble(parts: readonly Part[], color?: string): DisplayValue {
  let text = "";
  let fill: number | undefined;
  for (const part of parts) {
    if (typeof part === "string") text += part;
    else if (fill === undefined) fill = text.length;
  }
  return { text, ...(color ? { color } : {}), ...(fill !== undefined ? { fill } : {}) };
}

/** Literals, fills and percent signs outside a number's digits. */
const around = (tokens: readonly Token[]): Part[] =>
  tokens.map((token) =>
    token.kind === "literal"
      ? token.text
      : token.kind === "fill"
        ? { fill: true }
        : token.kind === "percent"
          ? "%"
          : "",
  );

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];
const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

const pad = (value: number, width: number) => String(value).padStart(width, "0");

/** Whether the "m" at `index` is minutes: after an hour, or before a second. */
function isMinute(tokens: readonly Token[], index: number): boolean {
  for (let back = index - 1; back >= 0; back--) {
    const token = tokens[back] as Token;
    if (token.kind !== "date") continue;
    if (/^\[?h/i.test(token.part)) return true;
    break;
  }
  for (let next = index + 1; next < tokens.length; next++) {
    const token = tokens[next] as Token;
    if (token.kind === "date") return /^\[?s/i.test(token.part);
  }
  return false;
}

/** A date or time serial as a section with date tokens shows it. */
function showDate(serial: number, tokens: readonly Token[], date1904: boolean): Part[] {
  const fraction = tokens.find(
    (token): token is { kind: "date"; part: string } =>
      token.kind === "date" && token.part.startsWith("."),
  );
  const places = fraction ? fraction.part.length - 1 : 0;
  // Rounded to the precision shown, as Excel rounds it.
  const perDay = 86_400 * 10 ** places;
  const ticks = Math.round(serial * perDay);
  const days = Math.floor(ticks / perDay);
  const within = ticks - days * perDay;
  const seconds = Math.floor(within / 10 ** places);
  const epoch = date1904 ? Date.UTC(1904, 0, 1) : Date.UTC(1899, 11, 30);
  const date = new Date(epoch + days * 86_400_000);
  const twelveHour = tokens.some(
    (token) => token.kind === "date" && /^(?:am\/pm|a\/p)$/i.test(token.part),
  );
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  const totalHours = days * 24 + hours;

  return tokens.map((token, index): Part => {
    switch (token.kind) {
      case "literal":
        return token.text;
      case "fill":
        return { fill: true };
      case "digit":
        return token.char;
      case "comma":
        return ",";
      case "slash":
        return "/";
      case "point":
        return ".";
      case "percent":
        return "%";
      case "date":
        break;
      default:
        return "";
    }
    const part = token.part;
    const lower = part.toLowerCase();
    if (lower.startsWith("[h")) return pad(totalHours, part.length - 2);
    if (lower.startsWith("[m")) return pad(totalHours * 60 + minutes, part.length - 2);
    if (lower.startsWith("[s"))
      return pad((totalHours * 60 + minutes) * 60 + secs, part.length - 2);
    if (lower.startsWith(".")) return `.${pad(within % 10 ** places, places)}`;
    if (lower === "am/pm") return hours < 12 ? "AM" : "PM";
    if (lower === "a/p") return (hours < 12 ? part[0] : part[2]) as string;
    switch (lower[0]) {
      case "y":
        return lower.length <= 2
          ? pad(date.getUTCFullYear() % 100, 2)
          : String(date.getUTCFullYear());
      case "m": {
        if (lower.length <= 2 && isMinute(tokens, index)) return pad(minutes, lower.length);
        if (lower.length <= 2) return pad(date.getUTCMonth() + 1, lower.length);
        const name = MONTHS[date.getUTCMonth()] as string;
        return lower.length === 3 ? name.slice(0, 3) : lower.length === 5 ? name.slice(0, 1) : name;
      }
      case "d": {
        if (lower.length <= 2) return pad(date.getUTCDate(), lower.length);
        const name = DAYS[date.getUTCDay()] as string;
        return lower.length === 3 ? name.slice(0, 3) : name;
      }
      case "h":
        return pad(twelveHour ? hours % 12 || 12 : hours, Math.min(lower.length, 2));
      case "s":
        return pad(secs, Math.min(lower.length, 2));
      default:
        return "";
    }
  });
}

/** What an unused placeholder shows: "0" a zero, "?" a space, "#" nothing. */
const unused = (char: "0" | "#" | "?") => (char === "0" ? "0" : char === "?" ? " " : "");

/**
 * Puts an integer's `digits` into placeholders from the right, keeping the
 * literals among them where they are; the leftmost placeholder takes any
 * digits left over, with their grouping commas.
 */
function fillInteger(digits: string, tokens: readonly Token[], grouped: boolean): string {
  const count = tokens.filter((token) => token.kind === "digit").length;
  const source = digits === "0" ? "" : grouped ? group(digits) : digits;
  const out: string[] = [];
  let take = source.length;
  let seen = 0;
  for (let index = tokens.length - 1; index >= 0; index--) {
    const token = tokens[index] as Token;
    if (token.kind === "literal") out.unshift(token.text);
    if (token.kind !== "digit") continue;
    seen++;
    let piece = "";
    if (seen === count) {
      piece = source.slice(0, take);
      take = 0;
    } else {
      while (take > 0) {
        const char = source[take - 1] as string;
        piece = char + piece;
        take--;
        if (char !== ",") break;
      }
    }
    out.unshift(piece || unused(token.char));
  }
  return out.join("");
}

/** Fills decimal placeholders from the left: "#"s drop trailing zeros, "?"s become spaces. */
function fillDecimals(digits: string, tokens: readonly Token[]): string {
  const kept = digits.replace(/0+$/, "").length;
  let out = "";
  let at = 0;
  for (const token of tokens) {
    if (token.kind === "literal") out += token.text;
    if (token.kind !== "digit") continue;
    out += at < kept || token.char === "0" ? (digits[at] ?? "0") : unused(token.char);
    at++;
  }
  return out;
}

/** A whole number in placeholders: padded at the start ("?" with spaces), or at the end. */
function padTo(value: string, tokens: readonly Token[], end: "start" | "end"): string {
  const digits = tokens.filter((token) => token.kind === "digit") as { char: "0" | "#" | "?" }[];
  if (value.length >= digits.length) return value;
  const padding = digits
    .slice(0, digits.length - value.length)
    .map((digit) => (end === "end" && digit.char === "0" ? " " : unused(digit.char)))
    .join("");
  return end === "start" ? padding + value : value + padding;
}

/** The fraction closest to `value` with a denominator of at most `places` digits. */
function bestFraction(value: number, places: number): [number, number] {
  const limit = 10 ** places - 1;
  let best: [number, number] = [Math.round(value), 1];
  let error = Math.abs(value - best[0]);
  for (let denominator = 2; denominator <= limit && error > 0; denominator++) {
    const numerator = Math.round(value * denominator);
    const off = Math.abs(value - numerator / denominator);
    if (off < error - 1e-12) {
      best = [numerator, denominator];
      error = off;
    }
  }
  return best;
}

/** Scientific notation: "0.00E+00". */
function showScientific(value: number, mantissa: readonly Token[], exponent: Token[]): string {
  const sign = (exponent.shift() as { sign: "+" | "-" }).sign;
  const point = mantissa.findIndex((token) => token.kind === "point");
  const integerTokens = point < 0 ? mantissa : mantissa.slice(0, point);
  const decimalTokens = point < 0 ? [] : mantissa.slice(point + 1);
  const width = Math.max(1, integerTokens.filter((token) => token.kind === "digit").length);
  const places = decimalTokens.filter((token) => token.kind === "digit").length;
  // Engineering formats ("##0.0E+0") keep the exponent a multiple of their integer digits.
  let power = value === 0 ? 0 : Math.floor(Math.log10(value));
  power = Math.floor(power / width) * width;
  let fixed = (value / 10 ** power).toFixed(places);
  if (Number(fixed) >= 10 ** width) {
    power += width;
    fixed = (value / 10 ** power).toFixed(places);
  }
  const [whole = "0", fraction = ""] = fixed.split(".");
  const exponentText = padTo(String(Math.abs(power)), exponent, "start");
  const decimals = point >= 0 ? `.${fillDecimals(fraction, decimalTokens)}` : "";
  const exponentSign = power < 0 ? "-" : sign === "+" ? "+" : "";
  return `${fillInteger(whole, integerTokens, false)}${decimals}E${exponentSign}${exponentText}`;
}

/** A fraction, "# ?/?": a whole part if the format has one, a numerator, "/", a denominator. */
function showFraction(value: number, top: readonly Token[], bottom: readonly Token[]): string {
  // The whole part is the digits before the last literal (a space) of the numerator's tokens.
  const gap = top.findLastIndex((token) => token.kind === "literal");
  const lastWhole =
    gap > 0 ? top.slice(0, gap).findLastIndex((token) => token.kind === "digit") : -1;
  const wholeTokens = lastWhole >= 0 ? top.slice(0, gap) : [];
  const separator = lastWhole >= 0 ? around(top.slice(lastWhole + 1, gap + 1)).join("") : "";
  const numeratorTokens = lastWhole >= 0 ? top.slice(gap + 1) : top;
  const fixed = bottom
    .map((token) => (token.kind === "literal" ? token.text : ""))
    .join("")
    .trim();
  let whole = lastWhole >= 0 ? Math.floor(value) : 0;
  const rest = value - whole;
  let numerator: number;
  let denominator: number;
  if (/^\d+$/.test(fixed)) {
    denominator = Number(fixed);
    numerator = Math.round(rest * denominator);
  } else {
    const places = Math.max(1, bottom.filter((token) => token.kind === "digit").length);
    [numerator, denominator] = bestFraction(rest, places);
  }
  if (lastWhole >= 0 && numerator === denominator) {
    whole += 1;
    numerator = 0;
  }
  const wholeText = padTo(whole === 0 ? "" : String(whole), wholeTokens, "start");
  if (numerator === 0 && whole !== 0) {
    // A whole number: the fraction's place is left blank.
    const blank = separator.length + numeratorTokens.length + 1 + Math.max(bottom.length, 1);
    return `${wholeText}${" ".repeat(blank)}`;
  }
  const denominatorText = /^\d+$/.test(fixed) ? fixed : padTo(String(denominator), bottom, "end");
  return `${wholeText}${separator}${padTo(String(numerator), numeratorTokens, "start")}/${denominatorText}`;
}

/** A number (not negative) as a section without date tokens shows it. */
function showNumber(value: number, tokens: readonly Token[], generalWidth: number): Part[] {
  const percents = tokens.filter((token) => token.kind === "percent").length;
  let number = value * 100 ** percents;
  const firstDigit = tokens.findIndex((token) => token.kind === "digit");
  if (firstDigit < 0) {
    // No digits: literals only (a zero section of "-"), or General among literals.
    return tokens.map((token) =>
      token.kind === "general" ? generalDisplay(number, generalWidth) : (around([token])[0] ?? ""),
    );
  }
  const slash = tokens.findIndex((token, index) => index > firstDigit && token.kind === "slash");
  const exponent = tokens.findIndex(
    (token, index) => index > firstDigit && token.kind === "exponent",
  );
  let lastDigit = tokens.findLastIndex((token) => token.kind === "digit");
  if (slash >= 0) {
    // The denominator: placeholders, or a fixed number written as literals ("?/8").
    lastDigit = slash;
    while (
      lastDigit + 1 < tokens.length &&
      (tokens[lastDigit + 1]?.kind === "digit" ||
        (tokens[lastDigit + 1]?.kind === "literal" &&
          /^\d+$/.test((tokens[lastDigit + 1] as { text: string }).text)))
    ) {
      lastDigit++;
    }
  }
  const before = around(tokens.slice(0, firstDigit));
  const after = tokens.slice(lastDigit + 1);
  if (slash >= 0) {
    const fraction = showFraction(
      number,
      tokens.slice(firstDigit, slash),
      tokens.slice(slash + 1, lastDigit + 1),
    );
    return [...before, fraction, ...around(after)];
  }
  if (exponent >= 0) {
    const scientific = showScientific(
      number,
      tokens.slice(firstDigit, exponent),
      tokens.slice(exponent, lastDigit + 1),
    );
    return [...before, scientific, ...around(after)];
  }
  const pattern = tokens.slice(firstDigit, lastDigit + 1);
  const point = pattern.findIndex((token) => token.kind === "point");
  const integerTokens = point < 0 ? pattern : pattern.slice(0, point);
  const decimalTokens = point < 0 ? [] : pattern.slice(point + 1);
  // Commas right after the digits, or just before the point, scale by a thousand each.
  let scale = 0;
  for (const token of after) {
    if (token.kind !== "comma") break;
    scale++;
  }
  for (let index = integerTokens.length - 1; integerTokens[index]?.kind === "comma"; index--) {
    scale++;
  }
  number /= 1000 ** scale;
  const grouped = integerTokens.some(
    (token, index) =>
      token.kind === "comma" &&
      integerTokens.slice(0, index).some((each) => each.kind === "digit") &&
      integerTokens.slice(index + 1).some((each) => each.kind === "digit"),
  );
  const places = decimalTokens.filter((token) => token.kind === "digit").length;
  const [whole = "0", fraction = ""] = number.toFixed(places).split(".");
  const integer = fillInteger(
    whole,
    integerTokens.filter((token) => token.kind !== "comma"),
    grouped,
  );
  const decimals = point >= 0 ? `.${fillDecimals(fraction, decimalTokens)}` : "";
  return [...before, `${integer}${decimals}`, ...around(after)];
}

/** A number as General shows it in a column `width` characters wide. */
function generalDisplay(value: number, width: number): string {
  if (!Number.isFinite(value)) return String(value);
  const plain = generalNumber(value);
  if (plain.length <= width) return plain;
  const sign = value < 0 ? 1 : 0;
  const magnitude = Math.abs(value);
  const integerDigits = magnitude >= 1 ? Math.floor(Math.log10(magnitude)) + 1 : 1;
  if (magnitude >= 1e-4 && integerDigits + sign <= width) {
    // Fewer decimal places, to fit.
    const places = Math.max(0, width - sign - integerDigits - 1);
    return generalNumber(Number(value.toFixed(places)));
  }
  // Scientific, with as many digits as fit: "1.23457E+11".
  for (let digits = Math.max(1, width - 5 - sign); digits >= 1; digits--) {
    const [mantissa = "", power = "0"] = value.toExponential(digits - 1).split("e");
    const trimmed = mantissa.includes(".") ? mantissa.replace(/\.?0+$/, "") : mantissa;
    const exponent = Number(power);
    const text = `${trimmed}E${exponent < 0 ? "-" : "+"}${pad(Math.abs(exponent), 2)}`;
    if (text.length <= width) return text;
  }
  return plain;
}

/** A number as Excel shows it with format `code` (`undefined` is General). */
export function displayNumber(
  value: number,
  code: string | undefined,
  options: DisplayOptions = {},
): DisplayValue {
  let source = code;
  if ((options.builtIn === 14 || options.builtIn === 22) && options.shortDate) {
    source = options.builtIn === 14 ? options.shortDate : `${options.shortDate} h:mm`;
  }
  const generalWidth = options.generalWidth ?? 11;
  if (!source || /^(general|@)$/i.test(source.trim())) {
    return { text: generalDisplay(value, generalWidth) };
  }
  const [section, signed] = chooseSection(splitSections(source).map(parseSection), value);
  const tokens = tokenize(section.body);
  if (tokens.some((token) => token.kind === "date")) {
    // Excel can't show a negative date.
    if (value < 0) return { text: "########" };
    return assemble(showDate(value, tokens, options.date1904 ?? false), section.color);
  }
  const shown = assemble(showNumber(Math.abs(value), tokens, generalWidth), section.color);
  // A negative value shows a minus sign first, unless its own section shows it otherwise;
  // one rounded to zero ("-0.001" as "0.00") shows none.
  if (value < 0 && signed && /[1-9]/.test(shown.text)) {
    return {
      ...shown,
      text: `-${shown.text}`,
      ...(shown.fill !== undefined ? { fill: shown.fill + 1 } : {}),
    };
  }
  return shown;
}

/** Text in a cell as its format shows it: in the format's text section, if it has one. */
export function displayText(value: string, code: string | undefined): DisplayValue {
  if (!code) return { text: value };
  const sections = splitSections(code);
  const textSection =
    sections.length >= 4
      ? sections[3]
      : sections.find(
          (section) => section.includes("@") && !/[0#?]/.test(section.replace(/"[^"]*"/g, "")),
        );
  if (textSection === undefined) return { text: value };
  const section = parseSection(textSection);
  const parts = tokenize(section.body).map((token) =>
    token.kind === "text" ? value : (around([token])[0] ?? ""),
  );
  return assemble(parts, section.color);
}

/**
 * The system's short date (Excel's built-in format 14) for a locale, as a
 * format code: "m/d/yyyy" in the US, "dd/mm/yyyy" in Britain.
 */
export function shortDatePattern(locale?: string): string {
  // 5 February 2006: a day and a month that show whether they are padded.
  const parts = new Intl.DateTimeFormat(locale, {
    year: "numeric",
    month: "numeric",
    day: "numeric",
    timeZone: "UTC",
  }).formatToParts(new Date(Date.UTC(2006, 1, 5)));
  const code = (type: string, value: string): string => {
    switch (type) {
      case "year":
        return value.length === 2 ? "yy" : "yyyy";
      case "month":
        return value.length === 2 ? "mm" : "m";
      case "day":
        return value.length === 2 ? "dd" : "d";
      case "literal":
        return value.replace(/[^\s./\-年月日,]/g, "");
      default:
        return "";
    }
  };
  return parts.map((part) => code(part.type, part.value)).join("");
}
