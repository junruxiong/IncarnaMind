/**
 * Citation Locations (ADR-0011): where in a Document a Citation points, in
 * the unit its readers use, and its short label: "p. 4", "slide 4",
 * "Revenue, rows 12–14", "§ 2.1 Sensitivity", "lines 120–134". Pure, shared
 * by the core (the check, the prompt, exports) and the renderer (the
 * Citation card, the viewer's mark).
 *
 * A Location is worked out from the Units a Citation names and where in
 * them its quote was found: for blocks of rows or lines, the rows or lines
 * the quote covers; for sections, the heading the quote sits under, and for
 * a quote of a Word comment, the comment's author too ("§ 2 Costs, comment
 * by Reviewer", #76).
 */
import type { CitationAttributes, CitationLocation } from "../core/api";
import { type MessageKey, type MessageParams, translate } from "./i18n";
import { findQuote, type MatchOptions, type TextRange } from "./quoteMatch";
import { anchorsOf, lineAt, parseCellRef, type UnitText } from "./units";

/** "12" or "12–14". */
const span = (from: number, to: number) => (to !== from ? `${from}–${to}` : `${from}`);

const isCount = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value >= 1;

/** A stored Location, checked; null if it isn't one. */
export function parseLocation(value: unknown): CitationLocation | null {
  if (typeof value === "string") {
    try {
      return parseLocation(JSON.parse(value));
    } catch {
      return null;
    }
  }
  if (typeof value !== "object" || value === null) return null;
  const location = value as Record<string, unknown>;
  const { kind, from, to } = location;
  switch (kind) {
    case "page":
    case "slide":
    case "lines":
      return isCount(from) && isCount(to) && to >= from ? { kind, from, to } : null;
    case "rows":
      if (!isCount(from) || !isCount(to) || to < from) return null;
      return {
        kind,
        sheet: typeof location.sheet === "string" ? location.sheet : null,
        from,
        to,
      };
    case "section": {
      const comment = location.comment;
      const author =
        typeof comment === "object" && comment !== null
          ? (comment as Record<string, unknown>).author
          : undefined;
      return {
        kind,
        heading: typeof location.heading === "string" ? location.heading : null,
        ...(location.notes === true ? { notes: true } : {}),
        ...(author !== undefined
          ? { comment: { author: typeof author === "string" && author ? author : null } }
          : {}),
      };
    }
    default:
      return null;
  }
}

/**
 * A Citation's Location: the one stored with it, or for a Citation from
 * before Locations, its pages (null for a whole TXT or Markdown file).
 */
export function citationLocation(
  attributes: Partial<Pick<CitationAttributes, "pageFrom" | "pageTo">> & { location?: unknown },
): CitationLocation | null {
  const stored = parseLocation(attributes.location);
  if (stored) return stored;
  const { pageFrom, pageTo } = attributes;
  if (!isCount(pageFrom)) return null;
  return {
    kind: "page",
    from: pageFrom,
    to: isCount(pageTo) && pageTo >= pageFrom ? pageTo : pageFrom,
  };
}

/** Words a Location in a language: `t` is the renderer's `t` or the core's `translate`. */
export type Translate = (key: MessageKey, params?: MessageParams) => string;

/** A Location's short label, e.g. "slide 4" or "Revenue, rows 12–14". */
export function formatLocation(location: CitationLocation, t: Translate): string {
  switch (location.kind) {
    case "page":
      return t("location.page", { range: span(location.from, location.to) });
    case "slide":
      return t(location.to !== location.from ? "location.slide.other" : "location.slide.one", {
        range: span(location.from, location.to),
      });
    case "rows": {
      const range = span(location.from, location.to);
      const plural = location.to !== location.from;
      return location.sheet
        ? t(plural ? "location.sheetRows.other" : "location.sheetRows.one", {
            sheet: location.sheet,
            range,
          })
        : t(plural ? "location.rows.other" : "location.rows.one", { range });
    }
    case "lines":
      return t(location.to !== location.from ? "location.lines.other" : "location.lines.one", {
        range: span(location.from, location.to),
      });
    case "section": {
      const section = location.notes
        ? t("location.section.notes")
        : location.heading
          ? t("location.section", { heading: location.heading })
          : t("location.section.start");
      if (!location.comment) return section;
      return location.comment.author
        ? t("location.section.comment", { section, author: location.comment.author })
        : t("location.section.comment.anonymous", { section });
    }
  }
}

/** A Location's label in English: for the model's prompt and for Citations copied as text. */
export const englishLocation = (location: CitationLocation): string =>
  formatLocation(location, (key, params) => translate("en", key, params));

/** The kinds of Location the card's "Open …" button names: the page, the slide, the section… */
export const openKind = (location: CitationLocation | null) =>
  location === null ? "document" : location.kind;

/** Whether a label reads "in …" (a section, a range of rows) rather than "on …" (a page, a slide, lines). */
export const readsIn = (location: CitationLocation) =>
  location.kind === "section" || location.kind === "rows";

/** The whole of one Unit as a Location: what the prompt marks each Unit with. */
export function unitLocation(unit: UnitText): CitationLocation | null {
  const number = unit.page;
  const label = unit.label ?? null;
  switch (unit.kind ?? "page") {
    case "page":
    case "slide":
      return number === null
        ? null
        : ({ kind: unit.kind ?? "page", from: number, to: number } as CitationLocation);
    case "rows":
      return isCount(label?.from) && isCount(label?.to)
        ? { kind: "rows", sheet: label?.sheet ?? null, from: label.from, to: label.to }
        : null;
    case "lines":
      return isCount(label?.from) && isCount(label?.to)
        ? { kind: "lines", from: label.from, to: label.to }
        : null;
    case "section":
      return {
        kind: "section",
        heading: label?.path?.at(-1) ?? null,
        ...(label?.notes ? { notes: true } : {}),
      };
    default:
      return null;
  }
}

/**
 * Where a quote is in the text of Units (see `joinUnits`), matched as the
 * Citation check matches it, number formatting normalised in sheets' rows,
 * and in slides when the quote isn't found as it is (see `numbersIn`; and
 * lost f-ligatures forgiven with `lostLigatures`, of their Document); null
 * if it isn't there.
 */
export function quoteInUnits(
  units: readonly UnitText[],
  quote: string,
  options: Pick<MatchOptions, "lostLigatures"> = {},
): TextRange[] | null {
  const { text } = joinUnits(units);
  if (!quote || text.trim() === "") return null;
  return findQuote(text, quote, { ...options, numbers: numbersIn(units) });
}

/**
 * How number formatting is matched in Units: normalised in sheets' rows; in
 * slides, whose charts are stored as the values the file caches and drawn in
 * its number formats, only when a quote isn't found as it is (#76).
 */
export function numbersIn(units: readonly Pick<UnitText, "kind">[]): MatchOptions["numbers"] {
  if (units.some((unit) => unit.kind === "rows")) return true;
  return units.some((unit) => unit.kind === "slide") ? "if-needed" : false;
}

/**
 * The anchors (cells, paragraphs…) a quote covers in Units, by Unit: where it
 * was found (`ranges`, in their joined text) mapped through each Unit's anchors.
 */
export function anchorsCovered(
  units: readonly UnitText[],
  ranges: readonly TextRange[],
): { unit: UnitText; target: string }[] {
  const { starts } = joinUnits(units);
  const covered: { unit: UnitText; target: string }[] = [];
  for (const part of partsByUnit(units, starts, ranges)) {
    const unit = units[part.unit] as UnitText;
    for (const anchor of anchorsOf(unit)) {
      if (anchor.start < part.end && anchor.end > part.start) {
        covered.push({ unit, target: anchor.target });
      }
    }
  }
  return covered;
}

/** The text of cited Units as the check reads it, joined by line breaks, with where each starts. */
export function joinUnits(units: readonly UnitText[]): { text: string; starts: number[] } {
  let text = "";
  const starts: number[] = [];
  units.forEach((unit, index) => {
    if (index > 0) text += "\n";
    starts.push(text.length);
    text += unit.text;
  });
  return { text, starts };
}

/** A span of the joined text, cut into the part inside each Unit (offsets relative to it). */
function partsByUnit(
  units: readonly UnitText[],
  starts: readonly number[],
  ranges: readonly TextRange[],
): { unit: number; start: number; end: number }[] {
  const parts: { unit: number; start: number; end: number }[] = [];
  for (const range of ranges) {
    units.forEach((unit, index) => {
      const offset = starts[index] as number;
      const start = Math.max(range.start, offset) - offset;
      const end = Math.min(range.end, offset + unit.text.length) - offset;
      if (start < end) parts.push({ unit: index, start, end });
    });
  }
  return parts;
}

/**
 * The Word comment a section's text has at `offset`, by its author (see
 * `UnitLabel.comments`); null when the offset is in the section's own text.
 */
function commentAt(unit: UnitText, offset: number): { author: string | null } | null {
  for (const anchor of anchorsOf(unit)) {
    if (anchor.start > offset || offset >= anchor.end || !anchor.target.startsWith("comment")) {
      continue;
    }
    const comment = unit.label?.comments?.find((each) => each.target === anchor.target);
    return { author: comment?.author ?? null };
  }
  return null;
}

/** A request for a narrower range than the Units cited: rows or lines the model named. */
export interface RowsOrLines {
  from: number;
  to: number;
}

/**
 * The Location of cited Units (consecutive, of one kind): with `ranges`
 * (where the quote was found in their joined text, see `joinUnits`), narrowed
 * to what the quote covers; otherwise to `requested`, or the Units' own span.
 * Null for a whole TXT or Markdown file stored before Units.
 */
export function locationOf(
  units: readonly UnitText[],
  ranges: readonly TextRange[] | null,
  requested: RowsOrLines | null = null,
): CitationLocation | null {
  const first = units[0];
  const last = units.at(-1);
  if (!first || !last) return null;
  const kind = first.kind ?? "page";
  const { starts } = joinUnits(units);
  const parts = ranges ? partsByUnit(units, starts, ranges) : [];
  const clamp = (whole: RowsOrLines): RowsOrLines => {
    if (!requested) return whole;
    const from = Math.max(whole.from, requested.from);
    const to = Math.min(whole.to, requested.to);
    return from <= to ? { from, to } : whole;
  };
  switch (kind) {
    case "page":
    case "slide":
      return first.page === null || last.page === null
        ? null
        : ({ kind, from: first.page, to: last.page } as CitationLocation);
    case "text":
      return null;
    case "section": {
      const where = units[parts[0]?.unit ?? 0] ?? first;
      const section = unitLocation(where);
      const comment = parts[0] ? commentAt(where, parts[0].start) : null;
      return section?.kind === "section" && comment ? { ...section, comment } : section;
    }
    case "rows": {
      const whole = { from: first.label?.from ?? 1, to: last.label?.to ?? first.label?.to ?? 1 };
      let rows: number[] = [];
      const headerRows: number[] = [];
      for (const part of parts) {
        const unit = units[part.unit] as UnitText;
        const headerRow = unit.label?.header ? unit.label.rows?.[0] : undefined;
        for (const anchor of anchorsOf(unit)) {
          if (anchor.start >= part.end || anchor.end <= part.start) continue;
          const row = parseCellRef(anchor.target)?.row;
          if (row === undefined) continue;
          (row === headerRow ? headerRows : rows).push(row);
        }
      }
      if (rows.length === 0) rows = headerRows;
      const range =
        rows.length > 0 ? { from: Math.min(...rows), to: Math.max(...rows) } : clamp(whole);
      return { kind: "rows", sheet: first.label?.sheet ?? null, ...range };
    }
    case "lines": {
      const whole = { from: first.label?.from ?? 1, to: last.label?.to ?? first.label?.to ?? 1 };
      const head = parts[0];
      const tail = parts.at(-1);
      if (!head || !tail) return { kind: "lines", ...clamp(whole) };
      const startUnit = units[head.unit] as UnitText;
      const endUnit = units[tail.unit] as UnitText;
      return {
        kind: "lines",
        from: lineAt(startUnit.text, startUnit.label ?? null, head.start),
        to: lineAt(endUnit.text, endUnit.label ?? null, tail.end - 1),
      };
    }
  }
}

/** A Location as the model may give it in a Citation record, read loosely. */
export type RequestedLocation =
  | { kind: "page" | "slide" | "lines"; from: number; to: number }
  | { kind: "rows"; sheet: string | null; from: number; to: number }
  | { kind: "section"; heading: string };

/**
 * What a comment's Location adds after its section, as its label words it in
 * English or Chinese (", comment by Reviewer", "，王丽华 的批注"): the
 * section alone names the Unit (#76).
 */
const A_COMMENT = /\s*(?:[,，;(]\s*(?:a\s+)?comment(?:\s+by\b.*)?|[,，]\s*[^,，]*批注)\)?$/iu;

const RANGE = String.raw`(\d{1,7})(?:\s*(?:-|–|—|to|and)\s*(\d{1,7}))?`;
const range = (from: string | undefined, to: string | undefined) => {
  const a = Number(from);
  const b = to === undefined ? a : Number(to);
  return { from: Math.min(a, b), to: Math.max(a, b) };
};

/**
 * Reads a Location written as its label ("slide 4", "Revenue, rows 12–14",
 * "§ 2.1 Sensitivity", "lines 120–134", "p. 4"), or as a cell range
 * ("Revenue!A12:F14"). Null if it can't be read.
 */
export function parseRequestedLocation(input: string): RequestedLocation | null {
  const text = input
    .trim()
    .replace(/^[[("'“]+|[\])"'”.]+$/g, "")
    .trim();
  if (!text) return null;
  let match = new RegExp(String.raw`^(?:pp?\.?|pages?)\s*${RANGE}$`, "i").exec(text);
  if (match) return { kind: "page", ...range(match[1], match[2]) };
  match = new RegExp(
    String.raw`^slides?\s*${RANGE}(?:\s*[(,]?\s*(?:speaker\s+)?notes\)?)?$`,
    "i",
  ).exec(text);
  if (match) return { kind: "slide", ...range(match[1], match[2]) };
  match = new RegExp(String.raw`^lines?\s*${RANGE}$`, "i").exec(text);
  if (match) return { kind: "lines", ...range(match[1], match[2]) };
  match = new RegExp(String.raw`^(?:(.+?)\s*[,:;!]\s*)?rows?\s*${RANGE}$`, "i").exec(text);
  if (match) return { kind: "rows", sheet: match[1]?.trim() || null, ...range(match[2], match[3]) };
  match = /^(?:'?(.+?)'?!)?\$?[A-Z]{1,3}\$?(\d{1,7})(?::\$?[A-Z]{1,3}\$?(\d{1,7}))?$/i.exec(text);
  if (match) return { kind: "rows", sheet: match[1]?.trim() || null, ...range(match[2], match[3]) };
  match = /^(?:§+|sections?\b|heading\b)\s*(.+)$/i.exec(text);
  const heading = match?.[1]?.replace(A_COMMENT, "").trim();
  if (heading) return { kind: "section", heading };
  return null;
}

const simplify = (text: string) =>
  text.toLowerCase().replace(/[§›>]/g, " ").replace(/\s+/g, " ").trim();

/** Whether a section Unit is the one a heading names: its heading, its path, or its number ("2.1"). */
function sectionMatches(unit: UnitText, heading: string): boolean {
  const wanted = simplify(heading).replace(/\s*\(\d+\)$/, "");
  const path = unit.label?.path ?? [];
  if (unit.label?.notes) return /^(notes|footnotes|endnotes)$/.test(wanted);
  if (path.length === 0) return /^(start|beginning|introduction)$/.test(wanted);
  const own = simplify(path.at(-1) as string);
  if (own === wanted || simplify(path.join(" ")) === wanted) return true;
  // "2.1", for a heading numbered "2.1 Sensitivity".
  return /^[\d.]+$/.test(wanted) && own.startsWith(`${wanted} `);
}

/**
 * The Units (their numbers) of `units`, a Passage's, that a requested
 * Location names; null if none does. Pages and slides name their numbers
 * outright, so they can lie outside the Passage (and the check says so).
 */
export function resolveRequested(
  requested: RequestedLocation,
  units: readonly UnitText[],
): { from: number; to: number } | null {
  const kind = units[0]?.kind ?? "page";
  if (requested.kind === "page" || requested.kind === "slide") {
    // A deck's "page 4" is its slide 4; pages mean nothing in the other kinds.
    return kind === "page" || kind === "slide" ? { from: requested.from, to: requested.to } : null;
  }
  const named = (unit: UnitText): boolean => {
    if (requested.kind === "section") {
      return unit.kind === "section" && sectionMatches(unit, requested.heading);
    }
    if (unit.kind !== requested.kind) return false;
    const label = unit.label ?? null;
    if (requested.kind === "rows" && requested.sheet && label?.sheet) {
      if (simplify(label.sheet) !== simplify(requested.sheet)) return false;
    }
    return requested.from <= (label?.to ?? 0) && requested.to >= (label?.from ?? 0);
  };
  const numbers: number[] = [];
  for (const unit of units) if (unit.page !== null && named(unit)) numbers.push(unit.page);
  if (numbers.length === 0) return null;
  return { from: Math.min(...numbers), to: Math.max(...numbers) };
}
