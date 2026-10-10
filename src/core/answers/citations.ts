/**
 * Citations (the v1 design's "Wedge mechanics"). An Answer cites a Passage by
 * writing a marker such as "[^1]" right after a claim, and giving, through the
 * `cite` Tool or structured output, a record for it: the Passage, the
 * Location its quote is at (ADR-0011: one or two pages, slides, sections, or
 * blocks of rows or lines, the Units of src/shared/units.ts), and the quote,
 * copied word for word. The core turns each marker with a record into an
 * inline Citation node; a marker without a record is removed, and a record
 * without a marker dropped.
 *
 * When the Answer finishes, each Citation is checked once, and the result
 * stored with it: the cited Units must lie within the Passage's Units and be
 * at most two consecutive ones of one sheet or Document (the Location rule),
 * and the quote must be in their text, matched exactly after both are
 * normalised the same way (see `findQuote`: letter case, spacing, hyphenation
 * at line ends, quote marks and "[^36]" for "[36]" don't count, and a quote
 * with an ellipsis is found when each part, of at least 3 words, is there in
 * order; in spreadsheets, number formatting doesn't count either). The Units
 * are read as stored when the Document was processed, a PDF's without
 * running headers, footers and page numbers, so a quote may run across a page
 * break. "Found" means the quote is there, never that it supports the sentence.
 *
 * Each Citation also stores its Location for its label: for blocks of rows or
 * lines, narrowed to those the quote covers ("Revenue, rows 12–14"); for
 * sections, the heading it sits under.
 */
import {
  englishLocation,
  locationOf,
  parseRequestedLocation,
  quoteInUnits,
  type RowsOrLines,
  resolveRequested,
  unitLocation,
} from "../../shared/locations";
import { findQuote } from "../../shared/quoteMatch";
import { foldRadicals } from "../../shared/text";
import { sameRun } from "../../shared/units";
import {
  CITATION_NODE,
  type Citation,
  type CitationAttributes,
  type CitationCheck,
  type CitationCheckReason,
  type CitationRecheck,
  MAX_CITED_PAGES,
} from "../api";
import type { CitationSource } from "../documents";
import type { PageText } from "../documents/passages";
import type { WindowedPassage } from "../documents/search";
import type { NodeJSON } from "./blocks";
import type { AnswerTools, CitationRecordInput, CiteOptions, SearchResultForModel } from "./engine";

/** How many times one Answer may search; later searches are refused, so the model answers. */
const MAX_SEARCHES_PER_ANSWER = 5;

/** A Citation marker as the model writes it in the text: "[^1]". */
const CITATION_MARKER = /\[\^(\d{1,4})\]/g;

/**
 * Where a new Unit starts inside a Passage given to the model ("[p. 4]",
 * "[slide 4]", "[§ 2.1 Sensitivity]", "[Revenue, rows 2–41]", "[lines 51–100]"),
 * and where a slide's speaker notes start ("[speaker notes]").
 */
const LOCATION_MARK =
  /\[(?:p\.\s*\d+(?:[–-]\d+)?|slides?\s+\d+(?:[–-]\d+)?|(?:[^\]\n]{1,100},\s*)?rows?\s+\d+(?:[–-]\d+)?|lines?\s+\d+(?:[–-]\d+)?|§[^\]\n]{0,200}|speaker notes)\]/gi;

/** Marks where a slide's speaker notes start, inside a Passage given to the model. */
const NOTES_MARK = "[speaker notes]";

const QUOTE_PAIRS: readonly (readonly [string, string])[] = [
  ['"', '"'],
  ["'", "'"],
  ["“", "”"],
  ["‘", "’"],
  ["「", "」"],
  ["『", "』"],
  ["«", "»"],
];

/**
 * The quote as the model meant it: without location marks, without the
 * ellipses and quotation marks models put around a quote. Its words are left alone.
 */
function cleanQuote(raw: string): string {
  let quote = raw.replace(LOCATION_MARK, " ").trim();
  for (;;) {
    const before = quote;
    quote = quote
      .replace(/^(?:\.{3}|…)\s*/u, "")
      .replace(/\s*(?:\.{3}|…)$/u, "")
      .trim();
    for (const [open, close] of QUOTE_PAIRS) {
      if (quote.length > 1 && quote.startsWith(open) && quote.endsWith(close)) {
        quote = quote.slice(open.length, -close.length).trim();
      }
    }
    if (quote === before) return quote;
  }
}

/** The Units a Citation names, by number: both null for a whole TXT or Markdown file stored before Units. */
export interface PageRange {
  pageFrom: number | null;
  pageTo: number | null;
}

const asPage = (value: unknown): number | null =>
  typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : null;

/** What a record cites: its Units, the rows or lines it named, and whether its Location named nothing there. */
interface Cited {
  range: PageRange;
  /** Rows or lines the record named more narrowly than its Units: the label's fallback. */
  requested: RowsOrLines | null;
  /** The record named a Location the Passage doesn't have. */
  unresolved: boolean;
}

/**
 * The Units a record cites, in order. A record names a Location ("slide 4",
 * "Revenue, rows 12–14", "§ 2.1 Sensitivity"), or for a PDF or deck, its
 * pages. One that names none cites the Passage's own Units; when they are
 * more than two, the quote's own Unit (or two) among them, if it is there:
 * a PDF's too, as a small model often names no page, and a Passage of a
 * page with little text runs over three or four (#67: 梯度下降法's covers
 * pp. 1–4). `units` are the Passage's.
 */
function citedUnits(
  record: Pick<CitationRecordInput, "pageFrom" | "pageTo" | "location" | "quote">,
  passage: PageRange,
  units: readonly PageText[],
): Cited {
  if (passage.pageFrom === null || passage.pageTo === null) {
    return { range: { pageFrom: null, pageTo: null }, requested: null, unresolved: false };
  }
  const own = { pageFrom: passage.pageFrom, pageTo: passage.pageTo };
  const kind = units[0]?.kind ?? "page";
  const requested =
    typeof record.location === "string" ? parseRequestedLocation(record.location) : null;
  if (requested) {
    const resolved = resolveRequested(requested, units);
    if (!resolved) return { range: own, requested: null, unresolved: true };
    return {
      range: { pageFrom: resolved.from, pageTo: resolved.to },
      requested:
        requested.kind === "rows" || requested.kind === "lines"
          ? { from: requested.from, to: requested.to }
          : null,
      unresolved: false,
    };
  }
  const from = asPage(record.pageFrom);
  const to = asPage(record.pageTo);
  if ((kind === "page" || kind === "slide") && (from !== null || to !== null)) {
    const first = (from ?? to) as number;
    const last = (to ?? from) as number;
    return {
      range: { pageFrom: Math.min(first, last), pageTo: Math.max(first, last) },
      requested: null,
      unresolved: false,
    };
  }
  if (own.pageTo - own.pageFrom + 1 > MAX_CITED_PAGES) {
    const quote = cleanQuote(typeof record.quote === "string" ? record.quote : "");
    const found = locateQuote(units, quote);
    if (found) return { range: found, requested: null, unresolved: false };
  }
  return { range: own, requested: null, unresolved: false };
}

/** The first Unit, or two consecutive Units of one run, among `units` whose text holds the quote. */
function locateQuote(units: readonly PageText[], quote: string): PageRange | null {
  if (!quote) return null;
  for (const unit of units) {
    if (unit.page !== null && quoteInUnits([unit], quote)) {
      return { pageFrom: unit.page, pageTo: unit.page };
    }
  }
  for (let index = 0; index + 1 < units.length; index++) {
    const first = units[index] as PageText;
    const second = units[index + 1] as PageText;
    if (first.page === null || second.page !== first.page + 1 || !sameRun(first, second)) continue;
    if (quoteInUnits([first, second], quote)) return { pageFrom: first.page, pageTo: second.page };
  }
  return null;
}

/**
 * The Location rule: the cited Units must lie within the cited Passage's
 * Units, and be one Unit or two consecutive ones of one run. Returns why they
 * break it, or null.
 */
function pageRangeProblem(
  range: PageRange,
  passage: PageRange,
  unresolved = false,
): CitationCheckReason | null {
  if (unresolved) return "pages-outside-passage";
  if (range.pageFrom === null || range.pageTo === null) return null;
  if (passage.pageFrom === null || passage.pageTo === null) return null;
  if (range.pageFrom < passage.pageFrom || range.pageTo > passage.pageTo) {
    return "pages-outside-passage";
  }
  if (range.pageTo - range.pageFrom + 1 > MAX_CITED_PAGES) return "too-many-pages";
  return null;
}

export interface CheckResult {
  check: Exclude<CitationCheck, "checking">;
  checkReason: CitationCheckReason | null;
}

/**
 * The Citation check. `pages` is the stored text of the cited Units (for a
 * whole TXT or Markdown file stored before Units, its one text), as
 * `AnswerDocuments.pageTexts` gives it.
 */
export function checkCitation(input: {
  quote: string;
  range: PageRange;
  passage: PageRange;
  documentDeleted: boolean;
  pages: readonly PageText[];
  /** The record named a Location the Passage doesn't have. */
  unresolved?: boolean;
}): CheckResult {
  if (input.documentDeleted) return { check: "cant-check", checkReason: "document-removed" };
  const problem = pageRangeProblem(input.range, input.passage, input.unresolved);
  if (problem) return { check: "not-found", checkReason: problem };
  const [first, second] = input.pages;
  if (first && second && !sameRun(first, second)) {
    return { check: "not-found", checkReason: "too-many-pages" };
  }
  if (input.pages.every((page) => page.text.trim() === "")) {
    return { check: "cant-check", checkReason: "no-text" };
  }
  return quoteInUnits(input.pages, input.quote)
    ? { check: "found", checkReason: null }
    : { check: "not-found", checkReason: "quote-not-on-pages" };
}

/** What the core needs from Documents for an Answer: search, and what Citations point to. */
export interface AnswerDocuments {
  /** How many live Documents have Passages to search. */
  searchableCount(): number;
  /** The document-search Tool. */
  search(query: string, signal?: AbortSignal): Promise<WindowedPassage[]>;
  /** A Passage and its Document, deleted or not; null if unknown. */
  citationSource(passageId: string): CitationSource | null;
  /** The stored text of one version of a live Document's Units (see `pageTexts` in ../documents). */
  pageTexts(
    documentId: string,
    contentHash: string,
    from: number | null,
    to: number | null,
  ): PageText[];
}

/** What re-checking a Citation against the current version needs from Documents. */
export interface RecheckDocuments {
  /** The current version of a live Document; null if it was deleted. */
  currentVersion(documentId: string): string | null;
  pageTexts: AnswerDocuments["pageTexts"];
  /** The live Passages of the current version covering these Units, in reading order. */
  passagesCovering(
    documentId: string,
    from: number | null,
    to: number | null,
  ): { id: string; text: string }[];
}

/**
 * Checks a Citation again against its Document's current version (see
 * `CoreApi.recheckCitation`): in the cited Units first, then in each Unit,
 * and each two consecutive Units of one run, of the current version, in
 * order. The Passage given is one of the current version that holds the
 * Units, preferably one whose text holds the quote too.
 */
export function recheckCitation(
  documents: RecheckDocuments,
  input: { documentId: string; quote: string; pageFrom: number | null; pageTo: number | null },
): CitationRecheck {
  const cited = { pageFrom: input.pageFrom, pageTo: input.pageTo };
  const contentHash = documents.currentVersion(input.documentId);
  if (contentHash === null) {
    return {
      check: "cant-check",
      checkReason: "document-removed",
      contentHash: null,
      passageId: null,
      ...cited,
      location: null,
    };
  }
  const quote = cleanQuote(input.quote);
  const units = documents.pageTexts(input.documentId, contentHash, null, null);
  const passageFor = (range: PageRange): string | null => {
    const covering = documents.passagesCovering(input.documentId, range.pageFrom, range.pageTo);
    return (
      covering.find((passage) => quote !== "" && findQuote(passage.text, quote))?.id ??
      covering[0]?.id ??
      null
    );
  };
  const inRange = (range: PageRange) =>
    units.filter(
      (unit) =>
        range.pageFrom === null ||
        range.pageTo === null ||
        (unit.page !== null && unit.page >= range.pageFrom && unit.page <= range.pageTo),
    );
  const result = (check: CheckResult, range: PageRange): CitationRecheck => {
    const covered = inRange(range);
    return {
      ...check,
      contentHash,
      passageId: passageFor(range),
      ...range,
      location: locationOf(covered, check.check === "found" ? quoteInUnits(covered, quote) : null),
    };
  };
  if (units.every((unit) => unit.text.trim() === "")) {
    return result({ check: "cant-check", checkReason: "no-text" }, cited);
  }
  // A whole TXT or Markdown file stored before Units is checked as one text.
  if (units.length === 1 && units[0]?.kind === "text") {
    const whole = { pageFrom: null, pageTo: null };
    return quote !== "" && quoteInUnits(units, quote)
      ? result({ check: "found", checkReason: null }, whole)
      : result({ check: "not-found", checkReason: "quote-not-on-pages" }, whole);
  }
  if (cited.pageFrom !== null && cited.pageTo !== null) {
    const range = inRange(cited);
    const [first, second] = range;
    if (
      range.length > 0 &&
      range.length <= MAX_CITED_PAGES &&
      (!first || !second || sameRun(first, second)) &&
      quoteInUnits(range, quote)
    ) {
      return result({ check: "found", checkReason: null }, cited);
    }
  }
  const located = locateQuote(units, quote);
  if (located) return result({ check: "found", checkReason: null }, located);
  return result({ check: "not-found", checkReason: "quote-not-on-pages" }, cited);
}

const pagesLabel = ({ pageFrom, pageTo }: PageRange) =>
  pageFrom === null
    ? ""
    : pageTo !== null && pageTo !== pageFrom
      ? `${pageFrom}-${pageTo}`
      : `${pageFrom}`;

/** The English label of a Unit, as its mark in a Passage shows it. */
function unitMark(unit: PageText): string | null {
  const location = unitLocation(unit);
  return location ? `[${englishLocation(location)}]` : null;
}

/** The start of a stretch of text, long enough to find it again where it was laid out. */
const headOf = (text: string) => text.trim().slice(0, 200);

/**
 * The Passage's text with marks where each of its later Units starts ("[p. 4]",
 * "[slide 4]", "[§ 2.1 Sensitivity]", "[Revenue, rows 2–41]") and where a
 * slide's speaker notes start ("[speaker notes]"), so the model can tell
 * where a quote is. Found by matching the start of each Unit's stored text
 * after a paragraph break, which is how Units were laid out when the Passages
 * were built (and how a slide's notes follow its text). `units` are the
 * Passage's, in order.
 */
function withUnitMarks(text: string, units: readonly PageText[]): string {
  const marks: { head: string; mark: string; unit: boolean }[] = [];
  units.forEach((unit, index) => {
    const mark = unitMark(unit);
    if (index > 0 && mark && unit.text.trim()) {
      marks.push({ head: headOf(unit.text), mark, unit: true });
    }
    const notes =
      unit.kind === "slide" ? unit.anchors?.find((each) => each.target === "notes") : null;
    if (notes) {
      marks.push({
        head: headOf(unit.text.slice(notes.start, notes.end)),
        mark: NOTES_MARK,
        unit: false,
      });
    }
  });
  let marked = "";
  let from = 0;
  for (const { head, mark } of marks) {
    if (!head) continue;
    if (from === 0 && marked === "" && text.startsWith(head) && mark === NOTES_MARK) {
      marked = `${mark} `;
      continue;
    }
    for (let at = text.indexOf("\n\n", from); at >= 0; at = text.indexOf("\n\n", at + 1)) {
      const rest = text.slice(at + 2);
      const length = Math.min(rest.length, head.length);
      if (length > 0 && rest.slice(0, length) === head.slice(0, length)) {
        marked += `${text.slice(from, at)}\n\n${mark} `;
        from = at + 2;
        break;
      }
    }
  }
  return marked + text.slice(from);
}

const attribute = (value: string) => value.replace(/"/g, "'").replace(/\s+/g, " ");

/** A Passage as the model sees it: its short id, Document, where it is, and its text. */
interface ShownPassage {
  id: string;
  documentName: string;
  range: PageRange;
  /** For every kind but PDF: the label of the Units it covers, e.g. "slides 3–4". */
  location: string | null;
  text: string;
}

/** Passages as the model sees them, each with its short id, Document and pages or Location. */
function formatPassages(passages: readonly ShownPassage[]): string {
  return passages
    .map(({ id, documentName, range, location, text }) => {
      const pages = pagesLabel(range);
      const where = location
        ? ` location="${attribute(location)}"`
        : pages
          ? ` pages="${pages}"`
          : "";
      return `<passage id="${id}" document="${attribute(documentName)}"${where}>\n${text}\n</passage>`;
    })
    .join("\n\n");
}

/** A record the core accepted, with what it points to. */
interface Accepted {
  marker: number;
  handle: string;
  source: CitationSource;
  range: PageRange;
  requested: RowsOrLines | null;
  unresolved: boolean;
  quote: string;
  /** Where it points, for its label, worked out once when the record is taken. */
  location: Citation["location"];
  result: CheckResult | null;
}

const toCitation = (record: Accepted, result: CheckResult | null): Citation => ({
  passageId: record.source.passageId,
  documentId: record.source.documentId,
  documentName: record.source.documentName,
  contentHash: record.source.contentHash,
  pageFrom: record.range.pageFrom,
  pageTo: record.range.pageTo,
  location: record.location,
  quote: record.quote,
  check: result?.check ?? "checking",
  checkReason: result?.checkReason ?? null,
});

/** A Citation node's attributes: every one set (null ones are left out, as y-tiptap stores them). */
function citationNode(citation: Citation | null): NodeJSON {
  const attrs: CitationAttributes = citation ?? {
    passageId: null,
    documentId: null,
    documentName: null,
    contentHash: null,
    pageFrom: null,
    pageTo: null,
    location: null,
    quote: null,
    check: "checking",
    checkReason: null,
  };
  return {
    type: CITATION_NODE,
    attrs: Object.fromEntries(Object.entries(attrs).filter(([, value]) => value !== null)),
  };
}

export interface CitationSessionEvents {
  /** The model gave a valid record for a marker (again: it replaces the earlier one). */
  onRecord(marker: number, citation: Citation): void;
}

/**
 * One Answer's Citations: the Passages it was given (each with a short id such
 * as "P3", which is what the model cites), the records it gave, and the Tools
 * the Answer engine offers the model.
 */
export function createCitationSession(documents: AnswerDocuments, events: CitationSessionEvents) {
  const handles = new Map<string, string>();
  const handleOf = new Map<string, string>();
  const records = new Map<number, Accepted>();
  /** Records naming no Passage the model was given. */
  let invalidRecords = 0;
  let searches = 0;
  /** Markers in the final text, in order of first appearance, and those that had no record. */
  const finalMarkers: number[] = [];
  let removedMarkers = 0;

  const handleFor = (passageId: string) => {
    let handle = handleOf.get(passageId);
    if (!handle) {
      handle = `P${handleOf.size + 1}`;
      handleOf.set(passageId, handle);
      handles.set(handle.toLowerCase(), passageId);
    }
    return handle;
  };

  /** The Passage a record names: by its short id or its full id, if the model was given it. */
  const resolve = (name: string): { handle: string; passageId: string } | null => {
    const key = name
      .trim()
      .replace(/^\[|\]$/g, "")
      .toLowerCase();
    const byHandle = handles.get(key);
    if (byHandle) return { handle: handleOf.get(byHandle) as string, passageId: byHandle };
    for (const [passageId, handle] of handleOf) {
      if (passageId.toLowerCase() === key) return { handle, passageId };
    }
    return null;
  };

  /** The stored text of the Units a record cites, of the version its Passage was built from. */
  const citedText = (record: Pick<Accepted, "source" | "range">) =>
    documents.pageTexts(
      record.source.documentId,
      record.source.contentHash,
      record.range.pageFrom,
      record.range.pageTo,
    );

  /**
   * The check, with the Document as it is now: it may have been deleted since
   * the record. The text read is that of the version the Passage was built
   * from, which the Citation records: the Document may have a newer one by now.
   */
  const check = (record: Accepted) => {
    const now = documents.citationSource(record.source.passageId);
    return checkCitation({
      quote: record.quote,
      range: record.range,
      passage: record.source,
      documentDeleted: !now || now.documentDeleted,
      pages: citedText(record),
      unresolved: record.unresolved,
    });
  };

  /** Where a record points, for its label: narrowed to its quote when the quote is there. */
  const locationFor = (record: Omit<Accepted, "location">): Citation["location"] => {
    if (record.range.pageFrom === null) return null;
    const units = citedText(record);
    if (units.length === 0) {
      return record.source.documentKind === "pdf" && record.range.pageTo !== null
        ? { kind: "page", from: record.range.pageFrom, to: record.range.pageTo }
        : null;
    }
    const usable = record.unresolved || units.length > MAX_CITED_PAGES ? null : units;
    return locationOf(units, usable ? quoteInUnits(usable, record.quote) : null, record.requested);
  };

  /** A Passage named by its number alone, "1" or "[1]", as structured output may: "P1". */
  const numbered = (name: string) => name.replace(/^\s*\[?\s*(\d{1,4})\s*\]?\s*$/, "P$1");

  /** Takes records; returns what to tell the model about them. */
  function cite(inputs: readonly CitationRecordInput[], options: CiteOptions = {}): string {
    const recorded: number[] = [];
    const problems: string[] = [];
    for (const input of inputs) {
      const marker = input.marker;
      if (!Number.isInteger(marker) || marker < 1) {
        invalidRecords++;
        problems.push(`A record needs a marker number (1, 2, …) that is in the Answer as [^n].`);
        continue;
      }
      const label = `[^${marker}]`;
      const named =
        typeof input.passage === "string" && options.structured
          ? numbered(input.passage)
          : input.passage;
      const found = typeof named === "string" ? resolve(named) : null;
      const source = found && documents.citationSource(found.passageId);
      if (!found || !source) {
        invalidRecords++;
        problems.push(
          `${label}: there is no Passage "${String(input.passage)}" in your search results; use a Passage's id, such as P1.`,
        );
        continue;
      }
      const quote = cleanQuote(typeof input.quote === "string" ? input.quote : "");
      const passageUnits = documents.pageTexts(
        source.documentId,
        source.contentHash,
        source.pageFrom,
        source.pageTo,
      );
      const { range, requested, unresolved } = citedUnits(
        { ...input, quote },
        source,
        passageUnits,
      );
      const taken = {
        marker,
        handle: found.handle,
        source,
        range,
        requested,
        unresolved,
        quote,
        result: null,
      };
      let record: Accepted = { ...taken, location: locationFor(taken) };
      // Structured output can't be told to fix a record (see `CiteOptions`): a quote that isn't where
      // it names is cited where it is in its Passage, never outside it, and the check still reads it there.
      if (options.structured && quote && check(record).check === "not-found") {
        const placedAt = locateQuote(passageUnits, quote);
        if (placedAt) {
          const placed = { ...taken, range: placedAt, requested: null, unresolved: false };
          record = { ...placed, location: locationFor(placed) };
        }
      }
      records.set(marker, record);
      recorded.push(marker);
      events.onRecord(marker, toCitation(record, null));

      // Tell the model now what the check will find, so it can fix the record.
      const problem = pageRangeProblem(record.range, source, record.unresolved);
      const where = passageUnits.length > 0 ? passageLocation(passageUnits) : null;
      if (problem) {
        problems.push(
          where
            ? `${label}: give a location inside ${found.handle} (${where}), naming one place, or two in a row.`
            : `${label}: cite one page, or two consecutive pages, within ${found.handle}'s pages (${pagesLabel(source)}).`,
        );
      } else if (!record.quote) {
        problems.push(`${label}: the quote is empty; copy a short quote from ${found.handle}.`);
      } else if (check(record).check === "not-found") {
        problems.push(
          where
            ? `${label}: the quote isn't word for word at ${cutLocation(record) ?? where} in ${found.handle}; copy it exactly, and check where it is.`
            : `${label}: the quote isn't word for word${record.range.pageFrom === null ? "" : ` on p. ${pagesLabel(record.range)}`} in ${found.handle}; copy it exactly, and check its page.`,
        );
      }
    }
    const done = recorded.length
      ? `Recorded ${recorded.map((marker) => `[^${marker}]`).join(", ")}.`
      : "Nothing was recorded.";
    const fix = problems.length
      ? ` Fix these with another cite call:\n- ${problems.join("\n- ")}`
      : " Write the Answer now if you haven't yet; don't repeat it.";
    return `${done}${fix}`;
  }

  /** The English label of a record's cited Units, as a whole: "slide 4", "§ 2.1 Sensitivity". */
  const cutLocation = (record: Accepted): string | null => {
    if (record.range.pageFrom === null) return null;
    const location = locationOf(citedText(record), null, record.requested);
    return location ? englishLocation(location) : null;
  };

  /** The English label of a Passage's Units, as its `location` attribute shows them. */
  const passageLocation = (units: readonly PageText[]): string | null => {
    if ((units[0]?.kind ?? "page") === "page") return null;
    const location = locationOf(units, null);
    return location ? englishLocation(location) : null;
  };

  const tools: AnswerTools = {
    get documentCount() {
      return documents.searchableCount();
    },

    async searchDocuments(query: string, signal?: AbortSignal): Promise<SearchResultForModel> {
      searches++;
      if (searches > MAX_SEARCHES_PER_ANSWER) {
        return {
          text: "You have searched enough for this Question: answer with the Passages you have.",
          passageCount: 0,
        };
      }
      const found = await documents.search(query, signal);
      if (found.length === 0) {
        return { text: "No Passages in the User's Documents match this search.", passageCount: 0 };
      }
      const formatted = found.map((passage): ShownPassage => {
        const range = { pageFrom: passage.pageFrom, pageTo: passage.pageTo };
        const units =
          passage.pageFrom !== null && passage.pageTo !== null
            ? documents.pageTexts(
                passage.documentId,
                passage.contentHash,
                passage.pageFrom,
                passage.pageTo,
              )
            : [];
        const marked =
          units.length > 1 || units[0]?.kind === "slide"
            ? withUnitMarks(passage.text, units)
            : passage.text;
        return {
          id: handleFor(passage.passageId),
          documentName: passage.documentName,
          range,
          location: passageLocation(units),
          // A browser-made PDF's "⼤" for "大": every other reader of the text folds it (ADR-0009).
          text: foldRadicals(marked),
        };
      });
      return {
        text: formatPassages(formatted),
        passageCount: found.length,
        ranks: found.map((passage, index) => passage.rank ?? index),
      };
    },

    cite,

    hasRecord: (marker) => records.has(marker),
  };

  return {
    tools,

    /** A marker's node while the Answer streams: its record's details, if given yet, "checking". */
    streamingNode(marker: number): NodeJSON {
      const record = records.get(marker);
      return citationNode(record ? toCitation(record, null) : null);
    },

    /** Checks every record, once, when the Answer finishes. */
    check(): void {
      for (const record of records.values()) record.result ??= check(record);
    },

    /**
     * A marker's node once the Answer has finished (after `check`), or null:
     * a marker with no record is removed. Call it for each marker in the final
     * text, in order: it counts them for `summary`.
     */
    finalNode(marker: number): NodeJSON | null {
      const record = records.get(marker);
      if (!finalMarkers.includes(marker)) {
        finalMarkers.push(marker);
        if (!record) removedMarkers++;
      }
      return record ? citationNode(toCitation(record, record.result)) : null;
    },

    /** The Answer's Citations and the counts the evaluation reads, after the final nodes. */
    summary(): { citations: Citation[]; droppedMarkers: number; droppedRecords: number } {
      const citations = finalMarkers.flatMap((marker) => {
        const record = records.get(marker);
        return record ? [toCitation(record, record.result)] : [];
      });
      const unused = [...records.keys()].filter((marker) => !finalMarkers.includes(marker)).length;
      return {
        citations,
        droppedMarkers: removedMarkers,
        droppedRecords: unused + invalidRecords,
      };
    },
  };
}

/**
 * Turns the markers in an Answer's text into Citation nodes: `node` gives a
 * marker's node, or null to remove the marker (with the space before it, so
 * "a claim [^3]." reads "a claim."). Text marked as code keeps its markers.
 */
export function withCitations(
  blocks: readonly NodeJSON[],
  node: (marker: number) => NodeJSON | null,
): NodeJSON[] {
  return blocks.map((block) => replaceMarkers(block, node));
}

function replaceMarkers(block: NodeJSON, node: (marker: number) => NodeJSON | null): NodeJSON {
  if (!block.content) return block;
  const content: NodeJSON[] = [];
  for (const child of block.content) {
    const isCode = child.marks?.some((mark) => mark.type === "code") ?? false;
    if (child.type !== "text" || !child.text || isCode) {
      content.push(replaceMarkers(child, node));
      continue;
    }
    let last = 0;
    for (const match of child.text.matchAll(CITATION_MARKER)) {
      let before = child.text.slice(last, match.index);
      const citation = node(Number(match[1]));
      // A removed marker takes the space before it, unless a word follows.
      if (!citation) {
        const after = child.text.slice(match.index + match[0].length);
        if (!/^\p{L}/u.test(after)) before = before.replace(/[ \t]+$/, "");
      }
      if (before) content.push({ ...child, text: before });
      if (citation) content.push(citation);
      last = match.index + match[0].length;
    }
    const rest = child.text.slice(last);
    if (rest) content.push(last === 0 ? child : { ...child, text: rest });
  }
  // Two text nodes with the same marks that a removed marker left side by side become one.
  const merged: NodeJSON[] = [];
  for (const child of content) {
    const previous = merged.at(-1);
    if (
      previous?.type === "text" &&
      child.type === "text" &&
      JSON.stringify(previous.marks ?? []) === JSON.stringify(child.marks ?? [])
    ) {
      merged[merged.length - 1] = {
        ...previous,
        text: `${previous.text ?? ""}${child.text ?? ""}`,
      };
    } else {
      merged.push(child);
    }
  }
  return { ...block, content: merged };
}

/**
 * Lines that are footnote definitions ("[^1]: …"): some models list their
 * sources this way at the end. The records are what counts, so these go.
 */
export function withoutFootnoteDefinitions(markdown: string): string {
  return markdown.replace(/^ {0,3}\[\^\d{1,4}\]:.*(?:\n|$)/gm, "");
}
