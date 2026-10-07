/**
 * Citations (the v1 design's "Wedge mechanics"). An Answer cites a Passage by
 * writing a marker such as "[^1]" right after a claim, and giving, through the
 * `cite` Tool or structured output, a record for it: the Passage, the page or
 * two consecutive pages its quote is on, and the quote, copied word for word.
 * The core turns each marker with a record into an inline Citation node; a
 * marker without a record is removed, and a record without a marker dropped.
 *
 * When the Answer finishes, each Citation is checked once, and the result
 * stored with it: the cited pages must lie within the Passage's pages and be
 * at most two consecutive ones (the page-range rule), and the quote must be in
 * their text, matched exactly after both are normalised the same way (see
 * `findQuote`). The pages are read as stored when the Document was processed,
 * without running headers, footers and page numbers, so a quote may run across
 * a page break. "Found" means the quote is there, never that it supports the
 * sentence.
 */
import { findQuote } from "../../shared/quoteMatch";
import {
  CITATION_NODE,
  type Citation,
  type CitationAttributes,
  type CitationCheck,
  type CitationCheckReason,
  MAX_CITED_PAGES,
} from "../api";
import type { CitationSource } from "../documents";
import type { PageText } from "../documents/passages";
import type { WindowedPassage } from "../documents/search";
import type { NodeJSON } from "./blocks";
import type { AnswerTools, CitationRecordInput, SearchResultForModel } from "./engine";

/** How many times one Answer may search; later searches are refused, so the model answers. */
const MAX_SEARCHES_PER_ANSWER = 5;

/** A Citation marker as the model writes it in the text: "[^1]". */
const CITATION_MARKER = /\[\^(\d{1,4})\]/g;

/** Where a new page starts inside a Passage given to the model, e.g. "[p. 4]". */
const PAGE_MARK = /\[p\.\s*\d+\]/gi;

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
 * The quote as the model meant it: without page marks, without the ellipses
 * and quotation marks models put around a quote. Its words are left alone.
 */
function cleanQuote(raw: string): string {
  let quote = raw.replace(PAGE_MARK, " ").trim();
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

/** The pages a Citation names: both null for a Document without pages. */
export interface PageRange {
  pageFrom: number | null;
  pageTo: number | null;
}

const asPage = (value: unknown): number | null =>
  typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : null;

/**
 * The pages a record cites, in order. A record that names no page cites the
 * Passage's own pages; a Document without pages has none.
 */
function citedPages(
  record: Pick<CitationRecordInput, "pageFrom" | "pageTo">,
  passage: PageRange,
): PageRange {
  if (passage.pageFrom === null || passage.pageTo === null) return { pageFrom: null, pageTo: null };
  const from = asPage(record.pageFrom);
  const to = asPage(record.pageTo);
  if (from === null && to === null) return { pageFrom: passage.pageFrom, pageTo: passage.pageTo };
  const first = (from ?? to) as number;
  const last = (to ?? from) as number;
  return { pageFrom: Math.min(first, last), pageTo: Math.max(first, last) };
}

/**
 * The page-range rule: the cited pages must lie within the cited Passage's
 * pages, and be one page or two consecutive ones. Returns why they break it,
 * or null.
 */
function pageRangeProblem(range: PageRange, passage: PageRange): CitationCheckReason | null {
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
 * The Citation check. `pages` is the stored text of the cited pages (for a
 * Document without pages, its one text), as `AnswerDocuments.pageTexts` gives it.
 */
export function checkCitation(input: {
  quote: string;
  range: PageRange;
  passage: PageRange;
  documentDeleted: boolean;
  pages: readonly PageText[];
}): CheckResult {
  if (input.documentDeleted) return { check: "cant-check", checkReason: "document-removed" };
  const problem = pageRangeProblem(input.range, input.passage);
  if (problem) return { check: "not-found", checkReason: problem };
  const text = input.pages.map((page) => page.text).join("\n");
  if (text.trim() === "") return { check: "cant-check", checkReason: "no-text" };
  return findQuote(text, input.quote)
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
  /** The stored text of a live Document's pages (see `pageTexts` in ../documents). */
  pageTexts(documentId: string, from: number | null, to: number | null): PageText[];
}

const pagesLabel = ({ pageFrom, pageTo }: PageRange) =>
  pageFrom === null
    ? ""
    : pageTo !== null && pageTo !== pageFrom
      ? `${pageFrom}-${pageTo}`
      : `${pageFrom}`;

/**
 * The Passage's text with "[p. N]" where each of its later pages starts, so
 * the model can tell which page a quote is on. Found by matching the start of
 * each page's stored text after a paragraph break, which is how pages were
 * laid out when the Passages were built.
 */
function withPageMarks(text: string, range: PageRange, pages: readonly PageText[]): string {
  if (range.pageFrom === null || range.pageTo === null || range.pageFrom === range.pageTo) {
    return text;
  }
  let marked = "";
  let from = 0;
  for (const page of pages) {
    if (page.page === null || page.page <= range.pageFrom || page.page > range.pageTo) continue;
    const start = page.text.trim();
    if (!start) continue;
    for (let at = text.indexOf("\n\n", from); at >= 0; at = text.indexOf("\n\n", at + 1)) {
      const rest = text.slice(at + 2);
      const length = Math.min(rest.length, start.length, 200);
      if (length > 0 && rest.slice(0, length) === start.slice(0, length)) {
        marked += `${text.slice(from, at)}\n\n[p. ${page.page}] `;
        from = at + 2;
        break;
      }
    }
  }
  return marked + text.slice(from);
}

const attribute = (value: string) => value.replace(/"/g, "'").replace(/\s+/g, " ");

/** Passages as the model sees them, each with its short id, Document and pages. */
function formatPassages(
  passages: readonly { id: string; documentName: string; range: PageRange; text: string }[],
): string {
  return passages
    .map(({ id, documentName, range, text }) => {
      const pages = pagesLabel(range);
      const attributes = `id="${id}" document="${attribute(documentName)}"${pages ? ` pages="${pages}"` : ""}`;
      return `<passage ${attributes}>\n${text}\n</passage>`;
    })
    .join("\n\n");
}

/** A record the core accepted, with what it points to. */
interface Accepted {
  marker: number;
  handle: string;
  source: CitationSource;
  range: PageRange;
  quote: string;
  result: CheckResult | null;
}

const toCitation = (record: Accepted, result: CheckResult | null): Citation => ({
  passageId: record.source.passageId,
  documentId: record.source.documentId,
  documentName: record.source.documentName,
  contentHash: record.source.contentHash,
  pageFrom: record.range.pageFrom,
  pageTo: record.range.pageTo,
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

  /** The check, with the Document as it is now: it may have been deleted since the record. */
  const check = (record: Accepted) => {
    const now = documents.citationSource(record.source.passageId);
    return checkCitation({
      quote: record.quote,
      range: record.range,
      passage: record.source,
      documentDeleted: !now || now.documentDeleted,
      pages: documents.pageTexts(
        record.source.documentId,
        record.range.pageFrom,
        record.range.pageTo,
      ),
    });
  };

  /** Takes records; returns what to tell the model about them. */
  function cite(inputs: readonly CitationRecordInput[]): string {
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
      const found = typeof input.passage === "string" ? resolve(input.passage) : null;
      const source = found && documents.citationSource(found.passageId);
      if (!found || !source) {
        invalidRecords++;
        problems.push(
          `${label}: there is no Passage "${String(input.passage)}" in your search results; use a Passage's id, such as P1.`,
        );
        continue;
      }
      const range = citedPages(input, source);
      const record: Accepted = {
        marker,
        handle: found.handle,
        source,
        range,
        quote: cleanQuote(typeof input.quote === "string" ? input.quote : ""),
        result: null,
      };
      records.set(marker, record);
      recorded.push(marker);
      events.onRecord(marker, toCitation(record, null));

      // Tell the model now what the check will find, so it can fix the record.
      const problem = pageRangeProblem(range, source);
      if (problem) {
        problems.push(
          `${label}: cite one page, or two consecutive pages, within ${found.handle}'s pages (${pagesLabel(source)}).`,
        );
      } else if (!record.quote) {
        problems.push(`${label}: the quote is empty; copy a short quote from ${found.handle}.`);
      } else if (check(record).check === "not-found") {
        const where = range.pageFrom === null ? "" : ` on p. ${pagesLabel(range)}`;
        problems.push(
          `${label}: the quote isn't word for word${where} in ${found.handle}; copy it exactly, and check its page.`,
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
      const formatted = found.map((passage) => {
        const range = { pageFrom: passage.pageFrom, pageTo: passage.pageTo };
        const pages =
          passage.pageFrom !== null && passage.pageTo !== null && passage.pageFrom < passage.pageTo
            ? documents.pageTexts(passage.documentId, passage.pageFrom + 1, passage.pageTo)
            : [];
        return {
          id: handleFor(passage.passageId),
          documentName: passage.documentName,
          range,
          text: withPageMarks(passage.text, range, pages),
        };
      });
      return { text: formatPassages(formatted), passageCount: found.length };
    },

    cite,
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
