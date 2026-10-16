/**
 * A reference list in a Mind's Blocks (#711): the Blocks after a heading
 * such as "References", "Bibliography" or "参考文献", until the next heading
 * or the first paragraph that isn't a reference (an appendix's text), each
 * entry as written. A Block holding several numbered entries ("[1] …
 * [2] …"), as a PDF's list can come out, is split at their numbers. Read by
 * the core to check the list, and by the editor to offer the check.
 */
import { ARXIV_ID, normalizeDoi } from "./references";

const HEADING =
  /^(?:[\dIVX]+\.?\s*)?(?:references|reference list|bibliography|works cited|literature cited|literature|参考文献|参考资料|引用文献)\s*[:：]?$/i;
/** Where a numbered entry starts inside a Block: "[12] " or "12. " after the previous entry. */
const NUMBERED = /\s(?=\[\d{1,3}\]\s)|(?<=[.)])\s+(?=\d{1,3}\.\s+\p{Lu})/u;
/** How a numbered entry starts. */
const NUMBER_START = /^(?:\[\d{1,3}\]|\d{1,3}\.)\s/;
/** How an entry starts with its authors, family name first: "Maas, Andrew L, …". */
const AUTHOR_START = /^\p{Lu}[\p{L}'’-]*,\s/u;
/** The longest an entry's opening paragraphs run before its year, DOI or number. */
const LONGEST_OPENING = 400;

/** A DOI as a reference writes it, after "doi:" or "https://doi.org/" or alone. */
export const DOI_IN_TEXT = /(?:doi\.org\/|doi\s*[:：]?\s*)?(10\.\d{4,9}\/[^\s"<>]+)/i;
/** An arXiv number as a reference writes it: "arXiv:1706.03762", "CoRR, abs/1409.0473", or in an arxiv.org address. */
export const ARXIV_IN_TEXT = new RegExp(
  `(?:arxiv\\s*[:：]\\s*|arxiv\\.org/(?:abs|pdf)/|\\bcorr,?\\s*abs/)${ARXIV_ID.source}`,
  "i",
);
/**
 * A year a reference names: four digits from 1500 to 2099, not a page in a
 * range ("9(8):1735–1780") or part of a number ("abs/1512.00567").
 */
export const YEAR_IN_TEXT = /(?<![\d/:–—-])(1[5-9]\d\d|20\d\d)(?![\d–—-]|\.\d)/;
/** What every reference holds one of: a year, a DOI, an arXiv number, a web address or its number in the list. */
const REFERENCE_SIGN = new RegExp(
  [
    YEAR_IN_TEXT.source,
    DOI_IN_TEXT.source,
    ARXIV_IN_TEXT.source,
    "https?://",
    "^\\[\\d{1,3}\\]",
  ].join("|"),
  "i",
);
/** Fewer title words than this, and a reference says too little to look up. */
const MIN_WORDS = 4;
const WEB_ADDRESS = /https?:\/\/\S+/i;

/** A Block as the list is read from it. */
export interface ListBlock {
  id: string;
  heading: boolean;
  text: string;
}

export interface ListedReference {
  blockId: string;
  text: string;
}

/** Folds case and accents, so "Müller" is "muller". */
const folded = (text: string) => text.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();

/** Common English words, which say nothing about which work a title is. */
const STOP_WORDS = new Set(
  "a an and are as at be by for from in into is it its of on or the their to with".split(" "),
);

/** A text's words: letters and digits, folded; a CJK run as its characters, two by two. */
export function titleWords(text: string): string[] {
  const words: string[] = [];
  for (const [run] of folded(text).matchAll(/[\p{L}\p{N}]+/gu)) {
    for (const [part] of run.matchAll(/\p{Script=Han}+|[^\p{Script=Han}]+/gu)) {
      if (/^\p{Script=Han}/u.test(part)) {
        const chars = [...part];
        if (chars.length === 1) words.push(part);
        for (let at = 0; at + 1 < chars.length; at++) words.push(`${chars[at]}${chars[at + 1]}`);
      } else if (!STOP_WORDS.has(part)) {
        words.push(part);
      }
    }
  }
  return words;
}

/** The DOI a reference gives, as the record keeps it. */
export function doiOf(reference: string): string | null {
  const written = DOI_IN_TEXT.exec(reference)?.[1];
  return written ? normalizeDoi(written) : null;
}

/** The arXiv number a reference gives, without its version: "1706.03762". */
export const arxivOf = (reference: string): string | null =>
  ARXIV_IN_TEXT.exec(reference)?.[1] ?? null;

/**
 * Whether a reference names a scholarly work a check can look up: one with a
 * DOI or arXiv number, or enough words and no web address (a web page, which
 * Crossref doesn't hold, would read as "not found").
 */
function isCheckable(reference: string): boolean {
  if (doiOf(reference) || arxivOf(reference)) return true;
  if (WEB_ADDRESS.test(reference)) return false;
  return titleWords(reference).length >= MIN_WORDS;
}

interface Part {
  blockId: string;
  heading: boolean;
  text: string;
}

/** The Blocks' texts, a Block holding several numbered entries split at their numbers. */
function partsOf(blocks: readonly ListBlock[]): Part[] {
  return blocks.flatMap((block) => {
    const text = block.text.replace(/\s+/g, " ").trim();
    const parts = block.heading || HEADING.test(text) ? [text] : text.split(NUMBERED);
    return parts
      .map((part) => part.trim())
      .filter(Boolean)
      .map((part) => ({ blockId: block.id, heading: block.heading, text: part }));
  });
}

/** A heading: a list starts at one that names it, and ends at any other. */
const isHeading = (part: Part) => part.heading || HEADING.test(part.text);

/**
 * The entries of a reference list, in order; none without a heading that
 * starts one. A PDF can break an entry into paragraphs: a numbered entry
 * goes on until the next number, and an entry whose authors stand alone
 * goes on into the paragraph with its year, DOI or number.
 */
function referenceEntries(blocks: readonly ListBlock[]): ListedReference[] {
  const parts = partsOf(blocks);
  const numberedLater = (from: number) => {
    for (const part of parts.slice(from + 1)) {
      if (isHeading(part)) return false;
      if (NUMBER_START.test(part.text)) return true;
    }
    return false;
  };
  const entries: ListedReference[] = [];
  let listing = false;
  let current: ListedReference | null = null;
  let opening: ListedReference | null = null;
  for (const [at, part] of parts.entries()) {
    if (isHeading(part)) {
      listing = HEADING.test(part.text);
      current = null;
      opening = null;
      continue;
    }
    if (!listing) continue;
    if (opening) {
      opening.text = `${opening.text} ${part.text}`;
      if (REFERENCE_SIGN.test(part.text)) {
        entries.push(opening);
        current = opening;
        opening = null;
      } else if (opening.text.length > LONGEST_OPENING) {
        listing = false;
      }
    } else if (current && NUMBER_START.test(current.text) && !NUMBER_START.test(part.text)) {
      if (numberedLater(at)) current.text = `${current.text} ${part.text}`;
      else listing = false;
    } else if (REFERENCE_SIGN.test(part.text)) {
      current = { blockId: part.blockId, text: part.text };
      entries.push(current);
    } else if (AUTHOR_START.test(part.text)) {
      opening = { blockId: part.blockId, text: part.text };
    } else {
      listing = false;
    }
  }
  return entries;
}

/** The entries of a reference list that a check looks up, in order. */
export const checkableReferences = (blocks: readonly ListBlock[]): ListedReference[] =>
  referenceEntries(blocks).filter((entry) => isCheckable(entry.text));
