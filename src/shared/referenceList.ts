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

/** A DOI as a reference writes it, after "doi:" or "https://doi.org/" or alone. */
export const DOI_IN_TEXT = /(?:doi\.org\/|doi\s*[:：]?\s*)?(10\.\d{4,9}\/[^\s"<>]+)/i;
/** An arXiv number as a reference writes it: "arXiv:1706.03762", or in an arxiv.org address. */
export const ARXIV_IN_TEXT = new RegExp(
  `(?:arxiv\\s*[:：]\\s*|arxiv\\.org/(?:abs|pdf)/)${ARXIV_ID.source}`,
  "i",
);
/** A year a reference names: four digits from 1500 to 2099. */
export const YEAR_IN_TEXT = /(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)/;
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

/**
 * The DOI a reference gives, as the record keeps it; for a preprint on arXiv
 * given by its number, the DOI arXiv registers for it ("10.48550/arXiv.…").
 */
export function doiOf(reference: string): string | null {
  const written = DOI_IN_TEXT.exec(reference)?.[1];
  if (written) return normalizeDoi(written);
  const arxiv = ARXIV_IN_TEXT.exec(reference)?.[1];
  return arxiv ? `10.48550/arXiv.${arxiv}` : null;
}

/**
 * Whether a reference names a scholarly work a check can look up: one with a
 * DOI or arXiv number, or enough words and no web address (a web page, which
 * Crossref doesn't hold, would read as "not found").
 */
function isCheckable(reference: string): boolean {
  if (doiOf(reference)) return true;
  if (WEB_ADDRESS.test(reference)) return false;
  return titleWords(reference).length >= MIN_WORDS;
}

/** The entries of a reference list, in order; none without a heading that starts one. */
function referenceEntries(blocks: readonly ListBlock[]): ListedReference[] {
  const entries: ListedReference[] = [];
  let listing = false;
  for (const block of blocks) {
    const text = block.text.replace(/\s+/g, " ").trim();
    if (HEADING.test(text)) {
      listing = true;
      continue;
    }
    if (block.heading) {
      listing = false;
      continue;
    }
    if (!listing || !text) continue;
    for (const part of text.split(NUMBERED).map((each) => each.trim())) {
      if (!part) continue;
      if (!REFERENCE_SIGN.test(part)) {
        listing = false;
        break;
      }
      entries.push({ blockId: block.id, text: part });
    }
  }
  return entries;
}

/** The entries of a reference list that a check looks up, in order. */
export const checkableReferences = (blocks: readonly ListBlock[]): ListedReference[] =>
  referenceEntries(blocks).filter((entry) => isCheckable(entry.text));
