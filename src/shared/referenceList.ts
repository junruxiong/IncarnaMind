/**
 * A reference list in a Mind's Blocks: the Blocks after a heading such as
 * "References" or "参考文献", until the next heading or the first paragraph
 * of prose. Read by the core to check the list, and by the editor to offer
 * the check, so both count the same entries.
 */
import { ARXIV_ID, normalizeDoi } from "./references";

const HEADING =
  /^(?:[\dIVX]+\.?\s*)?(?:references|reference list|bibliography|works cited|literature cited|literature|参考文献|参考资料|引用文献)\s*[:：]?$/i;
/** Paragraphs that end a list no heading closes. */
const AFTER_LIST =
  /^(?:[\dA-Z]{1,3}\.?\s+)?(?:acknowledge?ments?|appendix|appendices|supplementary material|致谢|附录)/i;
/** Where a numbered entry starts inside a Block, as a PDF's list can come out: "[12] ", "。[12]" or "12. ". */
const NUMBERED = /\s(?=\[\d{1,3}\])|(?<=[.。])(?=\[\d{1,3}\])|(?<=[.)])\s+(?=\d{1,3}\.\s+\p{Lu})/u;
/** "[12]", "12." or "12)", with or without a space after it, but not a DOI's "10.1016". */
const NUMBER_START = /^(?:\[\d{1,3}\]|\d{1,3}[.)](?!\d))/;
const PARTICLES = "(?:(?:van|von|de|der|den|du|da|di|del|della|le|la|ten|ter)\\s+)*";
const NAME = "\\p{Lu}[\\p{L}'’-]+";
/** How an entry starts with its authors, as styles write them; not "However, the" or "We thank". */
const ENTRY_START = new RegExp(
  [
    // "Maas, Andrew L", "Tieleman, T.", "Amari, Shun-Ichi.", "van der Maaten, L."
    `^${PARTICLES}${NAME},\\s+(?:\\p{Lu}\\.|${NAME}(?:,|\\.|\\s+\\p{Lu}\\b|\\s+and\\b))`,
    // "A. Vaswani", "J.-P. Sartre"
    `^\\p{Lu}\\.(?:[\\s-]?\\p{Lu}\\.)*\\s+${PARTICLES}${NAME}`,
    // "Diederik P. Kingma", not "Jane Doe and"
    `^${NAME}\\s+(?:\\p{Lu}\\.\\s*)+${PARTICLES}${NAME}`,
    // "Smith J,", "SMITH JA."
    `^${PARTICLES}${NAME}\\s+\\p{Lu}{1,3}[,.]`,
    // "OpenAI. (2022).", "World Health Organization (2020)."
    `^(?:${NAME}\\s+){0,5}${NAME}\\.?\\s*\\((?:1[5-9]|20)\\d\\d[a-z]?\\)\\.`,
    // "张三,", "周志华.", "中华人民共和国国家统计局."
    "^\\p{Script=Han}{2,4}[,，.。、]|^\\p{Script=Han}{5,30}\\.",
  ].join("|"),
  "u",
);
/** What a reference's last part names besides its year: a venue, a publisher, pages or a volume, an identifier. */
const VENUE_WORD =
  /\b(?:proc|proceedings|journal|conference|symposium|workshop|press|report|thesis|review|letters|transactions|arxiv|preprint|edition|publisher|(?:vol|pp|no)\.)/i;
const VENUE_MARK =
  /(?:^|[.,;]\s+)In\s+\p{Lu}|\b\p{Lu}{2,}\b|\d+\s*\(\d+\)|\d+\s*[-–—]\s*\d+|doi|https?:\/\//u;
const endsReference = (text: string) => VENUE_WORD.test(text) || VENUE_MARK.test(text);
/** An entry's paragraphs before its year, DOI or number run no longer than this. */
const LONGEST_OPENING = 400;
/** The longest paragraph read as the end of a last numbered entry broken over two. */
const SHORT_PART = 160;

export const DOI_IN_TEXT = /(?:doi\.org\/|doi\s*[:：]?\s*)?(10\.\d{4,9}\/[^\s"<>]+)/i;
/** An arXiv number as a reference writes it: "arXiv:1706.03762", "CoRR, abs/1409.0473", or in an arxiv.org address. */
export const ARXIV_IN_TEXT = new RegExp(
  `(?:arxiv\\s*[:：]\\s*|arxiv\\.org/(?:abs|pdf)/|\\bcorr,?\\s*abs/)${ARXIV_ID.source}`,
  "i",
);
const YEARS = /(?<!\d)(1[5-9]\d\d|20\d\d)(?!\d)/g;
/** What every reference holds one of, a year or a date standing in for one, as styles write them. */
const REFERENCE_SIGN = new RegExp(
  [
    YEARS.source,
    DOI_IN_TEXT.source,
    ARXIV_IN_TEXT.source,
    "https?://",
    NUMBER_START.source,
    "\\bin press\\b",
    "\\bn\\.\\s?d\\.",
    "\\[(?:J|M|C|D|R|S|P|N|EB/OL|Z)\\]",
  ].join("|"),
  "i",
);
/** A DOI a PDF broke after a dot, slash or hyphen: "10.1016/j.neunet. 2014.09.003". */
const BROKEN_DOI = /(10\.\d{4,9}\/\S*[./-])\s+(?=\d[\w-]*[./][\w./-]*\d)/g;

export const joinBrokenDoi = (text: string) => text.replace(BROKEN_DOI, "$1");

/**
 * The year a reference names. A number joined to another by a dash is a page
 * in a range, unless both are years and nothing else names one; then a year
 * in brackets, else the last: "(2019)", "pp. 1735-1780, 1997", "(2017–2018)",
 * "(2019-06-12)". Never part of a number ("abs/1512.00567").
 */
export function yearOf(text: string): number | null {
  const found: { year: number; bracketed: boolean; ranged: boolean }[] = [];
  for (const match of text.matchAll(YEARS)) {
    const before = text.slice(0, match.index);
    const after = text.slice(match.index + 4);
    if (/[/:.]$/.test(before) || /^\.\d/.test(after)) continue;
    const date = /^-\d{2}(?:-\d{2})?(?!\d)/.test(after);
    const ranged = !date && (/^\s?[-–—]\s?\d/.test(after) || /\d\s?[-–—]\s?$/.test(before));
    found.push({ year: Number(match[1]), bracketed: /\(\s*$/.test(before), ranged });
  }
  const alone = found.filter((each) => !each.ranged);
  const chosen =
    alone.find((each) => each.bracketed) ??
    alone.at(-1) ??
    found.find((each) => each.bracketed) ??
    found[0];
  return chosen?.year ?? null;
}
/** Fewer title words than this, and a reference says too little to look up. */
const MIN_WORDS = 4;
const WEB_ADDRESS = /https?:\/\/\S+/i;

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

export function doiOf(reference: string): string | null {
  const written = DOI_IN_TEXT.exec(joinBrokenDoi(reference))?.[1];
  return written ? normalizeDoi(written) : null;
}

/** The arXiv number a reference gives, without its version: "1706.03762". */
export const arxivOf = (reference: string): string | null =>
  ARXIV_IN_TEXT.exec(reference)?.[1] ?? null;

/** A web page, which Crossref doesn't hold, would read as "not found". */
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

/** A piece of an entry a PDF broke off: a DOI or address alone, or words going on in lower case. */
const continues = (text: string) =>
  !/\s/.test(text) || (/^\p{Ll}/u.test(text) && !REFERENCE_SIGN.test(text));

/**
 * The entries of a reference list, in order; none without a heading that
 * starts one. A numbered entry, or one whose authors come first, starts an
 * entry. A PDF can break an entry into paragraphs: a numbered one goes on
 * until the next number, and one whose authors stand alone goes on into the
 * paragraph with its year. The list ends at a paragraph that is neither.
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
  // An object, so the closures below and the loop see one state.
  const state: {
    listing: boolean;
    current: ListedReference | null;
    opening: ListedReference | null;
  } = { listing: false, current: null, opening: null };
  const start = (part: Part) => {
    state.current = { blockId: part.blockId, text: part.text };
    entries.push(state.current);
  };
  const closeOpening = () => {
    if (state.opening) entries.push(state.opening);
    state.current = state.opening;
    state.opening = null;
  };
  const end = () => {
    closeOpening();
    state.listing = false;
    state.current = null;
  };
  const append = (entry: ListedReference, text: string) => {
    entry.text = `${entry.text} ${text}`;
  };
  for (const [at, part] of parts.entries()) {
    const { text } = part;
    if (isHeading(part)) {
      end();
      state.listing = HEADING.test(text);
      continue;
    }
    if (!state.listing) continue;
    if (AFTER_LIST.test(text)) {
      end();
      continue;
    }
    const numbered = NUMBER_START.test(text);
    const startsEntry = numbered || ENTRY_START.test(text);
    if (state.opening && startsEntry) closeOpening();
    if (state.opening && REFERENCE_SIGN.test(text) && !endsReference(text)) {
      // A dated sentence after an undated entry is prose, not the entry's end.
      end();
    } else if (state.opening) {
      append(state.opening, text);
      if (REFERENCE_SIGN.test(text)) closeOpening();
      else if (state.opening.text.length > LONGEST_OPENING) {
        state.opening = null;
        end();
      }
    } else if (numbered) {
      start(part);
    } else if (state.current && NUMBER_START.test(state.current.text)) {
      const lastFinishes =
        yearOf(state.current.text) === null &&
        REFERENCE_SIGN.test(text) &&
        endsReference(text) &&
        text.length <= SHORT_PART;
      if (numberedLater(at) || continues(text) || lastFinishes) append(state.current, text);
      else end();
    } else if (startsEntry) {
      if (REFERENCE_SIGN.test(text)) start(part);
      else state.opening = { blockId: part.blockId, text };
    } else if (state.current && continues(text)) {
      append(state.current, text);
    } else if (entries.length === 0 && REFERENCE_SIGN.test(text)) {
      start(part);
    } else {
      end();
    }
  }
  closeOpening();
  return entries;
}

export const checkableReferences = (blocks: readonly ListBlock[]): ListedReference[] =>
  referenceEntries(blocks).filter((entry) => isCheckable(entry.text));
