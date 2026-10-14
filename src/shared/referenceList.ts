/**
 * A reference list in a Mind's Blocks (#711): the Blocks after a heading
 * such as "References", "Bibliography" or "参考文献", until the next heading,
 * each entry as written. A Block holding several numbered entries ("[1] …
 * [2] …"), as a PDF's list can come out, is split at their numbers. Read by
 * the core to check the list, and by the editor to offer the check.
 */

const HEADING =
  /^(?:[\dIVX]+\.?\s*)?(?:references|reference list|bibliography|works cited|literature cited|literature|参考文献|参考资料|引用文献)\s*[:：]?$/i;
/** Where a numbered entry starts inside a Block: "[12] " or "12. " after the previous entry. */
const NUMBERED = /\s(?=\[\d{1,3}\]\s)|(?<=[.)])\s+(?=\d{1,3}\.\s+\p{Lu})/u;

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

/** The entries of a reference list, in order; none without a heading that starts one. */
export function referenceEntries(blocks: readonly ListBlock[]): ListedReference[] {
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
    for (const part of text.split(NUMBERED)) {
      if (part.trim()) entries.push({ blockId: block.id, text: part.trim() });
    }
  }
  return entries;
}
