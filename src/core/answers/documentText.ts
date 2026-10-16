/**
 * A short Document's whole text, given with the Question as Passages (#1036):
 * a résumé, a letter or a memo shares no word with "What is this about?", so
 * a keyword search for the Question finds nothing in it. Given up front, the
 * model needs no search and its Citations are checked as a search's are.
 */
import type { WindowedPassage } from "../documents/search";
import type { ContextWindow } from "../providers/models";
import { estimateTokens } from "./context";
import type { SizedDocument } from "./coverage";

export const UP_FRONT_TOKEN_LIMIT = 6_000;
export const OPENING_TOKEN_LIMIT = 3_000;
/** Summing a large scope's sizes takes long (half a second at 300k Passages), and its text wouldn't fit. */
export const SMALL_SCOPE_DOCUMENTS = 12;

/** At most a quarter of the model's window, as a live transcript gets, when the window is known. */
export const upFrontTokens = (window: ContextWindow | undefined) =>
  window ? Math.min(UP_FRONT_TOKEN_LIMIT, Math.floor(window.tokens / 4)) : UP_FRONT_TOKEN_LIMIT;

// Long enough that a match is the overlap, not a phrase the next Passage happens to start with.
const OVERLAP_PROBE = 40;

/**
 * The Passages in reading order with what each repeats of the one before it
 * left out: consecutive Passages of a Document overlap (a sliding window).
 * A Passage left with nothing goes.
 */
export function withoutOverlap(passages: readonly WindowedPassage[]): WindowedPassage[] {
  const kept: WindowedPassage[] = [];
  let previous: WindowedPassage | undefined;
  for (const passage of passages) {
    let text = passage.text;
    if (previous && previous.documentId === passage.documentId) {
      const before = previous.text;
      const probe = text.slice(0, OVERLAP_PROBE);
      for (let at = before.indexOf(probe); at >= 0; at = before.indexOf(probe, at + 1)) {
        if (text.startsWith(before.slice(at))) {
          text = text.slice(before.length - at).trimStart();
          break;
        }
      }
    }
    previous = passage;
    if (text !== "") kept.push(text === passage.text ? passage : { ...passage, text });
  }
  return kept;
}

/** The opening Passages, without their overlap, that fit in `tokens`, in reading order. */
export function openingWithin(
  passages: readonly WindowedPassage[],
  tokens: number,
): WindowedPassage[] {
  const kept: WindowedPassage[] = [];
  let left = tokens;
  for (const passage of withoutOverlap(passages)) {
    const cost = estimateTokens(passage.text);
    if (cost > left) break;
    kept.push(passage);
    left -= cost;
  }
  return kept;
}

/**
 * The first of `candidates` (each a set of Documents) whose whole text, as
 * `block` sends it, fits in `tokens`: its Passages without their overlap,
 * and the block. Null when none fits.
 */
export function givenUpFront(
  candidates: readonly (readonly SizedDocument[])[],
  tokens: number,
  passagesOf: (documentId: string) => WindowedPassage[],
  block: (passages: readonly WindowedPassage[]) => string,
): { passages: WindowedPassage[]; block: string } | null {
  for (const documents of candidates) {
    // A token is at most 4 characters, and overlapping Passages hold less than twice the text.
    const size = documents.reduce((sum, each) => sum + each.size, 0);
    if (documents.length === 0 || size > tokens * 4 * 2) continue;
    const passages = withoutOverlap(documents.flatMap((each) => passagesOf(each.id)));
    if (passages.length === 0) continue;
    const text = block(passages);
    if (estimateTokens(text) <= tokens) return { passages, block: text };
  }
  return null;
}

/** The block that gives these Documents' whole text (`shown`: its Passages as the model reads them). */
export function documentTextBlock(names: readonly string[], shown: string): string {
  const which = names.map((name) => `"${name}"`).join(", ");
  return `<document-text>\nThe whole text of ${which}, as Passages. Answer from them and cite them as you cite any Passage; search only for what they don't hold.\n\n${shown}\n</document-text>`;
}
