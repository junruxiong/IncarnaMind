import type { EditorState } from "@tiptap/pm/state";
import { ANSWER_BLOCK, CITATION_NODE, FOOTNOTE_NODE, QUESTION_BLOCK } from "../../../core/api";

/** A paragraph shorter than this gets no suggestion (the core's limit too). */
export const MIN_PARAGRAPH_CHARACTERS = 40;

/** Where a suggestion would go: the end of a paragraph of the User's body text, and its text. */
export interface SuggestionPlace {
  /** The caret's position, at the paragraph's end. */
  pos: number;
  text: string;
}

/** Blocks whose text is never continued: a Question, an Answer, or code. */
const NOT_BODY = new Set<string>([QUESTION_BLOCK, ANSWER_BLOCK, "codeBlock"]);

/**
 * The paragraph the caret ends, when a suggestion could go there: the caret
 * is a cursor (no selection) at the end of a paragraph of body text, not a
 * heading, a code block, a Question or an Answer, and the paragraph is at
 * least `MIN_PARAGRAPH_CHARACTERS` long.
 */
export function suggestionPlace(state: EditorState): SuggestionPlace | null {
  const { selection } = state;
  if (!selection.empty) return null;
  const $at = selection.$head;
  const paragraph = $at.parent;
  if (paragraph.type.name !== "paragraph" || $at.parentOffset !== paragraph.content.size) {
    return null;
  }
  for (let depth = $at.depth - 1; depth > 0; depth--) {
    if (NOT_BODY.has($at.node(depth).type.name)) return null;
  }
  const text = paragraph.textContent;
  if ([...text.trim()].length < MIN_PARAGRAPH_CHARACTERS) return null;
  return { pos: $at.pos, text };
}

/** Where a sentence ends, in English or Chinese: a Chinese full stop, or a Latin one before a space. */
const SENTENCE_END = /[。！？]|[.!?](?=\s|$|["'”’)）])/gu;

/** How many sentences a paragraph has: those it finished, and the one being written. */
export function sentencesIn(text: string): number {
  const finished = text.match(SENTENCE_END)?.length ?? 0;
  const rest = text.split(SENTENCE_END).at(-1) ?? "";
  return finished + (/[\p{L}\p{N}]/u.test(rest) ? 1 : 0);
}

const LATIN_END = /[\p{Script=Latin}\p{N}.,;:!?%)\]"'”’]$/u;
const LATIN_START = /^[\p{Script=Latin}\p{N}(["'“‘]/u;

/**
 * What goes between the paragraph and the suggestion: a space when both are
 * Latin script and the paragraph doesn't end in one; nothing in Chinese.
 */
export function joinerFor(paragraph: string, suggestion: string): string {
  return LATIN_END.test(paragraph) && LATIN_START.test(suggestion) ? " " : "";
}

const words = new Intl.Segmenter("zh", { granularity: "word" });

/**
 * The next word of a suggestion's text, with the spaces and punctuation
 * before it, and the punctuation right after it ("They", " come", " moon."):
 * what ⌥→ accepts. Chinese splits into its words.
 */
export function nextWord(text: string): string {
  let end = 0;
  let found = false;
  for (const segment of words.segment(text)) {
    if (found && (segment.isWordLike || /\s/u.test(segment.segment))) break;
    end = segment.index + segment.segment.length;
    if (segment.isWordLike) found = true;
  }
  return text.slice(0, end);
}

/**
 * The number a Citation inserted at `pos` gets: one more than the Citations
 * and footnotes before it in the Mind, which share one numbering (see
 * `numberFootnotes`). A Citation inside a footnote has no number of its own.
 */
export function citationNumberAt(state: EditorState, pos: number): number {
  let before = 0;
  state.doc.nodesBetween(0, pos, (node, at) => {
    const name = node.type.name;
    if (name === FOOTNOTE_NODE) {
      if (at + node.nodeSize <= pos) before++;
      return false;
    }
    if (name === CITATION_NODE && at < pos) before++;
    return true;
  });
  return before + 1;
}
