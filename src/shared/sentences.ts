/**
 * The sentences of a paragraph's text, as checks and note health count them
 * (#96): read by the core from the Mind's Yjs document and by the editor
 * from its own, the same way. In that text an inline node (a Citation, a
 * formula, a footnote) is one U+FFFC and a line break is "\n", so offsets
 * are the core's `TextPlace` offsets.
 */

/** An inline node in a paragraph's text. */
export const INLINE_NODE = "￼";

const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });

/** Where a sentence is in its paragraph's text, without the spaces around it. */
export interface SentenceSpan {
  from: number;
  to: number;
}

const isSpace = (char: string | undefined) => char !== undefined && /\s/u.test(char);

/** Holds a letter or a digit: a stray mark or a lone Citation isn't a sentence. */
const hasWords = (text: string) => /[\p{L}\p{N}]/u.test(text);

/** A title or an abbreviation the segmenter takes for a sentence's end: "Mr. Smith", "Fig. 2". */
const ABBREVIATION = /(?:^|\s)(?:mr|mrs|ms|dr|prof|st|sr|jr|vs|etc|fig|no|inc|ltd|co|corp)\.$/iu;

/**
 * The sentences of a paragraph's text, in order. A Citation's marker right
 * after a sentence's full stop ("…quarter.[1] Next") belongs to that
 * sentence, though the segmenter starts the next one at it.
 */
export function sentenceSpans(text: string): SentenceSpan[] {
  const spans: SentenceSpan[] = [];
  for (const { segment, index } of segmenter.segment(text)) {
    let from = index;
    let to = index + segment.length;
    const previous = spans.at(-1);
    while (from < to && (text[from] === INLINE_NODE || isSpace(text[from]))) {
      if (text[from] === INLINE_NODE && previous) previous.to = from + 1;
      from++;
    }
    while (to > from && isSpace(text[to - 1])) to--;
    const joined = previous && ABBREVIATION.test(text.slice(previous.from, previous.to));
    if (joined && from < to) previous.to = to;
    else if (from < to && hasWords(text.slice(from, to))) spans.push({ from, to });
    else if (previous && from < to) previous.to = to;
  }
  return spans;
}

/** The text of a sentence as checks compare it: without inline nodes, spaces collapsed. */
export const comparableSentence = (text: string): string =>
  text.replaceAll(INLINE_NODE, "").replace(/\s+/gu, " ").trim();

const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu;
const WORD = /[\p{L}\p{N}][\p{L}\p{N}'’.,-]*/gu;

/**
 * Whether a sentence makes a claim worth a Citation, for note health: at
 * least three words, or four characters of Chinese, Japanese or Korean. A
 * word or two ("Thanks!", "Next steps") claims nothing.
 */
export function isClaim(text: string): boolean {
  const plain = comparableSentence(text);
  const cjk = plain.match(CJK)?.length ?? 0;
  if (cjk >= 4) return true;
  const words = plain.replace(CJK, " ").match(WORD)?.length ?? 0;
  return words + cjk >= 3;
}
