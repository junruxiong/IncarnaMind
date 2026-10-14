/**
 * The line of a Passage a check read that is nearest a sentence (#770): the one
 * that shares the most of its words, when it shares enough to speak to the same
 * thing. A sentence found unsupported may still have a Document that says
 * something different about it; this finds the line to show and to judge.
 */

/** Words that say what a sentence is about: four letters or more, and numbers; Chinese by pairs. */
const wordsOf = (text: string): Set<string> => {
  const words = new Set(text.toLowerCase().match(/[\p{L}\p{N}]{4,}|\d+(?:[.,]\d+)?/gu) ?? []);
  for (const run of text.match(/\p{Script=Han}{2,}/gu) ?? []) {
    for (let at = 0; at + 2 <= run.length; at++) words.add(run.slice(at, at + 2));
  }
  return words;
};

/** A mark where a new Unit starts in a Passage: "[p. 4] ", "[slide 4] ", "[§ 2.1 Sensitivity] "… */
const UNIT_MARK = /^\[(?:p\. \d+|slides? [^\]]+|§[^\]]*|[^\]]*rows? [\d–-]+|lines? [\d–-]+)\] /;

/** The fewest words a line must share with the sentence to count as speaking to it. */
const MIN_SHARED = 2;

export interface NearestLine {
  /** The id the model was shown the Passage by. */
  id: string;
  /** The line, without its Unit mark. */
  text: string;
  shared: number;
}

/** The line, among the Passages shown, that shares the most words with the sentence; null if none shares enough. */
export function nearestLine(
  sentence: string,
  passages: readonly { id: string; text: string }[],
): NearestLine | null {
  const wanted = wordsOf(sentence);
  let best: NearestLine | null = null;
  for (const passage of passages) {
    for (const raw of passage.text.split("\n")) {
      const text = raw.replace(UNIT_MARK, "").trim();
      if (text === "") continue;
      let shared = 0;
      for (const word of wordsOf(text)) if (wanted.has(word)) shared++;
      if (shared >= MIN_SHARED && (!best || shared > best.shared)) {
        best = { id: passage.id, text, shared };
      }
    }
  }
  return best;
}
