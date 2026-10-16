/**
 * A short Document's whole text, given with the Question as Passages (#1036):
 * a résumé, a letter or a memo shares no word with "What is this about?", so
 * a keyword search for the Question finds nothing in it. Given up front, the
 * model needs no search and its Citations are checked as a search's are.
 */
import type { WindowedPassage } from "../documents/search";
import { estimateTokens } from "./context";
import type { AnswerMessage } from "./engine";
import { withBeforeQuestion } from "./liveTranscript";

/** The most text given up front: about 6k tokens, a résumé, a letter or a memo. */
export const UP_FRONT_TOKEN_LIMIT = 6_000;
/** How much of a working copy's original opens an empty search's result. */
export const OPENING_TOKEN_LIMIT = 3_000;
// A pre-check by size: a Document over this many characters is never given whole.
const UP_FRONT_CHARACTERS = UP_FRONT_TOKEN_LIMIT * 4;

/** Every Passage when all of them fit in `tokens`, in reading order; none when they don't. */
export function allWithin(passages: readonly WindowedPassage[], tokens: number): WindowedPassage[] {
  let total = 0;
  for (const passage of passages) {
    total += estimateTokens(passage.text);
    if (total > tokens) return [];
  }
  return [...passages];
}

/** The opening Passages that fit in `tokens`, in reading order. */
export function openingWithin(
  passages: readonly WindowedPassage[],
  tokens: number,
): WindowedPassage[] {
  const kept: WindowedPassage[] = [];
  let left = tokens;
  for (const passage of passages) {
    const cost = estimateTokens(passage.text);
    if (cost > left) break;
    kept.push(passage);
    left -= cost;
  }
  return kept;
}

/**
 * The Documents given whole: the working copy's original when it fits, else
 * every Document of the scope when all of them fit together; none otherwise.
 * `sized`: the scope's Documents, `size` in characters.
 */
export function givenWhole(
  sized: readonly { id: string; size: number }[],
  workingCopy: string | null,
  passagesOf: (documentId: string) => WindowedPassage[],
): WindowedPassage[] {
  const original = workingCopy === null ? undefined : sized.find((each) => each.id === workingCopy);
  if (original && original.size <= UP_FRONT_CHARACTERS) {
    const whole = allWithin(passagesOf(original.id), UP_FRONT_TOKEN_LIMIT);
    if (whole.length > 0) return whole;
  }
  const total = sized.reduce((sum, each) => sum + each.size, 0);
  if (sized.length === 0 || total > UP_FRONT_CHARACTERS) return [];
  return allWithin(
    sized.flatMap((each) => passagesOf(each.id)),
    UP_FRONT_TOKEN_LIMIT,
  );
}

/** The messages with the Documents' text (`shown`: its Passages as the model reads them) before the Question. */
export function withDocumentText(
  messages: readonly AnswerMessage[],
  shown: string,
): AnswerMessage[] {
  return withBeforeQuestion(messages, `<document-text>\n${shown}\n</document-text>`);
}
