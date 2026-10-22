import { answerSentences } from "../../shared/answerSentences";

/** An Answer with Citations is long when it runs past this many sentences. */
export const LONG_ANSWER_SENTENCES = 3;

/**
 * Whether Add to note is an Answer's one button and names its Mind: the Mind in front is
 * empty, or the Answer has Citations and is long. Otherwise it stays one item beside Copy.
 */
export function addNamesItsMind(
  answer: { text: string; citations: readonly unknown[] },
  mindEmpty: boolean,
): boolean {
  return (
    mindEmpty ||
    (answer.citations.length > 0 && answerSentences(answer.text).length > LONG_ANSWER_SENTENCES)
  );
}
