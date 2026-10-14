/**
 * A Question asked during a call (#144): the assistant listens to the live
 * transcript, and reads the shared screen when the User chose a window to
 * read (ADR-0024). The meeting's Transcript, indexed up to the moment the
 * Question is asked, goes with it as Passages the model can cite by their
 * times, the newest that fit; the recent screens go as text the model can
 * cite as it cites a Passage. Both come after the Mind it reads and before
 * what the User selected there and the Question itself.
 */
import type { WindowedPassage } from "../documents/search";
import type { ContextWindow } from "../providers/models";
import { estimateTokens } from "./context";
import type { AnswerMessage } from "./engine";

/** The most of a live transcript a Question takes: about 8k tokens, half an hour of talk or so. */
export const LISTENED_TOKEN_LIMIT = 8_000;

/** How much of the transcript a model reads: less where its window is small. */
export const listenedTokens = (window: ContextWindow | undefined) =>
  window ? Math.min(LISTENED_TOKEN_LIMIT, Math.floor(window.tokens / 4)) : LISTENED_TOKEN_LIMIT;

/** A screen read from the shared window, as a Question gets it. */
export interface SharedScreenText {
  id: string;
  /** When it was read (ISO 8601). */
  at: string;
  window: string;
  /** As read. */
  text: string;
  /** As the model reads it: each figure that may be misread marked "⟦…⟧". */
  marked: string;
  /** The Library Passages its text is from, citable after the call. */
  library: WindowedPassage[];
}

/** What a Question asked during a call goes with. */
export interface Listened {
  /** The Transcript's Passages, indexed up to now, in reading order. */
  transcript: WindowedPassage[];
  /** The screens read from the shared window, in the order read. */
  screens: SharedScreenText[];
}

/** The newest Passages that fit in `tokens`, in reading order. */
export function newestWithin(
  passages: readonly WindowedPassage[],
  tokens: number,
): WindowedPassage[] {
  const kept: WindowedPassage[] = [];
  let left = tokens;
  for (let index = passages.length - 1; index >= 0; index--) {
    const passage = passages[index] as WindowedPassage;
    const cost = estimateTokens(passage.text);
    if (cost > left) break;
    kept.push(passage);
    left -= cost;
  }
  return kept.reverse();
}

/**
 * The messages with what was heard and seen in the Question's message: the
 * screens (`screens`, as the model reads them) and the transcript (`shown`:
 * its Passages as the model reads them), after the Mind read with it, where
 * that message holds the Mind, so the Mind keeps its place for the prompt
 * cache and the selection stays right before the Question. Either may be "".
 */
export function withTranscript(
  messages: readonly AnswerMessage[],
  shown: string,
  screens = "",
): AnswerMessage[] {
  const last = messages.at(-1);
  if (last?.role !== "user") return [...messages];
  const blocks = [screens, shown ? `<live-transcript>\n${shown}\n</live-transcript>` : ""].filter(
    Boolean,
  );
  if (blocks.length === 0) return [...messages];
  const heard = blocks.join("\n\n");
  const at = messages.length === 1 ? (last.stablePrefix ?? 0) : 0;
  const content =
    at > 0
      ? `${last.content.slice(0, at)}\n\n${heard}\n\n${last.content.slice(at).trimStart()}`
      : `${heard}\n\n${last.content}`;
  return [...messages.slice(0, -1), { ...last, content }];
}
