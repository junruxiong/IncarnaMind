/**
 * Leftover echo (#941). The echo canceller (./echo) takes most of the
 * computer's sound out of the microphone, and what is left can still be read
 * as words. A line from the microphone that repeats a line of the computer's
 * sound said at the same time is that echo, and the other side's words must
 * not show as the User's.
 */
import { SPEAKER_ME, SPEAKER_THEM } from "../../shared/speakers";
import type { Cue } from "../documents/formats/vtt";

/**
 * How much of the shorter line the two must share in time: the two streams cut speech in different
 * places, but a reply that begins as the other side ends only touches their line.
 */
const TOGETHER = 0.5;
/** How much of a line must be found in what they said: the echo is read from a degraded copy. */
const CONTAINED = 0.6;
/** A line of fewer pairs of letters than this ("Right.") is an echo only when it is a whole line of theirs. */
const SHORT_PAIRS = 8;
const SAME = 0.8;

type Pairs = Map<string, number>;

/** Pairs of neighbouring letters, case and punctuation aside: it reads Chinese as it does English. */
function pairsOf(text: string): Pairs {
  const letters = Array.from(
    text
      .normalize("NFKC")
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]/gu, ""),
  );
  const pairs: Pairs = new Map();
  if (letters.length === 1) pairs.set(letters[0] as string, 1);
  for (let i = 0; i + 1 < letters.length; i++) {
    const pair = `${letters[i]}${letters[i + 1]}`;
    pairs.set(pair, (pairs.get(pair) ?? 0) + 1);
  }
  return pairs;
}

const sizeOf = (pairs: Pairs) => [...pairs.values()].reduce((sum, count) => sum + count, 0);

function shared(a: Pairs, b: Pairs): number {
  let count = 0;
  for (const [pair, times] of a) count += Math.min(times, b.get(pair) ?? 0);
  return count;
}

/** Whether `mine` repeats what is in `theirs`, the lines said at the same time. */
export function repeatsTheirs(mine: string, theirs: readonly string[]): boolean {
  const own = pairsOf(mine);
  const size = sizeOf(own);
  if (size === 0 || theirs.length === 0) return false;
  if (size >= SHORT_PAIRS) {
    return shared(own, pairsOf(theirs.join(" "))) / size >= CONTAINED;
  }
  return theirs.some((line) => {
    const other = pairsOf(line);
    return (2 * shared(own, other)) / (size + sizeOf(other)) >= SAME;
  });
}

/** Whether two lines are said at the same time. */
function together(mine: Cue, theirs: Cue): boolean {
  const shared = Math.min(mine.endMs, theirs.endMs) - Math.max(mine.startMs, theirs.startMs);
  const shorter = Math.min(mine.endMs - mine.startMs, theirs.endMs - theirs.startMs);
  return shared > 0 && shared >= TOGETHER * shorter;
}

/** Whether a line of the microphone is the echo of a line of the computer's sound among `cues`. */
export function isEcho(mine: Cue, cues: readonly Cue[]): boolean {
  if (mine.speaker !== SPEAKER_ME) return false;
  const theirs = cues.filter((cue) => cue.speaker === SPEAKER_THEM && together(mine, cue));
  return repeatsTheirs(
    mine.text,
    theirs.map((cue) => cue.text),
  );
}

/** The lines of the microphone among `cues` that are the echo of `theirs`, now it is among them. */
export function echoedBy(theirs: Cue, cues: readonly Cue[]): Cue[] {
  return cues.filter(
    (cue) => cue.speaker === SPEAKER_ME && together(cue, theirs) && isEcho(cue, cues),
  );
}
