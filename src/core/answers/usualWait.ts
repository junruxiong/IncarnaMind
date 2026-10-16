const RECENT = 10;
const FEWEST = 3;

/**
 * How long, in whole seconds, a model usually takes before an Answer has
 * words: the median of its latest waits (`waitsMs`, newest first). Null with
 * too few to say.
 */
export function usualWait(waitsMs: readonly number[]): number | null {
  const recent = waitsMs.slice(0, RECENT).sort((a, b) => a - b);
  if (recent.length < FEWEST) return null;
  const middle = Math.floor(recent.length / 2);
  const median =
    recent.length % 2 === 1
      ? (recent[middle] ?? 0)
      : ((recent[middle - 1] ?? 0) + (recent[middle] ?? 0)) / 2;
  return Math.max(1, Math.round(median / 1000));
}
