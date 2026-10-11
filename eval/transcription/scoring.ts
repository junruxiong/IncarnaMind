/**
 * Scoring transcription (#136, until #85's set and scorer): word error rate
 * for English, character error rate for Chinese, after the same normalisation
 * on both sides. The rules, fixed and tested (tests/eval/transcriptionScoring.test.ts):
 * - case is folded, and full-width letters and digits become ASCII (NFKC);
 * - punctuation and symbols are dropped, in both scripts;
 * - English is cut into words at whitespace; Chinese is compared character
 *   by character, Han characters and each run of Latin letters or digits as
 *   one unit, with whitespace dropped;
 * - numbers are left as written: "6" and "六" differ. #85 fixes a rule for them.
 */

/** Text with case folded and punctuation and symbols gone, words separated by single spaces. */
export function normalise(text: string): string {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/(\p{L})['’](\p{L})/gu, "$1$2")
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** English: the words. */
export const words = (text: string): string[] => normalise(text).split(" ").filter(Boolean);

/** Chinese: each Han character, and each run of Latin letters or digits, as a unit. */
export const characters = (text: string): string[] =>
  normalise(text).match(/\p{Script=Han}|[^\p{Script=Han}\s]+/gu) ?? [];

/** Edits (substitutions, deletions, insertions) to turn `hypothesis` into `reference`. */
export function editDistance(reference: readonly string[], hypothesis: readonly string[]): number {
  let previous = Array.from({ length: hypothesis.length + 1 }, (_, index) => index);
  for (let i = 1; i <= reference.length; i++) {
    const current = [i];
    for (let j = 1; j <= hypothesis.length; j++) {
      const substitution =
        (previous[j - 1] as number) + (reference[i - 1] === hypothesis[j - 1] ? 0 : 1);
      current[j] = Math.min(
        substitution,
        (previous[j] as number) + 1,
        (current[j - 1] as number) + 1,
      );
    }
    previous = current;
  }
  return previous[hypothesis.length] as number;
}

export interface ErrorRate {
  errors: number;
  /** Units in the reference. */
  total: number;
  rate: number;
}

export function errorRate(reference: readonly string[], hypothesis: readonly string[]): ErrorRate {
  const errors = editDistance(reference, hypothesis);
  return {
    errors,
    total: reference.length,
    rate: reference.length === 0 ? 0 : errors / reference.length,
  };
}
