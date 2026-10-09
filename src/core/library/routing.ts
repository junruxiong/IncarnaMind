/**
 * How much readable text a page has, in Latin letters' worth: letters and
 * digits count once, and a Chinese, Japanese or Korean character, which
 * says about as much as a short word, three times. Without that weight a
 * one-page Chinese invoice of 300 characters looked like a scan, and was
 * read from page images, five times slower, for no better answer.
 */
function readableText(text: string): number {
  const letters = text.match(/[\p{L}\p{N}]/gu)?.length ?? 0;
  const cjk =
    text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu)
      ?.length ?? 0;
  return letters + 2 * cjk;
}

/** Bounded text-coverage heuristic. It does not try to understand a diagram. */
export function pdfNeedsPageImages(
  pages: readonly { page: number; text: string }[],
  pageCount: number | null,
): boolean {
  const count = Math.min(pageCount ?? pages.length, 12);
  if (count < 1) return true;
  const characters = Array.from({ length: count }, (_, index) =>
    readableText(pages.find((page) => page.page === index + 1)?.text ?? ""),
  );
  return (
    characters.reduce((sum, value) => sum + value, 0) < 600 ||
    characters.filter((value) => value < 200).length >= count / 2
  );
}
