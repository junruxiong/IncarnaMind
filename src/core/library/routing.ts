/** Bounded text-coverage heuristic. It does not try to understand a diagram. */
export function pdfNeedsPageImages(
  pages: readonly { page: number; text: string }[],
  pageCount: number | null,
): boolean {
  const count = Math.min(pageCount ?? pages.length, 12);
  if (count < 1) return true;
  const characters = Array.from({ length: count }, (_, index) => {
    const text = pages.find((page) => page.page === index + 1)?.text ?? "";
    return text.match(/[\p{L}\p{N}]/gu)?.length ?? 0;
  });
  return (
    characters.reduce((sum, value) => sum + value, 0) < 600 ||
    characters.filter((value) => value < 200).length >= count / 2
  );
}
