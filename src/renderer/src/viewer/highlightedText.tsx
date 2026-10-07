import type { ReactNode } from "react";
import type { TextRange } from "../../../shared/quoteMatch";

/** A string with the parts inside `ranges` (in order) marked as a quote's highlight, as the text view marks it. */
export function HighlightedText({
  text,
  ranges,
}: {
  text: string;
  ranges: readonly TextRange[] | undefined;
}): ReactNode {
  if (!ranges || ranges.length === 0) return text;
  const parts: ReactNode[] = [];
  let at = 0;
  for (const range of ranges) {
    if (range.start > at) parts.push(text.slice(at, range.start));
    parts.push(
      <mark key={range.start} data-quote-highlight="" className="quote-highlight">
        {text.slice(range.start, range.end)}
      </mark>,
    );
    at = range.end;
  }
  if (at < text.length) parts.push(text.slice(at));
  return <>{parts}</>;
}
