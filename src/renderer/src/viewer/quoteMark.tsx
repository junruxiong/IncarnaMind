import type { CSSProperties } from "react";
import type { ViewerCitation } from "../../../shared/documentViewer";
import { useT } from "../i18n";
import { ExclamationIcon, TickIcon } from "./icons";

/**
 * How a quote is washed: green, the colour of a found Citation's check, unless
 * it was opened anyway after its check said "not found", which is amber.
 */
export type QuoteTone = "found" | "not-found";

export const quoteTone = (citation: ViewerCitation | undefined): QuoteTone =>
  citation?.check === "not-found" ? "not-found" : "found";

/** The check mark to show beside a quote: only when the Citation's check and number are known. */
export interface QuoteMark {
  tone: QuoteTone;
  number: number;
}

export function quoteMarkOf(citation: ViewerCitation | undefined): QuoteMark | null {
  if (citation?.number === undefined) return null;
  if (citation.check !== "found" && citation.check !== "not-found") return null;
  return { tone: citation.check, number: citation.number };
}

/** The mark's height, as in the Mind's margin. */
const MARK_HEIGHT = 18;

/**
 * Where the mark's top goes, inside `container`, to centre it on the first line
 * of `highlight` (a quote's first highlighted part).
 */
export function markTopFor(highlight: Element, container: Element): number {
  const line = highlight.getClientRects()[0] ?? highlight.getBoundingClientRect();
  return line.top - container.getBoundingClientRect().top + line.height / 2 - MARK_HEIGHT / 2;
}

/** The Citation's check mark (its icon and number), in the page's margin beside its quote. */
export function CitationMark({ mark, style }: { mark: QuoteMark; style: CSSProperties }) {
  const t = useT();
  const label = t(mark.tone === "found" ? "viewer.mark.found" : "viewer.mark.notFound", {
    number: mark.number,
  });
  const Icon = mark.tone === "found" ? TickIcon : ExclamationIcon;
  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      data-testid="viewer-quote-mark"
      data-check={mark.tone}
      className={`quote-mark quote-mark--${mark.tone}`}
      style={style}
    >
      <Icon className="size-[11px] shrink-0" />
      {mark.number}
    </span>
  );
}
