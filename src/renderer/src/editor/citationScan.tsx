/**
 * A Citation of a scanned page (#397), on its card: it says the text was read
 * from a scan, and marks the words of the quote the reading wasn't sure of,
 * so the User looks at the page before trusting them.
 */
import { Fragment, type ReactNode, useEffect, useState } from "react";
import type { CitationAttributes, ScannedQuote } from "../../../core/api";
import { normaliseText } from "../../../shared/text";
import { core } from "../core";
import { useT } from "../i18n";

/** What the core says of the Citation's quote: null until it answers, or when it isn't a scan's. */
export function useScannedQuote(attributes: CitationAttributes): ScannedQuote | null {
  const { documentId, contentHash, pageFrom, pageTo, quote } = attributes;
  const [scanned, setScanned] = useState<ScannedQuote | null>(null);
  useEffect(() => {
    if (!documentId || !quote) return;
    let current = true;
    core
      .readScannedQuote({
        documentId,
        contentHash: contentHash ?? null,
        pageFrom: pageFrom ?? null,
        pageTo: pageTo ?? null,
        quote,
      })
      .then(
        (answer) => {
          if (current) setScanned(answer);
        },
        // A card that can't say it was a scan is the card as it always was.
        (error: unknown) => console.error(error),
      );
    return () => {
      current = false;
    };
  }, [documentId, contentHash, pageFrom, pageTo, quote]);
  return scanned;
}

/** "Text read from a scan", with how many of the quote's words may be misread. */
export function ScanLine({ scanned }: { scanned: ScannedQuote }) {
  const t = useT();
  const count = scanned.unsure.length;
  return (
    <p data-testid="citation-scan" className="citation-card-source">
      {count === 0
        ? t("citation.scan")
        : t(count === 1 ? "citation.scan.unsure.one" : "citation.scan.unsure.other", { count })}
    </p>
  );
}

/**
 * The quote with each word the reading wasn't sure of marked, where the
 * quote has it (compared as the Citation check compares text).
 */
export function QuoteWithUnsureWords({
  quote,
  unsure,
}: {
  quote: string;
  unsure: readonly string[];
}) {
  const t = useT();
  if (unsure.length === 0) return <>{quote}</>;
  const wanted = new Set(unsure.map((word) => normaliseText(word)).filter(Boolean));
  const parts: ReactNode[] = [];
  // Words and the spaces between them, each Chinese character a word of its own.
  for (const [index, piece] of (
    quote.match(/\s+|[\p{Script=Han}]|[^\s\p{Script=Han}]+/gu) ?? []
  ).entries()) {
    const key = normaliseText(piece);
    parts.push(
      wanted.has(key) ? (
        <span
          key={index}
          data-testid="citation-unsure-word"
          className="citation-card-unsure"
          title={t("citation.scan.unsureWord")}
        >
          {piece}
        </span>
      ) : (
        <Fragment key={index}>{piece}</Fragment>
      ),
    );
  }
  return <>{parts}</>;
}
