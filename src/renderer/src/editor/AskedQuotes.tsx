import type { AskedQuote } from "../../../core/api";
import { citationLocation, formatLocation, type Translate } from "../../../shared/locations";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { QuoteChip } from "./QuoteChip";

/** Where a quote is from, as a chip and a Question name it: its Document and Location. */
export function quoteSource(quote: AskedQuote, t: Translate): string {
  const location = citationLocation(quote);
  return [quote.documentName, location ? formatLocation(location, t) : ""]
    .filter((part) => part !== "")
    .join(" ");
}

// A Question and the composer hold each quote once.
const keyOf = (quote: AskedQuote) =>
  `${quote.documentId}:${JSON.stringify(quote.location)}:${quote.quote}`;

export const showQuote = (quote: AskedQuote) =>
  useAppStore.getState().openDocument({
    documentId: quote.documentId,
    ...(quote.pageFrom !== null && { pageFrom: quote.pageFrom }),
    ...(quote.pageTo !== null && { pageTo: quote.pageTo }),
    quote: quote.quote,
  });

/** The words selected in Documents to ask about (#101), in the composer: a chip each, with ×. */
export function AskedQuoteChips({
  quotes,
  onRemove,
}: {
  quotes: readonly AskedQuote[];
  onRemove(quote: AskedQuote): void;
}) {
  const t = useT();
  return quotes.map((quote) => (
    <QuoteChip
      key={keyOf(quote)}
      testId="composer-quote"
      whereFirst
      quote={quote.quote}
      from={quoteSource(quote, t)}
      removeLabel={t("composer.selection.drop")}
      onShow={() => showQuote(quote)}
      onRemove={() => onRemove(quote)}
    />
  ));
}

/**
 * The words a Question was asked about, above its own in the thread: each with where it is
 * from, which shows it there, or says its Document is gone.
 */
export function AskedQuoteLines({ quotes }: { quotes: readonly AskedQuote[] }) {
  const t = useT();
  const documents = useAppStore((state) => state.documents);
  return quotes.map((quote) => {
    const source = quoteSource(quote, t);
    return (
      <span key={keyOf(quote)} className="assistant-question-asked">
        <span
          data-testid="assistant-question-quote"
          title={quote.quote}
          className="assistant-question-quote"
        >
          <span className="sr-only">{t("assistant.question.quote")}</span>“{quote.quote.trim()}”
        </span>
        {documents.some((each) => each.id === quote.documentId) ? (
          <button
            type="button"
            data-testid="assistant-question-quote-source"
            onClick={() => showQuote(quote)}
            className="assistant-question-quote-source"
          >
            {source}
          </button>
        ) : (
          <span
            data-testid="assistant-question-quote-source"
            className="assistant-question-quote-source"
          >
            {t("assistant.question.quoteGone", { source })}
          </span>
        )}
      </span>
    );
  });
}
