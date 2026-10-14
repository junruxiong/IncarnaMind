import { RemoveIcon } from "./icons";

/**
 * A quote in the composer, as a chip before its text (DESIGN.md, Composer):
 * “the words” · where they are from, with × to leave them out of the
 * Question. The tooltip is the whole passage; a click shows it where it is.
 */
export function QuoteChip({
  quote,
  from,
  removeLabel,
  onShow,
  onRemove,
  testId,
}: {
  quote: string;
  from: string;
  /** The × button's name and tooltip. */
  removeLabel: string;
  onShow(): void;
  onRemove(): void;
  testId: string;
}) {
  const words = quote.replace(/\s+/g, " ").trim();
  return (
    <span
      data-testid={testId}
      title={quote}
      className="question-skill composer-quote composer-quote--passage"
    >
      <button
        type="button"
        data-testid={`${testId}-show`}
        onClick={onShow}
        className="quote-chip-show"
      >
        <span className="truncate quote-chip-words">“{words}”</span>
        <span className="quote-chip-from truncate">· {from}</span>
      </button>
      <button
        type="button"
        data-testid={`${testId}-remove`}
        aria-label={removeLabel}
        title={removeLabel}
        // The text keeps the focus.
        onMouseDown={(event) => event.preventDefault()}
        onClick={onRemove}
        className="scope-chip-remove"
      >
        <RemoveIcon className="size-2.5" />
      </button>
    </span>
  );
}
