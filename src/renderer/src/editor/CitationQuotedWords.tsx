import type { CitationAttributes, QuotedPiece } from "../../../core/api";
import { closestWordsMessage, parseQuotedWords } from "../../../shared/citations";
import { useT } from "../i18n";

/** Words in quotation marks, in pieces, those washed that `wash` says are. */
function Words({
  label,
  pieces,
  wash,
  testId,
}: {
  label: string;
  pieces: readonly QuotedPiece[];
  /** Which pieces are washed, by their `found`: amber for the Answer's words the Document lacks, green for the Document's that match. */
  wash: { found: boolean; className: string };
  testId: string;
}) {
  return (
    <figure className="citation-card-words" data-testid={testId}>
      <figcaption className="citation-card-label">{label}</figcaption>
      <blockquote>
        “
        {pieces.map((piece, index) => (
          <span
            // biome-ignore lint/suspicious/noArrayIndexKey: pieces have no identity but their place.
            key={index}
            className={piece.found === wash.found ? wash.className : undefined}
          >
            {piece.text}
          </span>
        ))}
        ”
      </blockquote>
    </figure>
  );
}

/**
 * Words an Answer put in quotation marks that weren't found where its Citation
 * points (#483): the Answer's own, those the Document lacks washed amber,
 * above the Document's closest, the matching stretch washed green.
 */
export function CitationQuotedWords({ attributes }: { attributes: CitationAttributes }) {
  const t = useT();
  const quoted = parseQuotedWords(attributes.quotedWords);
  if (!quoted) return null;
  const closest = closestWordsMessage(attributes, t);
  return (
    <>
      <Words
        label={t("citation.quoted.answer")}
        pieces={quoted.answer}
        wash={{ found: false, className: "citation-card-wash" }}
        testId="citation-answer-words"
      />
      {quoted.closest && (
        <Words
          label={t(closest.key, closest.params)}
          pieces={quoted.closest}
          wash={{ found: true, className: "citation-card-match" }}
          testId="citation-closest-words"
        />
      )}
    </>
  );
}
