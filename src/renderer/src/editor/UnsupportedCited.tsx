import type { UnsupportedReason } from "../../../core/api";
import { MetaLine } from "../components/MetaLine";
import { LineAction } from "../components/ProblemLine";
import { useT } from "../i18n";
import { openCited, quoteSaysSo } from "../openCited";
import type { NumberedCitation } from "./citationNumbers";

/**
 * What a cited sentence's Unsupported card says in place of "Nothing in your
 * Documents says this": what its Citations support and what they don't, or
 * the reason the check gave, or that it gave none; then each Citation's quote,
 * which opens at its place.
 */
export function UnsupportedCited({
  cited,
  why,
  onCheckAgain,
  onOpen,
}: {
  cited: readonly NumberedCitation[];
  why: UnsupportedReason | null;
  onCheckAgain(): void;
  onOpen(): void;
}) {
  const t = useT();
  return (
    <>
      {why?.covered && why.uncovered ? (
        <p data-testid="claim-covers" className="claim-card-says">
          {t(
            cited.length === 1 ? "claim.unsupported.covers.one" : "claim.unsupported.covers.other",
            {
              covered: why.covered,
              uncovered: why.uncovered,
            },
          )}
        </p>
      ) : why?.reason ? (
        <p data-testid="claim-why" className="claim-card-says">
          {why.reason}
        </p>
      ) : (
        <div className="citation-card-note">
          <MetaLine
            data-testid="claim-no-reason"
            parts={[
              t("claim.unsupported.noReason"),
              <LineAction
                key="again"
                action={{
                  label: t("claim.checkAgain"),
                  testId: "claim-no-reason-check",
                  onClick: onCheckAgain,
                }}
              />,
            ]}
          />
        </div>
      )}
      {cited.map(({ key, number, attributes }) => {
        const { documentId, quote } = attributes;
        const content = (
          <>
            {quote && (
              <span data-testid="claim-cited-quote" className="line-clamp-2 min-w-0">
                “{quote}”
              </span>
            )}
            <span
              data-testid="claim-cited-chip"
              data-check={attributes.check}
              className="citation-marker claim-card-cited-chip"
            >
              {number}
            </span>
          </>
        );
        return documentId ? (
          <button
            key={key}
            type="button"
            data-testid="claim-cited"
            className="claim-card-cited"
            title={quote ?? undefined}
            aria-label={t("claim.openCitation", { number })}
            onClick={() => {
              openCited(
                { ...attributes, documentId },
                {
                  check: attributes.check,
                  withQuote: quoteSaysSo(attributes.check),
                  number,
                },
              );
              onOpen();
            }}
          >
            {content}
          </button>
        ) : (
          <div key={key} data-testid="claim-cited" className="claim-card-cited">
            {content}
          </div>
        );
      })}
    </>
  );
}
