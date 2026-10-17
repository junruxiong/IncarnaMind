import type { UnsupportedReason } from "../../../core/api";
import { MetaLine } from "../components/MetaLine";
import { LineAction } from "../components/ProblemLine";
import { useLanguage, useT } from "../i18n";
import { openCited, quoteSaysSo } from "../openCited";
import type { NumberedCitation } from "./citationNumbers";

/**
 * What a cited sentence's Unsupported card says in place of "Nothing in your
 * Documents says this": the part its Citation doesn't cover, or the reason
 * the check gave, or that it gave none; then each Citation's quote, which
 * opens at its place.
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
  const language = useLanguage();
  const numbers = new Intl.ListFormat(language, { type: "conjunction" }).format(
    cited.map((each) => String(each.number)),
  );
  return (
    <>
      {why?.part ? (
        <p data-testid="claim-part" className="claim-card-says">
          {t(cited.length === 1 ? "claim.unsupported.part.one" : "claim.unsupported.part.other", {
            number: numbers,
            part: why.part,
          })}
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
        return (
          <div key={key} className="citation-card-note">
            <MetaLine
              data-testid="claim-cited"
              parts={[
                quote ? (
                  <span key="quote" className="line-clamp-2" title={quote}>
                    “{quote}”
                  </span>
                ) : null,
                documentId ? (
                  <LineAction
                    key="open"
                    action={{
                      label: t("claim.openCitation", { number }),
                      testId: "claim-open-citation",
                      onClick: () => {
                        openCited(
                          { ...attributes, documentId },
                          {
                            check: attributes.check,
                            withQuote: quoteSaysSo(attributes.check),
                            number,
                          },
                        );
                        onOpen();
                      },
                    }}
                  />
                ) : null,
              ]}
            />
          </div>
        );
      })}
    </>
  );
}
