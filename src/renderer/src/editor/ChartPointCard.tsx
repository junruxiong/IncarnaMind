import type { CitationAttributes } from "../../../core/api";
import { badgeMessage, type CitationState } from "../../../shared/citations";
import { citationLocation, formatLocation } from "../../../shared/locations";
import { useT } from "../i18n";
import { ChartCheck } from "./ChartCheck";

/**
 * A chart value's card (#436): what the value is, as written, and its
 * source with its check and Open, which opens the cited page with the
 * quote washed, as a Citation's Open does. A value with no Citation says so
 * quietly; one whose cell holds no number says it isn't drawn.
 */
export function ChartPointCard({
  title,
  value,
  detail,
  notDrawn,
  cut,
  citation,
  state,
  onOpen,
}: {
  title: string;
  /** As written: "$377.9 million", "1.5亿". */
  value: string;
  /** A pie slice's share, or a scatter point's value across. */
  detail: string | null;
  notDrawn: boolean;
  /** Said when the table was cut to fit the chart. */
  cut: string | null;
  citation: CitationAttributes | null;
  state: CitationState | null;
  onOpen(): void;
}) {
  const t = useT();
  const location = citation ? citationLocation(citation) : null;
  const where = location ? formatLocation(location, t) : null;
  const name = citation?.documentName ?? "";
  const badge = citation && state ? badgeMessage(state, citation, t) : null;
  const canOpen = state !== null && state.documentId !== null;
  return (
    <div className="chart-card" data-testid="chart-card-body">
      <p className="chart-card-title">{title}</p>
      <p className="chart-card-value" data-testid="chart-card-value">
        {value}
      </p>
      {detail && <p className="chart-card-detail">{detail}</p>}
      {notDrawn && <p className="chart-card-detail">{t("chart.point.notDrawn")}</p>}
      <div className="chart-card-source" data-testid="chart-card-source">
        {citation && state && badge ? (
          <>
            <ChartCheck check={state.check} label={t(badge.key, badge.params)} />
            <span className="chart-card-where" title={where ? `${name}, ${where}` : name}>
              {where ? t("chart.point.where", { document: name, location: where }) : name}
            </span>
            {canOpen && (
              <button
                type="button"
                data-testid="chart-card-open"
                className="chart-card-open"
                onClick={onOpen}
              >
                {t("chart.point.open")}
              </button>
            )}
          </>
        ) : (
          <>
            <ChartCheck check={null} label={t("chart.point.noCitation")} />
            <span className="chart-card-where">{t("chart.point.noCitation")}</span>
          </>
        )}
      </div>
      {cut && <p className="chart-card-cut">{cut}</p>}
    </div>
  );
}
