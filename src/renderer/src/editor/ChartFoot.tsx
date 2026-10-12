import type { CitationCheck } from "../../../core/api";
import type { ChartInk } from "../../../shared/chartLayout";
import { type ChartSources, MAX_CHART_VALUES } from "../../../shared/charts";
import { citationLocation, formatLocation } from "../../../shared/locations";
import { useT } from "../i18n";
import { ChartCheck } from "./ChartCheck";

/**
 * Under a chart (#436): the key to its series' colours, where its values
 * were quoted from, with one check for them all, and what was left out: a
 * table cut to fit, and cells with no number. When its table changed since,
 * a quiet line offers to draw it as it is now.
 */
export function ChartFoot({
  legend,
  sources,
  check,
  notFound,
  rows,
  cutFrom,
  onUpdate,
}: {
  legend: { ink: ChartInk; name: string }[];
  sources: ChartSources;
  /** The check of all the values cited, the worst first; null when none is. */
  check: CitationCheck | null;
  /** Values whose quote wasn't found. */
  notFound: number;
  rows: number;
  cutFrom: number | null;
  /** Its table changed since: draws it as it is now. Null when it hasn't, or can't be. */
  onUpdate: (() => void) | null;
}) {
  const t = useT();
  const cited = sources.values - sources.uncited;
  const place = sources.places[0]?.citation;
  const location = place ? citationLocation(place) : null;
  const document = place?.documentName ?? "";
  const plural = (count: number) => (count === 1 ? "one" : "other");

  let said: string | null = null;
  if (sources.values > 0 && cited === 0) {
    said = t(`chart.source.none.${plural(sources.values)}`, { count: sources.values });
  } else if (cited > 0 && sources.places.length === 1 && location) {
    said = t(`chart.source.place.${plural(cited)}`, {
      count: cited,
      document,
      location: formatLocation(location, t),
    });
  } else if (cited > 0 && sources.documents === 1) {
    said = t(`chart.source.document.${plural(cited)}`, { count: cited, document });
  } else if (cited > 0) {
    said = t("chart.source.documents", { count: cited, documents: sources.documents });
  }
  const more = [
    cited > 0 && sources.uncited > 0 ? t("chart.source.uncited", { count: sources.uncited }) : null,
    notFound > 0 ? t(`chart.source.notFound.${plural(notFound)}`, { count: notFound }) : null,
    sources.notDrawn > 0
      ? t(`chart.source.notDrawn.${plural(sources.notDrawn)}`, { count: sources.notDrawn })
      : null,
  ].filter((part): part is string => part !== null);

  return (
    <div className="chart-foot" data-testid="chart-foot">
      {legend.length > 1 && (
        <span className="chart-legend" data-testid="chart-legend">
          {legend.map((item) => (
            <span key={`${item.ink}-${item.name}`} className="chart-legend-item">
              <i className={`chart-swatch chart-fill-${item.ink}`} aria-hidden="true" />
              {item.name}
            </span>
          ))}
        </span>
      )}
      {said && (
        <span className="chart-source" data-testid="chart-source">
          {check && <ChartCheck check={check} label={said} size="md" />}
          <span>{[said, ...more].join(" · ")}</span>
        </span>
      )}
      {!said && more.length > 0 && <span className="chart-source">{more.join(" · ")}</span>}
      {cutFrom !== null && (
        <span className="chart-source" data-testid="chart-cut">
          {t("chart.cut", { rows, total: cutFrom, max: MAX_CHART_VALUES })}
        </span>
      )}
      {onUpdate && (
        <span className="chart-source" data-testid="chart-changed">
          {t("chart.tableChanged")} ·{" "}
          <button
            type="button"
            data-chart-ui=""
            data-testid="chart-update"
            className="chart-link"
            onClick={onUpdate}
          >
            {t("chart.update")}
          </button>
        </span>
      )}
    </div>
  );
}
