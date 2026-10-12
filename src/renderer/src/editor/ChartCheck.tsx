import type { CitationCheck } from "../../../core/api";
import { CheckMarkIcon } from "./icons";

/**
 * A chart value's check, as a small filled mark (#436): found in `success`,
 * a quote not found in `warning` on its wash, contradicted outlined in
 * `error`, can't check on `chip`. With no Citation, a quiet grey ring. Its
 * meaning is in `label`, for the tooltip and a screen reader.
 */
export function ChartCheck({
  check,
  label,
  size = "lg",
}: {
  /** Null: the value has no Citation. */
  check: CitationCheck | null;
  label: string;
  /** "sm" in a table cell, "md" under the chart, "lg" on a value's card. */
  size?: "sm" | "md" | "lg";
}) {
  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      data-testid="chart-check"
      data-check={check ?? "none"}
      className={`chart-check chart-check--${size} chart-check--${check ?? "none"}`}
    >
      {check && <CheckMarkIcon check={check} className="chart-check-icon" />}
    </span>
  );
}
