import type { TranscriptionServiceStatus } from "../../../../core/api";
import { useGap, useLanguage, useT } from "../../i18n";

/**
 * The line that makes choosing a cloud route the consent (ADR-0022): where meeting audio goes,
 * what the service's own agreement says about it where we have checked one, and that the
 * others should be told.
 */
export function RouteData({
  service,
  className,
  testId,
}: {
  service: TranscriptionServiceStatus;
  className: string;
  testId?: string;
}) {
  const t = useT();
  const gap = useGap();
  const words = useLanguage() === "zh-CN" ? "zh" : "en";
  const sentence = t("meetings.route.data", { provider: service.recipient[words] });
  const tell = t("meetings.route.tell");
  const link = `${t("meetings.route.agreement")} ↗`;
  const summary = service.dataUse?.summary[words];
  return (
    <p data-testid={testId} className={className}>
      {sentence}
      {summary && (
        <>
          {gap(sentence, summary)}
          {summary}
          {gap(summary, link)}
          <a
            href={service.dataUse?.url[words]}
            target="_blank"
            rel="noreferrer"
            className="underline decoration-rule-strong underline-offset-2 hover:text-ink"
          >
            {link}
          </a>
        </>
      )}
      {gap(summary ? link : sentence, tell)}
      {tell}
    </p>
  );
}
