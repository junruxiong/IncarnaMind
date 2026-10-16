import { useId, useState } from "react";
import { chosenFlowLines } from "../../chosenFlows";
import { useGap, useT } from "../../i18n";

/**
 * Where a service is chosen, what goes to it, in one line (DESIGN.md › Words on screen); Details
 * opens the full list of flows choosing it allows, which choosing consents to (ADR-0022).
 */
export function SendsLine({
  provider,
  testId = "provider-sends",
}: {
  provider: string;
  testId?: string;
}) {
  const t = useT();
  const gap = useGap();
  const [open, setOpen] = useState(false);
  const list = useId();
  const line = t("providers.sendsLine", { provider });
  return (
    <div data-testid={testId} className="text-[12px] leading-5 text-ink-meta">
      <p>
        {line}
        {gap(line, "·")}
        <span className="whitespace-nowrap">
          <span aria-hidden="true">· </span>
          <button
            type="button"
            data-testid={`${testId}-details`}
            aria-expanded={open}
            aria-controls={list}
            onClick={() => setOpen(!open)}
            className="font-semibold text-accent underline-offset-2 hover:underline"
          >
            {t("providers.sendsDetails")}
          </button>
        </span>
      </p>
      {open && (
        <ul id={list} data-testid={`${testId}-list`} className="list-disc pl-4">
          {chosenFlowLines().map((item) => (
            <li key={item}>{t(item)}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
