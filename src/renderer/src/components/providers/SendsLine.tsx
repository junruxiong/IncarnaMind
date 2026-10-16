import { chosenFlowLines } from "../../chosenFlows";
import { useT } from "../../i18n";
import { DetailsLine } from "../DetailsLine";

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
  return (
    <DetailsLine
      testId={testId}
      className="text-[12px] leading-5 text-ink-meta"
      details={
        <ul className="list-disc pl-4">
          {chosenFlowLines().map((item) => (
            <li key={item}>{t(item)}</li>
          ))}
        </ul>
      }
    >
      {t("providers.sendsLine", { provider })}
    </DetailsLine>
  );
}
