import { CHOSEN_PROVIDER_FLOWS, type ChosenProviderFlow } from "../../core/api";
import type { MessageKey } from "../../shared/i18n";
import type { Translate } from "../../shared/locations";

const FLOW_LINES = {
  chat: "providers.sends.chat",
  checks: "providers.sends.checks",
  suggestions: "providers.sends.suggestions",
  edits: "providers.sends.edits",
  tagging: "providers.sends.tagging",
  classification: "providers.sends.tagging",
} as const satisfies Record<ChosenProviderFlow, MessageKey>;

const SHORT_LINES = {
  chat: "providers.sendsShort.chat",
  checks: "providers.sendsShort.checks",
  suggestions: "providers.sendsShort.suggestions",
  edits: "providers.sendsShort.edits",
  tagging: "providers.sendsShort.tagging",
  classification: "providers.sendsShort.tagging",
} as const satisfies Record<ChosenProviderFlow, MessageKey>;

const linesOf = (lines: Record<ChosenProviderFlow, MessageKey>): MessageKey[] => [
  ...new Set(CHOSEN_PROVIDER_FLOWS.map((flow) => lines[flow])),
];

/** The items of the list above the button: what every flow that choosing the provider allows sends. */
export const chosenFlowLines = (): MessageKey[] => linesOf(FLOW_LINES);

/** The same, as one phrase for a line with no room for the list. */
export function chosenFlowSummary(t: Translate): string {
  const items = linesOf(SHORT_LINES).map((key) => t(key));
  return t("providers.sendsShort.last", {
    rest: items.slice(0, -1).join(t("providers.sendsShort.separator")),
    last: items.at(-1) ?? "",
  });
}
