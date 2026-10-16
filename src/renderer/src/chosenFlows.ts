import { CHOSEN_PROVIDER_FLOWS, type ChosenProviderFlow } from "../../core/api";
import type { MessageKey } from "../../shared/i18n";

/** The words for what each flow sends; a flow added to `CHOSEN_PROVIDER_FLOWS` fails the type-check until it has some. */
const FLOW_LINES = {
  chat: "providers.sends.chat",
  checks: "providers.sends.checks",
  suggestions: "providers.sends.suggestions",
  tagging: "providers.sends.tagging",
  classification: "providers.sends.tagging",
  edits: "providers.sends.edits",
} as const satisfies Record<ChosenProviderFlow, MessageKey>;

export const chosenFlowLines = (): MessageKey[] => [
  ...new Set(CHOSEN_PROVIDER_FLOWS.map((flow) => FLOW_LINES[flow])),
];
