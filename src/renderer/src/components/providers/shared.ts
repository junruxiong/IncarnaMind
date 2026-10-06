import type { ChatProvider } from "../../../../core/api";
import type { MessageKey, MessageParams } from "../../../../shared/i18n";

type Translate = (key: MessageKey, params?: MessageParams) => string;

export const buttonClass =
  "rounded-[9px] border border-gray-300 px-4 py-2 text-sm hover:bg-gray-100 disabled:cursor-default disabled:opacity-50 disabled:hover:bg-transparent";

export const primaryButtonClass =
  "rounded-[9px] bg-gray-800 px-4 py-2 text-sm text-white hover:bg-gray-700 disabled:cursor-default disabled:opacity-50 disabled:hover:bg-gray-800";

export const inputClass =
  "mt-1 w-full rounded-[9px] border border-gray-300 px-3 py-2 text-sm outline-none focus:border-gray-500";

/** "OpenAI", "OpenAI-compatible server · api.deepseek.com", "Ollama · on this computer". */
export function providerLabel(provider: ChatProvider, t: Translate): string {
  const kind = t(`providers.kind.${provider.kind}`);
  if (provider.kind === "openai" || provider.kind === "anthropic" || provider.kind === "google") {
    return kind;
  }
  return `${kind} · ${provider.service?.name ?? t("providers.settings.local")}`;
}

/** Who receives the data, for messages like "You chose not to send data to {service}". */
export const serviceName = (provider: ChatProvider, t: Translate) =>
  provider.service?.name ?? providerLabel(provider, t);
