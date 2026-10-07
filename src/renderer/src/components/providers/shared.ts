import type { ChatProvider, ChatReadiness, ProviderErrorKind } from "../../../../core/api";
import type { MessageKey, MessageParams } from "../../../../shared/i18n";

type Translate = (key: MessageKey, params?: MessageParams) => string;

/** What a failed connection test means, in words. The ChatGPT plan's own kinds live under `codex.`. */
export function testErrorKey(kind: ProviderErrorKind): MessageKey {
  switch (kind) {
    case "not-signed-in":
    case "plan-limit":
    case "blocked":
      return `codex.test.${kind}`;
    default:
      return `providers.test.${kind}`;
  }
}

/** What to configure before Questions can be asked, in words. */
export function readinessKey(
  reason: Extract<ChatReadiness, { ready: false }>["reason"],
): MessageKey {
  return reason === "sign-in-required"
    ? "codex.readiness.sign-in-required"
    : `providers.readiness.${reason}`;
}

/** "OpenAI","OpenAI-compatible server · api.deepseek.com", "Ollama · on this computer". */
export function providerLabel(provider: ChatProvider, t: Translate): string {
  if (provider.kind === "chatgpt") return t("codex.provider.name");
  const kind = t(`providers.kind.${provider.kind}`);
  if (provider.kind === "openai" || provider.kind === "anthropic" || provider.kind === "google") {
    return kind;
  }
  return `${kind} · ${provider.service?.name ?? t("providers.settings.local")}`;
}

/** Who receives the data, for messages like "You chose not to send data to {service}". */
export const serviceName = (provider: ChatProvider, t: Translate) =>
  provider.service?.name ?? providerLabel(provider, t);
