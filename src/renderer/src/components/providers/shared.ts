import { useRef, useState } from "react";
import type { ChatProvider, ChatReadiness, ProviderErrorKind } from "../../../../core/api";
import { catalogProviderOfKind } from "../../../../core/providers/catalog/providers";
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
  const catalog = catalogProviderOfKind(provider.kind);
  if (catalog && catalog.endpoints.length > 0) {
    // A hosted provider: its name, and its region when it has more than one.
    if (catalog.endpoints.length === 1) return kind;
    return provider.endpoint === "cn" || provider.endpoint === "intl"
      ? `${kind} · ${t(`providers.region.${provider.endpoint}`)}`
      : kind;
  }
  return `${kind} · ${provider.service?.name ?? t("providers.settings.local")}`;
}

/** Who receives the data, for messages like "You chose not to send data to {service}". */
export const serviceName = (provider: ChatProvider, t: Translate) =>
  provider.service?.name ?? providerLabel(provider, t);

/** The host of a server address, or null while it isn't one yet. */
function hostOf(baseUrl: string): string | null {
  try {
    return new URL(baseUrl.trim()).host || null;
  } catch {
    return null;
  }
}

/**
 * An API key field that keeps a key only for the provider and server it was
 * typed for, so it is never sent to another: choosing another provider clears
 * it, and so does leaving the address field pointing at another host.
 */
export function useProviderKey() {
  const [apiKey, setApiKey] = useState("");
  /** The host the address field held when the key was typed, if it held one. */
  const typedFor = useRef<string | null>(null);
  return {
    apiKey,
    /** The key as typed, with the server address the form holds now. */
    type(value: string, baseUrl = "") {
      setApiKey(value);
      typedFor.current = hostOf(baseUrl);
    },
    /** Another provider was chosen, or the key was saved. */
    clear() {
      setApiKey("");
      typedFor.current = null;
    },
    /** The address field was left: a key typed for another host goes. */
    leftAddress(baseUrl: string) {
      const host = hostOf(baseUrl);
      if (typedFor.current === null) typedFor.current = host;
      else if (host !== typedFor.current) {
        setApiKey("");
        typedFor.current = null;
      }
    },
  };
}
