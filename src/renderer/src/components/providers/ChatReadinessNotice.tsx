import type { ChatReadiness } from "../../../../core/api";
import { useT } from "../../i18n";
import { type SettingsPage, useAppStore } from "../../store";
import { readinessKey, serviceName } from "./shared";

/** Where to fix it: a declined data flow is allowed again on the Privacy page. */
export const settingsPageFor = (
  readiness: Extract<ChatReadiness, { ready: false }>,
): SettingsPage => (readiness.reason === "consent-declined" ? "privacy" : "general");

/** What to configure before Questions can be asked. */
export function ReadinessExplanation({
  readiness,
}: {
  readiness: Extract<ChatReadiness, { ready: false }>;
}) {
  const t = useT();
  const service = "provider" in readiness ? serviceName(readiness.provider, t) : "";
  return <>{t(readinessKey(readiness.reason), { service })}</>;
}

/**
 * Shown in a Mind while Questions can't be asked, with a way to fix it.
 * Answers (#29) disable asking with the same explanation.
 */
export function ChatReadinessNotice() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const openSettings = useAppStore((state) => state.openSettings);
  if (!readiness || readiness.ready) return null;
  return (
    <p
      data-testid="chat-readiness"
      className="mx-3 flex items-center gap-3 rounded-[9px] bg-gray-50 px-3 py-2 text-sm text-gray-600"
    >
      <span className="flex-1">
        <ReadinessExplanation readiness={readiness} />
      </span>
      <button
        type="button"
        onClick={() => openSettings(settingsPageFor(readiness))}
        className="shrink-0 rounded-[9px] px-2 py-1 text-gray-700 hover:bg-gray-200"
      >
        {t("providers.readiness.setUp")}
      </button>
    </p>
  );
}
