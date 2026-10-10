import type { ChatReadiness } from "../../../../core/api";
import { useT } from "../../i18n";
import type { SettingsPage } from "../../settingsPages";
import { useAppStore } from "../../store";
import { buttonStyle } from "../ui";
import { readinessKey, serviceName } from "./shared";

/** Where to fix it: a declined data flow is allowed again on the Privacy page, the rest under Models. */
export const settingsPageFor = (
  readiness: Extract<ChatReadiness, { ready: false }>,
): SettingsPage => (readiness.reason === "consent-declined" ? "privacy" : "models");

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
      // The box reaches 12px into the margins, so its text keeps the Mind's text edge.
      className="-mx-3 flex items-center gap-3 rounded-lg bg-frame py-1.5 pr-1.5 pl-3 text-[13px] leading-5 text-ink-secondary"
    >
      <span className="flex-1">
        <ReadinessExplanation readiness={readiness} />
      </span>
      <button
        type="button"
        onClick={() => openSettings(settingsPageFor(readiness))}
        className={buttonStyle("secondary", "sm")}
      >
        {t("providers.readiness.setUp")}
      </button>
    </p>
  );
}
