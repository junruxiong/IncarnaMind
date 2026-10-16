import type { MessageKey } from "../../../../shared/i18n";
import { useComposer } from "../../composer";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { settingsPageFor } from "../providers/ChatReadinessNotice";

const STARTERS: readonly MessageKey[] = [
  "meeting.ask.recap",
  "meeting.ask.next",
  "meeting.ask.actions",
];

/**
 * A new conversation during a call (#144, #839): quick asks that follow the call, each asked at
 * once from the composer, as if typed there. With no AI model ready, one line and Set up instead.
 */
export function MeetingStarters() {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const openSettings = useAppStore((state) => state.openSettings);
  if (readiness && !readiness.ready) {
    return (
      <p data-testid="meeting-set-up" className="assistant-empty">
        {t("meetings.help.setUp")}
        <span aria-hidden="true"> · </span>
        <button
          type="button"
          onClick={() => openSettings(settingsPageFor(readiness))}
          className="font-semibold text-accent hover:underline"
        >
          {t("providers.readiness.setUp")}
        </button>
      </p>
    );
  }
  return (
    <ul
      aria-label={t("meeting.ask.label")}
      data-testid="meeting-starters"
      className="assistant-starters"
    >
      {STARTERS.map((starter) => (
        <li key={starter}>
          <button
            type="button"
            data-testid="meeting-starter"
            onClick={() => useComposer.getState().requestAsk(t(starter))}
            className="assistant-starter"
          >
            {t(starter)}
          </button>
        </li>
      ))}
    </ul>
  );
}
