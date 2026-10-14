import type { MessageKey } from "../../../../shared/i18n";
import { useComposer } from "../../composer";
import { useT } from "../../i18n";

/** Each quick ask: what it shows, and the Question it asks. */
const STARTERS: readonly { label: MessageKey; question: MessageKey }[] = [
  { label: "meeting.ask.catchUp", question: "meeting.ask.catchUp" },
  { label: "meeting.ask.recap", question: "meeting.ask.recap" },
  { label: "meeting.ask.next", question: "meeting.ask.next" },
  { label: "meeting.ask.check", question: "meeting.ask.check.question" },
];

/**
 * A new conversation during a call (#144): quick asks about the call, each asked at once
 * from the composer, as if typed there.
 */
export function MeetingStarters() {
  const t = useT();
  return (
    <ul
      aria-label={t("meeting.ask.label")}
      data-testid="meeting-starters"
      className="assistant-starters"
    >
      {STARTERS.map((starter) => (
        <li key={starter.label}>
          <button
            type="button"
            data-testid="meeting-starter"
            onClick={() => useComposer.getState().requestAsk(t(starter.question))}
            className="assistant-starter"
          >
            {t(starter.label)}
          </button>
        </li>
      ))}
    </ul>
  );
}
