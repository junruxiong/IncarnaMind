import { useComingUp } from "../../comingUp";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";

export function ComingUp() {
  const t = useT();
  const meetings = useComingUp();
  // Meetings' rows come with calendars, in #190.
  if (meetings.length > 0) return null;
  return (
    <section data-testid="home-coming-up" className="mb-7">
      <h2 className="mb-1.5 text-label font-semibold text-ink-meta">{t("home.comingUp")}</h2>
      <p data-testid="home-no-meetings" className="py-1.5 text-ui text-ink-meta">
        {t("home.noMeetings")}
        <span aria-hidden="true"> · </span>
        <button
          type="button"
          data-testid="home-connect-calendar"
          onClick={() => useAppStore.getState().openSettings("meetings")}
          className="rounded-sm text-accent hover:underline"
        >
          {t("home.connectCalendar")}
        </button>
      </p>
    </section>
  );
}
