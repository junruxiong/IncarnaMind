import { useEffect, useId, useState } from "react";
import type { MessageKey } from "../../../shared/i18n";
import { core } from "../core";
import { useGap, useT } from "../i18n";
import { useAppStore } from "../store";
import { CloseLineIcon } from "./lineIcons";
import { buttonStyle } from "./ui";

/** Every event and field usage data can hold, for people to read. */
export const USAGE_DATA_DETAILS_URL =
  "https://github.com/IncarnaMind/IncarnaMind/blob/main/docs/privacy.md#usage-data";

/** What usage data looks like, in words. */
const EXAMPLES: readonly MessageKey[] = [
  "usageData.example.question",
  "usageData.example.answer",
  "usageData.example.documents",
];

/**
 * Usage data (DESIGN.md, First run), in a card at the bottom of the sidebar
 * where Get started was, shown only by a copy that can send it. A release
 * build asks once, when the first Answer to the User's own Question arrives,
 * and sends nothing unless the User chooses Send. A test build says from the
 * first run that it sends usage data, as testers agreed, with the switch to
 * turn it off and a button to hide the notice. Either way the choice lives in
 * Settings → Privacy.
 */
export function UsageDataCard() {
  const t = useT();
  const gap = useGap();
  const id = useId();
  const privacy = useAppStore((state) => state.privacy);
  const ownAnswered = useAppStore(
    (state) => state.ownAnswerArrived || state.settings?.device.gettingStarted.askedOwn === true,
  );
  const [what, setWhat] = useState(false);
  /** Gone as soon as the User answers, before the core confirms it. */
  const [answered, setAnswered] = useState(false);
  /** A test build's notice stays until hidden, even once the switch has been used. */
  const [kept, setKept] = useState(false);
  const usageData = privacy?.usageData;
  const tester = usageData?.testerBuild === true;
  const waiting = usageData?.available === true && !usageData.asked && !usageData.localMode;
  useEffect(() => {
    if (tester && waiting) setKept(true);
  }, [tester, waiting]);
  const shown =
    usageData?.available === true &&
    !usageData.localMode &&
    !answered &&
    (tester ? waiting || kept : waiting && ownAnswered);
  if (!usageData || !shown) return null;

  const answer = async (agree: boolean) => {
    setAnswered(true);
    try {
      await core.updatePrivacySettings({ usageData: agree });
    } catch (failure) {
      setAnswered(false);
      useAppStore.getState().reportError(failure);
    }
  };
  const setSwitch = async (on: boolean) => {
    try {
      await core.updatePrivacySettings({ usageData: on });
    } catch (failure) {
      useAppStore.getState().reportError(failure);
    }
  };
  const hide = async () => {
    setKept(false);
    await answer(usageData.enabled);
  };

  const title = t(tester ? "usageData.card.tester.title" : "usageData.card.title");
  return (
    <section
      aria-labelledby={`${id}-title`}
      data-testid="usage-data-card"
      data-tester={tester}
      className="mx-2 mb-2 flex shrink-0 flex-col gap-2 rounded-lg bg-sheet p-2.5 shadow-[inset_0_0_0_1px_var(--color-rule)]"
    >
      <div className="flex items-start justify-between gap-2">
        <h2 id={`${id}-title`} className="text-[13px] leading-5 font-semibold text-ink">
          {title}
        </h2>
        {tester && (
          <button
            type="button"
            data-testid="usage-data-hide"
            aria-label={t("usageData.card.tester.hide")}
            title={t("usageData.card.tester.hide")}
            onClick={() => void hide()}
            className="inline-flex size-[22px] shrink-0 items-center justify-center rounded-md text-ink-meta hover:bg-chip hover:text-ink focus-visible:outline-offset-0"
          >
            <CloseLineIcon strokeWidth={2.5} className="size-3" />
          </button>
        )}
      </div>
      <p className="text-[13px] leading-5 text-ink-secondary">
        {t(tester ? "usageData.card.tester.body" : "usageData.card.body")}
      </p>
      {tester ? (
        <label className="flex items-center justify-between gap-2 text-[13px] leading-5 font-semibold text-ink">
          {t("usageData.card.tester.switch")}
          <input
            type="checkbox"
            role="switch"
            data-testid="usage-data-switch"
            className="switch"
            checked={usageData.enabled}
            aria-checked={usageData.enabled}
            onChange={(event) => void setSwitch(event.target.checked)}
          />
        </label>
      ) : (
        <div className="flex gap-2">
          <button
            type="button"
            data-testid="usage-data-send"
            onClick={() => void answer(true)}
            className={`${buttonStyle("secondary", "sm")} flex-1`}
          >
            {t("usageData.card.send")}
          </button>
          <button
            type="button"
            data-testid="usage-data-decline"
            onClick={() => void answer(false)}
            className={`${buttonStyle("secondary", "sm")} flex-1`}
          >
            {t("usageData.card.decline")}
          </button>
        </div>
      )}
      <button
        type="button"
        data-testid="usage-data-what"
        aria-expanded={what}
        aria-controls={`${id}-what`}
        onClick={() => setWhat(!what)}
        className="self-start rounded-sm text-[13px] leading-5 text-accent underline-offset-2 hover:underline focus-visible:outline-offset-0"
      >
        {t("usageData.card.what")}
      </button>
      {what && (
        <div id={`${id}-what`} className="flex flex-col gap-1.5 text-[13px] leading-5">
          <p className="text-ink-secondary">{t("usageData.card.example")}</p>
          <ul
            data-testid="usage-data-examples"
            className="flex list-disc flex-col gap-1 pl-4 text-ink"
          >
            {EXAMPLES.map((key) => (
              <li key={key}>{t(key)}</li>
            ))}
          </ul>
          <p className="text-ink-meta">
            {t("usageData.card.change")}
            {gap(t("usageData.card.change"), t("usageData.card.details"))}
            <a
              href={USAGE_DATA_DETAILS_URL}
              target="_blank"
              rel="noreferrer"
              className="text-accent underline-offset-2 hover:text-accent-strong hover:underline"
            >
              {t("usageData.card.details")}
            </a>
          </p>
        </div>
      )}
    </section>
  );
}
