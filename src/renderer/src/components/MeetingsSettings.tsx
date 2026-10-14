import { isMacOS } from "@tiptap/core";
import { useId } from "react";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import {
  hintClass,
  rowTextClass,
  rowTitleClass,
  ruledListClass,
  ruledRow,
  ruledRowClass,
  sectionTitleClass,
} from "./ui";

/**
 * Settings → Meetings → Recording: the switch that turns recording on (it is
 * new, and off until transcripts and speakers come), what is recorded, and
 * whether a recording asks to read the shared screen (ADR-0024).
 */
export function MeetingsSettings() {
  const t = useT();
  const device = useAppStore((state) => state.settings?.device);
  const canRecord = useAppStore((state) => state.settings?.meetingAudio);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const enabledId = useId();
  const whatId = useId();
  const screenId = useId();
  if (!device || !canRecord) return null;
  const microphoneOnly = canRecord !== "both";
  const both = device.meetingSystemAudio && !microphoneOnly;

  return (
    <section data-testid="meetings-settings" className="flex flex-col">
      <h4 className={`mb-2 ${sectionTitleClass}`}>{t("meetings.settings.title")}</h4>
      <div className={ruledListClass}>
        <div className={ruledRowClass}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={enabledId} className={rowTitleClass}>
              {t("meetings.settings.enabled")}
            </label>
            <p id={`${enabledId}-description`} className={rowTextClass}>
              {t(
                canRecord === "none"
                  ? "meetings.settings.cantRecord"
                  : "meetings.settings.enabledBody",
              )}
            </p>
          </div>
          <input
            id={enabledId}
            type="checkbox"
            role="switch"
            aria-checked={device.meetingsEnabled}
            aria-describedby={`${enabledId}-description`}
            data-testid="meetings-enabled"
            checked={device.meetingsEnabled}
            disabled={canRecord === "none"}
            onChange={(event) =>
              void updateSettings({ device: { meetingsEnabled: event.target.checked } })
            }
            className="switch"
          />
        </div>
        <div className={ruledRow("center")}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={whatId} className={rowTitleClass}>
              {t("meetings.settings.what")}
            </label>
            <p className={rowTextClass}>
              {t(
                microphoneOnly
                  ? "meetings.line.systemUnavailable"
                  : both
                    ? "meetings.settings.whatBoth"
                    : "meetings.settings.whatMe",
              )}
            </p>
          </div>
          <select
            id={whatId}
            data-testid="meetings-what"
            value={both ? "both" : "me"}
            disabled={microphoneOnly}
            onChange={(event) =>
              void updateSettings({ device: { meetingSystemAudio: event.target.value === "both" } })
            }
            className="h-8 rounded-sm border border-rule-strong bg-sheet px-2 text-[13px] leading-5 text-ink focus:border-accent disabled:opacity-60"
          >
            <option value="both">{t("meetings.audio.both")}</option>
            <option value="me">{t("meetings.audio.me")}</option>
          </select>
        </div>
        <div className={ruledRowClass}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={screenId} className={rowTitleClass}>
              {t("meetings.settings.screen")}
            </label>
            <p id={`${screenId}-description`} className={rowTextClass}>
              {t("meetings.settings.screenBody")}
            </p>
          </div>
          <input
            id={screenId}
            type="checkbox"
            role="switch"
            aria-checked={device.meetingScreenAsk}
            aria-describedby={`${screenId}-description`}
            data-testid="meetings-screen-ask"
            checked={device.meetingScreenAsk}
            disabled={canRecord === "none"}
            onChange={(event) =>
              void updateSettings({ device: { meetingScreenAsk: event.target.checked } })
            }
            className="switch"
          />
        </div>
      </div>
      <p className={hintClass}>
        {t(isMacOS() ? "meetings.settings.kept.mac" : "meetings.settings.kept.other")}
      </p>
    </section>
  );
}
