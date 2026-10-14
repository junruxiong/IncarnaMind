import { isMacOS } from "@tiptap/core";
import { useGap, useT } from "../i18n";
import { useMeetings } from "../meetings";
import {
  buttonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  primaryButtonClass,
} from "./ui";
import { useModal } from "./useModal";

/**
 * The one-time recording notice: the first "Start recording" on this device
 * says what is recorded and where it stays, and reminds the User to tell the
 * others. Starting from it records; Esc or Cancel doesn't.
 */
export function MeetingNoticeDialog() {
  const t = useT();
  const gap = useGap();
  const waiting = useMeetings((state) => state.noticeFor);
  const { acceptNotice, cancelNotice } = useMeetings.getState();
  const dialog = useModal(waiting !== null);
  const recording = t(
    waiting?.systemAudio === false ? "meetings.notice.microphone" : "meetings.notice.both",
  );
  const kept = t(isMacOS() ? "meetings.notice.kept.mac" : "meetings.notice.kept.other");

  return (
    <dialog
      ref={dialog}
      data-testid="meeting-notice"
      aria-labelledby="meeting-notice-title"
      aria-describedby="meeting-notice-body"
      onClose={cancelNotice}
      className={`${dialogClass} w-[28rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id="meeting-notice-title" className={dialogTitleClass}>
          {t("meetings.notice.title")}
        </h2>
        <div id="meeting-notice-body" className="flex flex-col gap-2">
          <p className={dialogTextClass}>
            {recording}
            {gap(recording, kept)}
            {kept}
          </p>
          <p className={dialogTextClass}>{t("meetings.notice.tell")}</p>
        </div>
        <div className={dialogActionsClass}>
          <button type="button" onClick={cancelNotice} className={buttonClass}>
            {t("meetings.notice.cancel")}
          </button>
          <button
            type="button"
            data-testid="meeting-notice-start"
            onClick={() => void acceptNotice()}
            className={primaryButtonClass}
          >
            {t("meetings.start.record")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
