import { isMacOS } from "@tiptap/core";
import { useEffect, useId, useState } from "react";
import type { Meeting } from "../../../core/api";
import { formatDateTime } from "../../../shared/dates";
import { useGap, useLanguage, useT } from "../i18n";
import { isLive, useMeetingOf, useMeetings } from "../meetings";
import { clockTime } from "../meetingTime";
import { useAppStore } from "../store";
import { CaptureDetails, CaptureDetailsLine, CaptureProblem } from "./CaptureDetails";
import { CloseLineIcon, RecordingLineIcon } from "./lineIcons";
import { MeetingBar } from "./MeetingBar";
import { ProblemLine } from "./ProblemLine";
import { TranscriptLines } from "./TranscriptLines";
import { buttonStyle, fieldLabelClass, inputClass } from "./ui";

/**
 * A Meeting's place in its Mind, under the title: the start block while it is
 * ready, the meeting bar while it records or is paused, and a line about its
 * audio, with "Delete audio now", once it is paused or stopped. Its recording
 * details (#161) open from the bar, or from the line under a stopped Meeting.
 */
export function MeetingInMind({ mindId }: { mindId: string }) {
  const meeting = useMeetingOf(mindId);
  if (!meeting) return null;
  if (meeting.status === "ready") return <MeetingStart meeting={meeting} />;
  if (isLive(meeting)) {
    return (
      <MeetingBar meeting={meeting}>
        <CaptureProblem meeting={meeting} className="meeting-line" />
        {meeting.status === "paused" && <MeetingAudioLine meeting={meeting} />}
        <CaptureDetails meeting={meeting} />
      </MeetingBar>
    );
  }
  return (
    <div className="mb-4">
      <TranscriptLines meeting={meeting} className="" />
      <CaptureProblem meeting={meeting} />
      <MeetingAudioLine meeting={meeting} />
      <CaptureDetailsLine meeting={meeting} />
      <CaptureDetails meeting={meeting} />
    </div>
  );
}

/**
 * Starting a meeting, inline in its new Mind (the canvas board
 * Final-03-MeetingStart): what is recorded, "Start recording", and the
 * reminder to tell everyone. ✕ or Esc keeps the Mind as a note.
 */
function MeetingStart({ meeting }: { meeting: Meeting }) {
  const t = useT();
  const audioId = useId();
  const canRecord = useAppStore((state) => state.settings?.meetingAudio ?? "none");
  const [systemAudio, setSystemAudio] = useState(meeting.systemAudio);
  const [copied, setCopied] = useState(false);
  const problem = useMeetings((state) => state.problems[meeting.id]);
  const { start, discard } = useMeetings.getState();
  const microphoneOnly = canRecord === "microphone";

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timer);
  }, [copied]);

  const copyNotice = async () => {
    try {
      await navigator.clipboard.writeText(t("meetings.start.noticeText"));
      setCopied(true);
    } catch (error) {
      useAppStore.getState().reportError(error);
    }
  };

  return (
    <section
      data-testid="meeting-start"
      aria-label={t("meetings.start.label")}
      className="meeting-start"
      onKeyDown={(event) => {
        if (event.key !== "Escape" || event.defaultPrevented) return;
        event.preventDefault();
        void discard(meeting.id);
      }}
    >
      <div className="flex items-start gap-3">
        <div className="w-56 max-w-full min-w-0">
          <label htmlFor={audioId} className={fieldLabelClass}>
            {t("meetings.start.audio")}
          </label>
          <select
            id={audioId}
            data-testid="meeting-start-audio"
            value={systemAudio && !microphoneOnly ? "both" : "me"}
            disabled={microphoneOnly}
            onChange={(event) => setSystemAudio(event.target.value === "both")}
            className={inputClass}
          >
            <option value="both">{t("meetings.audio.both")}</option>
            <option value="me">{t("meetings.audio.me")}</option>
          </select>
        </div>
        <button
          type="button"
          data-testid="meeting-start-dismiss"
          aria-label={t("meetings.start.dismiss")}
          title={t("meetings.start.dismiss")}
          onClick={() => void discard(meeting.id)}
          className="ml-auto inline-flex size-7 shrink-0 items-center justify-center rounded-md text-ink-meta hover:bg-chip hover:text-ink"
        >
          <CloseLineIcon className="size-4" />
        </button>
      </div>
      {microphoneOnly && (
        <p data-testid="meeting-start-microphone-only" className="meeting-start-note">
          {t("meetings.line.systemUnavailable")}
        </p>
      )}
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
        <button
          type="button"
          data-testid="meeting-start-record"
          onClick={() => void start(meeting.id, systemAudio && !microphoneOnly)}
          className={buttonStyle("primary", "lg")}
        >
          <RecordingLineIcon className="size-4" />
          {t("meetings.start.record")}
        </button>
        <div className="min-w-0 flex-1 basis-64 text-[13px] leading-5">
          <p className="text-ink-strong">
            {t("meetings.start.tell")}
            <span aria-hidden="true"> · </span>
            <button
              type="button"
              data-testid="meeting-start-copy-notice"
              onClick={() => void copyNotice()}
              className="font-semibold text-accent underline-offset-2 hover:underline"
            >
              {t("meetings.start.copyNotice")}
            </button>
            {copied && (
              <span role="status" className="ml-2 text-ink-meta">
                {t("meetings.start.copied")}
              </span>
            )}
          </p>
          <p className="text-[12px] leading-[18px] text-ink-meta">
            {t(isMacOS() ? "meetings.start.kept.mac" : "meetings.start.kept.other")}
          </p>
        </div>
      </div>
      {problem && <ProblemLine testId="meeting-start-problem">{problem}</ProblemLine>}
    </section>
  );
}

/**
 * How long was recorded, how long the audio stays on this computer, and
 * "Delete audio now", which asks first, on the same line: it can't be undone.
 */
function MeetingAudioLine({ meeting, className = "" }: { meeting: Meeting; className?: string }) {
  const t = useT();
  const language = useLanguage();
  const gap = useGap();
  const { deleteAudio } = useMeetings.getState();
  const [confirming, setConfirming] = useState(false);
  const { audio } = meeting;
  if (confirming && audio.kept) {
    return (
      <p
        role="alert"
        data-testid="meeting-audio-confirm"
        className={`font-sans text-[13px] leading-5 break-words text-ink-meta ${className}`}
        onKeyDown={(event) => {
          if (event.key !== "Escape") return;
          event.preventDefault();
          setConfirming(false);
        }}
      >
        {t("meetings.audio.confirm")}
        {gap(t("meetings.audio.confirm"), t("meetings.audio.deleteConfirm"))}
        <button
          type="button"
          // biome-ignore lint/a11y/noAutofocus: the User just asked to delete: the choice is here.
          autoFocus
          data-testid="meeting-audio-delete-confirm"
          onClick={() => {
            setConfirming(false);
            void deleteAudio(meeting.id);
          }}
          className="rounded-sm font-semibold text-ink underline underline-offset-2"
        >
          {t("meetings.audio.deleteConfirm")}
        </button>
        <span aria-hidden="true"> · </span>
        <button
          type="button"
          onClick={() => setConfirming(false)}
          className="rounded-sm font-semibold text-accent underline-offset-2 hover:underline"
        >
          {t("meetings.audio.keep")}
        </button>
      </p>
    );
  }
  const kept =
    audio.expiresAt === null
      ? t("meetings.audio.keptWhile")
      : t("meetings.audio.keptUntil", {
          time: formatDateTime(audio.expiresAt, language),
        });
  return (
    <ProblemLine
      role="status"
      testId="meeting-audio"
      className={className}
      action={
        audio.kept
          ? {
              label: t("meetings.audio.delete"),
              testId: "meeting-audio-delete",
              onClick: () => setConfirming(true),
            }
          : undefined
      }
    >
      {meeting.status === "stopped" && (
        <>
          {t("meetings.audio.recorded", { time: clockTime(meeting.recordedMs) })}
          <span aria-hidden="true"> · </span>
        </>
      )}
      {audio.kept ? kept : audio.deletedAt ? t("meetings.audio.deleted") : t("meetings.audio.none")}
    </ProblemLine>
  );
}
