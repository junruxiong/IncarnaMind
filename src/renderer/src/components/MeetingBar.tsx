import { isMacOS } from "@tiptap/core";
import { type ReactNode, useEffect } from "react";
import type { AudioSource, Meeting, MeetingSourceState } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";
import { useMeetings, useMeetingTime } from "../meetings";
import { clockTime, durationLabel } from "../meetingTime";
import { PauseLineIcon, ResumeLineIcon, StopLineIcon } from "./lineIcons";
import { type ProblemAction, ProblemLine } from "./ProblemLine";

/** Pause and Resume (DESIGN.md, Names and shortcuts). */
export const PAUSE_SHORTCUT = isMacOS() ? "⌘⇧P" : "Ctrl+Shift+P";

const isPauseKey = (event: KeyboardEvent) =>
  event.code === "KeyP" &&
  event.shiftKey &&
  !event.altKey &&
  (isMacOS() ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey);

/**
 * The meeting bar, while the Mind's Meeting records or is paused: at the top
 * of the Mind, under its title, and kept in view as the notes scroll (the
 * owner's direction of 2026-10-11, in place of DESIGN.md's bar in the
 * composer). Its controls, then a line for each thing about the recording
 * the User should know, then whatever the Mind adds (its audio, once paused).
 */
export function MeetingBar({ meeting, children }: { meeting: Meeting; children?: ReactNode }) {
  const t = useT();
  return (
    <section
      data-testid="meeting-bar"
      data-status={meeting.status}
      aria-label={t("meetings.bar.label")}
      className="meeting-bar"
    >
      <MeetingControls meeting={meeting} />
      <MeetingLines meeting={meeting} />
      {children}
    </section>
  );
}

/**
 * The live-recording dot (red, the one red outside Citation checks; grey
 * while paused), "Recording 24:16" (or "Paused"), Pause or Resume (⌘⇧P) and Stop.
 */
function MeetingControls({ meeting }: { meeting: Meeting }) {
  const t = useT();
  const time = clockTime(useMeetingTime(meeting));
  const { pause, resume, stop } = useMeetings.getState();
  const recording = meeting.status === "recording";
  const toggle = () => void (recording ? pause(meeting.id) : resume(meeting.id));

  // ⌘⇧P anywhere in the window but a dialog, as long as the bar is shown.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || !isPauseKey(event)) return;
      if (document.querySelector("dialog[open]")) return;
      event.preventDefault();
      toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div data-testid="meeting-controls" className="meeting-controls">
      <span className="meeting-state">
        <span
          aria-hidden="true"
          data-testid="meeting-dot"
          data-status={meeting.status}
          className="recording-dot"
        />
        <span className="font-semibold">
          {t(recording ? "meetings.bar.recording" : "meetings.bar.paused")}
        </span>
        <span role="timer" data-testid="meeting-time" className="meeting-time">
          {time}
        </span>
      </span>
      <button
        type="button"
        data-testid={recording ? "meeting-pause" : "meeting-resume"}
        aria-label={t(recording ? "meetings.bar.pause" : "meetings.bar.resume", {
          shortcut: PAUSE_SHORTCUT,
        })}
        title={t(recording ? "meetings.bar.pause" : "meetings.bar.resume", {
          shortcut: PAUSE_SHORTCUT,
        })}
        onClick={toggle}
        className="meeting-button meeting-button--icon"
      >
        {recording ? (
          <PauseLineIcon className="size-3.5" />
        ) : (
          <ResumeLineIcon className="size-3.5" />
        )}
      </button>
      <button
        type="button"
        data-testid="meeting-stop"
        aria-label={t("meetings.bar.stopLabel")}
        title={t("meetings.bar.stopLabel")}
        onClick={() => void stop(meeting.id)}
        className="meeting-button"
      >
        <StopLineIcon className="size-3.5" />
        {t("meetings.bar.stop")}
      </button>
    </div>
  );
}

/** What the User should know about the recording, a line each: its health, a cut, a gap. */
function MeetingLines({ meeting }: { meeting: Meeting }) {
  const t = useT();
  const problem = useMeetings((state) => state.problems[meeting.id]);
  const gapsSeen = useMeetings((state) => state.gapsSeen[meeting.id] ?? 0);
  const { openSettings, resume, dismissGaps } = useMeetings.getState();
  const settings = (source: AudioSource): ProblemAction => ({
    label: t(isMacOS() ? "meetings.line.openSettings.mac" : "meetings.line.openSettings.other"),
    testId: `meeting-settings-${source}`,
    onClick: () => void openSettings(meeting.id, source),
  });
  const lines: { key: string; text: string; action?: ProblemAction }[] = [];

  if (problem) lines.push({ key: "problem", text: problem });
  if (meeting.saveError) {
    lines.push({
      key: "save",
      text: t("meetings.line.saveFailed", { message: meeting.saveError }),
    });
  }
  if (meeting.status === "paused" && meeting.pausedBy === "interrupted") {
    lines.push({
      key: "interrupted",
      text: t("meetings.line.interrupted", { time: clockTime(meeting.recordedMs) }),
      action: {
        label: t("meetings.line.resume"),
        testId: "meeting-line-resume",
        onClick: () => void resume(meeting.id),
      },
    });
  }
  const microphone = MICROPHONE_LINES[meeting.sources.microphone];
  if (microphone) {
    lines.push({ key: "microphone", text: t(microphone), action: settings("microphone") });
  }
  const system = meeting.systemAudio ? SYSTEM_LINES[meeting.sources.system] : undefined;
  if (system) {
    lines.push({
      key: "system",
      text: t(system),
      ...(meeting.sources.system !== "unavailable" && { action: settings("system") }),
    });
  }
  const gap = meeting.gaps.length > gapsSeen ? meeting.gaps.at(-1) : undefined;
  if (gap && meeting.status === "recording") {
    lines.push({
      key: "gap",
      text: t(gap.reason === "sleep" ? "meetings.line.gapSleep" : "meetings.line.gapClosed", {
        missed: durationLabel(gap.missedMs, t),
        time: clockTime(gap.atMs),
      }),
      action: { label: t("error.dismiss"), onClick: () => dismissGaps(meeting.id) },
    });
  }

  return (
    <>
      {lines.map((line) => (
        <ProblemLine
          key={line.key}
          testId={`meeting-line-${line.key}`}
          role={line.key === "gap" || line.key === "interrupted" ? "status" : "alert"}
          className="meeting-line"
          action={line.action}
        >
          {line.text}
        </ProblemLine>
      ))}
    </>
  );
}

const MICROPHONE_LINES: Partial<Record<MeetingSourceState, MessageKey>> = {
  denied: "meetings.line.microphoneDenied",
  unavailable: "meetings.line.microphoneUnavailable",
  ended: "meetings.line.microphoneEnded",
};

const SYSTEM_LINES: Partial<Record<MeetingSourceState, MessageKey>> = {
  denied: "meetings.line.systemDenied",
  silent: "meetings.line.systemSilent",
  ended: "meetings.line.systemEnded",
  unavailable: "meetings.line.systemUnavailable",
};
