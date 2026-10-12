import type { Meeting } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";
import { formatBytes } from "../linkedFolders";
import { useMeetings } from "../meetings";
import { showTranscript } from "../transcripts";
import { type ProblemAction, ProblemLine } from "./ProblemLine";
import { useTabPlace } from "./tabs/tabItems";

/**
 * What the User should know about a Meeting's Transcript (#136), one line:
 * it is getting ready, live text fell behind ("Recording…"), it is being
 * finished or was saved, or what stopped it, with the one thing to do.
 */
export function TranscriptLines({
  meeting,
  className = "meeting-line",
}: {
  meeting: Meeting;
  className?: string;
}) {
  const t = useT();
  const place = useTabPlace();
  const transcript = meeting.transcript;
  if (!transcript) return null;
  const { problem, model, state } = transcript;
  const retry = (key: MessageKey): ProblemAction => ({
    label: t(key),
    testId: "meeting-transcript-retry",
    onClick: () => void useMeetings.getState().retryTranscript(meeting.id),
  });
  const downloading =
    model?.state === "downloading" || model?.state === "not-downloaded"
      ? {
          percent: Math.floor((model.downloadedBytes / Math.max(1, model.totalBytes)) * 100),
          size: formatBytes(model.totalBytes, t),
        }
      : null;

  let line: { text: string; action?: ProblemAction; role: "alert" | "status" } | null = null;
  if (problem) {
    const message = problem.message;
    if (problem.kind === "download") {
      line = {
        text: t("meetings.transcript.downloadFailed", { message }),
        action: retry("meetings.transcript.retry"),
        role: "alert",
      };
    } else if (problem.kind === "engine") {
      line = {
        text: message
          ? t("meetings.transcript.engineFailed", { message })
          : t("meetings.transcript.engineStopped"),
        action: retry("meetings.transcript.retry"),
        role: "alert",
      };
    } else {
      line = {
        text: message
          ? t("meetings.transcript.saveFailed", { message })
          : t("meetings.transcript.notSaved"),
        action: retry("meetings.transcript.save"),
        role: "alert",
      };
    }
  } else if (state === "preparing") {
    line = {
      text: downloading
        ? t("meetings.transcript.downloading", downloading)
        : t("meetings.transcript.starting"),
      role: "status",
    };
  } else if (state === "behind") {
    line = {
      text: t(
        meeting.status === "recording"
          ? "meetings.transcript.behind"
          : "meetings.transcript.behindPaused",
      ),
      role: "status",
    };
  } else if (state === "finishing") {
    line = {
      text: downloading
        ? t("meetings.transcript.finishingDownload", downloading)
        : t("meetings.transcript.finishing"),
      role: "status",
    };
  } else if (state === "saved" && transcript.documentId) {
    const documentId = transcript.documentId;
    line = {
      text: t("meetings.transcript.saved"),
      action: {
        label: t("meetings.transcript.open"),
        testId: "meeting-transcript-open",
        onClick: () => showTranscript(documentId, place?.groupId),
      },
      role: "status",
    };
  } else if (state === "empty") {
    line = { text: t("meetings.transcript.empty"), role: "status" };
  }
  if (!line) return null;
  return (
    <ProblemLine
      testId="meeting-transcript-line"
      role={line.role}
      className={className}
      action={line.action}
    >
      {line.text}
    </ProblemLine>
  );
}
