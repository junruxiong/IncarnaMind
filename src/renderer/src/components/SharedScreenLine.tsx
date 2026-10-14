import { isMacOS } from "@tiptap/core";
import { Fragment } from "react";
import type { Meeting } from "../../../core/api";
import { useT } from "../i18n";
import { formatBytes } from "../linkedFolders";
import { useMeetings } from "../meetings";
import { ScreenLineIcon } from "./lineIcons";

interface LineAction {
  label: string;
  testId: string;
  onClick(): void;
  /** The way on the line asks for: in the accent. The others are in ink. */
  primary?: boolean;
}

/**
 * The meeting bar's line for the shared screen (ADR-0024, D210): while the
 * Meeting records, it asks whether someone will share their screen, until
 * the User answers; then it says which window is read, with Stop, or offers
 * "Read the shared screen…" for later in the call. What stopped the reading
 * is said here too, with the one thing to do. Nothing while paused: only a
 * recording Meeting reads its screen.
 */
export function SharedScreenLine({ meeting }: { meeting: Meeting }) {
  const t = useT();
  const { chooseScreen, declineScreen, stopScreen, openSettings } = useMeetings.getState();
  const explained = useMeetings((state) => state.screenExplained[meeting.id] ?? false);
  const screen = meeting.screen;
  if (!screen || meeting.status !== "recording") return null;
  const choose = (primary = true): LineAction => ({
    label: t("meetings.screen.choose"),
    testId: "meeting-screen-choose",
    onClick: () => void chooseScreen(meeting.id),
    primary,
  });

  let text: string | null = null;
  let actions: LineAction[] = [];
  let role: "status" | "alert" = "status";
  const decline: LineAction = {
    label: t("meetings.screen.decline"),
    testId: "meeting-screen-decline",
    onClick: () => void declineScreen(meeting.id),
  };
  if (screen.permission === "not-asked" && explained && screen.state !== "reading") {
    // Said before macOS asks for Screen Recording, which reading a window needs.
    text = t("meetings.screen.explain");
    actions = [
      {
        label: t("meetings.screen.continue"),
        testId: "meeting-screen-continue",
        onClick: () => void chooseScreen(meeting.id),
        primary: true,
      },
      ...(screen.state === "ask" ? [decline] : []),
    ];
  } else if (screen.state === "ask") {
    text = t("meetings.screen.ask");
    actions = [choose(), decline];
  } else if (screen.state === "choosing") {
    text = t("meetings.screen.choosing");
  } else if (screen.state === "reading") {
    const window = screen.window ?? "";
    const model = screen.model;
    // The system's picker doesn't always say which window it gave.
    text =
      model && (model.state === "downloading" || model.state === "not-downloaded")
        ? t(window ? "meetings.screen.downloading" : "meetings.screen.downloadingShared", {
            window,
            percent: String(
              Math.floor((model.downloadedBytes / Math.max(1, model.totalBytes)) * 100),
            ),
            size: formatBytes(model.totalBytes, t),
          })
        : t(window ? "meetings.screen.reading" : "meetings.screen.readingShared", { window });
    actions = [
      {
        label: t("meetings.screen.stop"),
        testId: "meeting-screen-stop",
        onClick: () => void stopScreen(meeting.id),
      },
    ];
  } else if (screen.problem) {
    role = "alert";
    if (screen.problem === "whole-screen") {
      text = t("meetings.screen.wholeScreen");
      actions = [choose()];
    } else if (screen.problem === "denied") {
      text = t("meetings.screen.denied");
      // On macOS, the window is captured with Screen & System Audio Recording, where the computer's sound is.
      actions = isMacOS()
        ? [
            {
              label: t("meetings.line.openSettings.mac"),
              testId: "meeting-screen-settings",
              onClick: () => void openSettings(meeting.id, "system"),
              primary: true,
            },
          ]
        : [choose()];
    } else if (screen.problem === "closed") {
      text = t("meetings.screen.closed");
      actions = [choose()];
    } else {
      text = t("meetings.screen.failed");
      actions = [
        {
          label: t("meetings.screen.retry"),
          testId: "meeting-screen-retry",
          onClick: () => void chooseScreen(meeting.id),
          primary: true,
        },
      ];
    }
  }

  return (
    <p
      data-testid="meeting-screen-line"
      data-state={screen.state}
      data-problem={screen.problem ?? undefined}
      role={role}
      className="meeting-screen-line"
    >
      <ScreenLineIcon className="meeting-screen-icon" />
      <span className="meeting-screen-text">
        {text === null ? (
          <button
            type="button"
            data-testid="meeting-screen-offer"
            onClick={() => void chooseScreen(meeting.id)}
            className="meeting-screen-action"
          >
            {t("meetings.screen.offer")}
          </button>
        ) : (
          <>
            <span data-testid="meeting-screen-words">{text}</span>
            {actions.map((action) => (
              <Fragment key={action.testId}>
                <span aria-hidden="true"> · </span>
                <button
                  type="button"
                  data-testid={action.testId}
                  onClick={action.onClick}
                  className={`meeting-screen-action ${action.primary ? "meeting-screen-action--primary" : ""}`}
                >
                  {action.label}
                </button>
              </Fragment>
            ))}
          </>
        )}
      </span>
    </p>
  );
}
