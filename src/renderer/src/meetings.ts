import { useEffect, useState } from "react";
import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";
import type { AudioSource, Meeting, Settings, ShareableWindow } from "../../core/api";
import { meetingTimeAt } from "../../shared/blockTimes";
import { core } from "./core";
import { errorMessage } from "./errors";
import { translateNow } from "./i18n";
import { useAppStore } from "./store";

interface MeetingsState {
  meetings: Meeting[];
  /** What went wrong with the last thing done to a Meeting, by its id, until the next try. */
  problems: Readonly<Record<string, string>>;
  /** The Meeting whose Start waits for the User to read the one-time recording notice. */
  noticeFor: { meetingId: string; systemAudio: boolean } | null;
  /** The gaps the User has dismissed, by Meeting: how many of its gaps they have seen. */
  gapsSeen: Readonly<Record<string, number>>;
  /** The Meetings whose recording details are shown (#161). */
  detailsShown: Readonly<Record<string, boolean>>;
  /**
   * IncarnaMind's window picker for a Meeting's shared screen (ADR-0024),
   * while open: the windows to choose from, null while they are found.
   */
  screenPicker: { meetingId: string; windows: ShareableWindow[] | null } | null;
  /**
   * The Meetings whose bar has said why macOS will ask for Screen Recording:
   * Choose window… then asks for it.
   */
  screenExplained: Readonly<Record<string, boolean>>;
  /** Lists the Meetings, at startup; the core's events keep the list after that. */
  load(): Promise<void>;
  /** "Record a meeting": a new Mind, in a tab of its own, with the start block. */
  record(): Promise<void>;
  /** Start recording, after the one-time notice if this device hasn't shown it. */
  start(meetingId: string, systemAudio: boolean): Promise<void>;
  /** The notice was read: recording starts, and the notice isn't shown again. */
  acceptNotice(): Promise<void>;
  cancelNotice(): void;
  pause(meetingId: string): Promise<void>;
  resume(meetingId: string): Promise<void>;
  stop(meetingId: string): Promise<void>;
  discard(meetingId: string): Promise<void>;
  deleteAudio(meetingId: string): Promise<void>;
  /** Tries the Meeting's Transcript again after a problem: the model's download, its engine, or saving it. */
  retryTranscript(meetingId: string): Promise<void>;
  openSettings(meetingId: string, source: AudioSource): Promise<void>;
  dismissGaps(meetingId: string): void;
  /** Shows or hides a Meeting's recording details; with no `shown`, the other way from now. */
  showDetails(meetingId: string, shown?: boolean): void;
  /**
   * Choose window…: IncarnaMind's picker with the windows it can read; on
   * macOS, the first time, the bar first says why macOS will ask.
   */
  chooseScreen(meetingId: string): Promise<void>;
  /** Reads the window chosen in IncarnaMind's picker, which closes. */
  readScreen(meetingId: string, windowId: string): Promise<void>;
  closeScreenPicker(): void;
  /** Not this time: the meeting bar stops asking. */
  declineScreen(meetingId: string): Promise<void>;
  stopScreen(meetingId: string): Promise<void>;
}

/** Meetings (#135), as the core lists them, and what the UI does with them. */
export const useMeetings = create<MeetingsState>()((set, get) => {
  /** Runs one action on a Meeting, keeping its failure as that Meeting's problem. */
  const attempt = async (meetingId: string, action: () => Promise<unknown>) => {
    set((state) => ({ problems: withoutKey(state.problems, meetingId) }));
    try {
      await action();
    } catch (error) {
      set((state) => ({
        problems: { ...state.problems, [meetingId]: errorMessage(error, translateNow) },
      }));
    }
  };

  return {
    meetings: [],
    problems: {},
    noticeFor: null,
    gapsSeen: {},
    detailsShown: {},
    screenPicker: null,
    screenExplained: {},

    async load() {
      try {
        set({ meetings: await core.listMeetings() });
      } catch (error) {
        useAppStore.getState().reportError(error);
      }
    },

    async record() {
      try {
        const meeting = await core.createMeeting();
        // A tab of its own, as New Mind opens one: never in place of the Mind the User is in.
        await useAppStore.getState().changeLayout({
          type: "open",
          item: { kind: "mind", mindId: meeting.mindId },
          where: { in: "group" },
        });
      } catch (error) {
        useAppStore.getState().reportError(error);
      }
    },

    async start(meetingId, systemAudio) {
      if (!useAppStore.getState().settings?.device.meetingNoticeAccepted) {
        set({ noticeFor: { meetingId, systemAudio } });
        return;
      }
      await attempt(meetingId, () => core.startMeeting({ meetingId, systemAudio }));
    },

    async acceptNotice() {
      const waiting = get().noticeFor;
      set({ noticeFor: null });
      if (!waiting) return;
      await attempt(waiting.meetingId, async () => {
        await core.updateSettings({ device: { meetingNoticeAccepted: true } });
        await core.startMeeting(waiting);
      });
    },

    cancelNotice: () => set({ noticeFor: null }),
    pause: (meetingId) => attempt(meetingId, () => core.pauseMeeting(meetingId)),
    resume: (meetingId) => attempt(meetingId, () => core.resumeMeeting(meetingId)),
    stop: (meetingId) => attempt(meetingId, () => core.stopMeeting(meetingId)),
    discard: (meetingId) => attempt(meetingId, () => core.discardMeeting(meetingId)),
    deleteAudio: (meetingId) => attempt(meetingId, () => core.deleteMeetingAudio(meetingId)),
    retryTranscript: (meetingId) => attempt(meetingId, () => core.retryTranscript(meetingId)),
    openSettings: (meetingId, source) => attempt(meetingId, () => core.openAudioSettings(source)),
    dismissGaps(meetingId) {
      const meeting = get().meetings.find((each) => each.id === meetingId);
      set((state) => ({
        gapsSeen: { ...state.gapsSeen, [meetingId]: meeting?.gaps.length ?? 0 },
      }));
    },
    async chooseScreen(meetingId) {
      const meeting = get().meetings.find((each) => each.id === meetingId);
      // macOS asks for Screen Recording the first time: the bar says why before it does.
      if (meeting?.screen?.permission === "not-asked" && !get().screenExplained[meetingId]) {
        set((state) => ({ screenExplained: { ...state.screenExplained, [meetingId]: true } }));
        return;
      }
      set({ screenPicker: { meetingId, windows: null } });
      await attempt(meetingId, async () => {
        try {
          const windows = await core.listShareableWindows(meetingId);
          set((state) =>
            state.screenPicker?.meetingId !== meetingId
              ? {}
              : { screenPicker: windows ? { meetingId, windows } : null },
          );
        } catch (error) {
          set({ screenPicker: null });
          throw error;
        }
      });
    },
    async readScreen(meetingId, windowId) {
      set({ screenPicker: null });
      await attempt(meetingId, () => core.readSharedScreen({ meetingId, windowId }));
    },
    closeScreenPicker: () => set({ screenPicker: null }),
    declineScreen: (meetingId) => attempt(meetingId, () => core.declineSharedScreen(meetingId)),
    stopScreen: (meetingId) => attempt(meetingId, () => core.stopSharedScreen(meetingId)),
    showDetails(meetingId, shown) {
      set((state) => ({
        detailsShown: {
          ...state.detailsShown,
          [meetingId]: shown ?? !state.detailsShown[meetingId],
        },
      }));
    },
  };
});

function withoutKey<T>(record: Readonly<Record<string, T>>, key: string): Record<string, T> {
  const { [key]: _, ...rest } = record;
  return rest;
}

core.on("meetings.changed", (meetings) => useMeetings.setState({ meetings }));

/** Meetings are turned on in Settings, and this copy can record. */
export const canRecordMeetings = (settings: Settings | null): boolean =>
  !!settings?.device.meetingsEnabled && settings.meetingAudio !== "none";

/** The Mind's Meeting, if it has one. */
export const useMeetingOf = (mindId: string): Meeting | undefined =>
  useMeetings((state) => state.meetings.find((each) => each.mindId === mindId));

/** A Meeting is recording or paused: its Mind shows the meeting bar. */
export const isLive = (meeting: Meeting | undefined): meeting is Meeting =>
  meeting?.status === "recording" || meeting?.status === "paused";

/**
 * The shared screens still kept (ADR-0024): each recording Meeting's memory,
 * which starts its screens' ids. A screen Citation whose screen isn't kept was deleted.
 */
export const useKeptScreens = (): string[] =>
  useMeetings(
    useShallow((state) => state.meetings.flatMap((meeting) => meeting.screen?.memory ?? [])),
  );

/** Whether some Meeting's audio is being captured now: background work waits meanwhile. */
export const useCapturing = (): boolean =>
  useMeetings((state) => state.meetings.some((each) => each.recordingSince !== null));

/** The Meeting's time now, in ms, ticking each second while it records. */
export function useMeetingTime(meeting: Meeting): number {
  const since = meeting.recordingSince;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (since === null) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [since]);
  return meetingTimeAt(meeting, now);
}
