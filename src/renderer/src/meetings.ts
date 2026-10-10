import { useEffect, useState } from "react";
import { create } from "zustand";
import type { AudioSource, Meeting } from "../../core/api";
import { core } from "./core";
import { errorMessage } from "./errors";
import { useAppStore } from "./store";

interface MeetingsState {
  meetings: Meeting[];
  /** What went wrong with the last thing done to a Meeting, by its id, until the next try. */
  problems: Readonly<Record<string, string>>;
  /** The Meeting whose Start waits for the User to read the one-time recording notice. */
  noticeFor: { meetingId: string; systemAudio: boolean } | null;
  /** The gaps the User has dismissed, by Meeting: how many of its gaps they have seen. */
  gapsSeen: Readonly<Record<string, number>>;
  /** "Record a meeting": a new Mind, in a new tab, with the start block. */
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
  openSettings(meetingId: string, source: AudioSource): Promise<void>;
  dismissGaps(meetingId: string): void;
}

/** Meetings (#135), as the core lists them, and what the UI does with them. */
export const useMeetings = create<MeetingsState>()((set, get) => {
  /** Runs one action on a Meeting, keeping its failure as that Meeting's problem. */
  const attempt = async (meetingId: string, action: () => Promise<unknown>) => {
    set((state) => ({ problems: withoutKey(state.problems, meetingId) }));
    try {
      await action();
    } catch (error) {
      set((state) => ({ problems: { ...state.problems, [meetingId]: errorMessage(error) } }));
    }
  };

  return {
    meetings: [],
    problems: {},
    noticeFor: null,
    gapsSeen: {},

    async record() {
      try {
        const meeting = await core.createMeeting();
        useAppStore.getState().openMind(meeting.mindId, { newTab: true });
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
    openSettings: (meetingId, source) => attempt(meetingId, () => core.openAudioSettings(source)),
    dismissGaps(meetingId) {
      const meeting = get().meetings.find((each) => each.id === meetingId);
      set((state) => ({
        gapsSeen: { ...state.gapsSeen, [meetingId]: meeting?.gaps.length ?? 0 },
      }));
    },
  };
});

function withoutKey<T>(record: Readonly<Record<string, T>>, key: string): Record<string, T> {
  const { [key]: _, ...rest } = record;
  return rest;
}

core.listMeetings().then(
  (meetings) => useMeetings.setState({ meetings }),
  (error: unknown) => useAppStore.getState().reportError(error),
);
core.on("meetings.changed", (meetings) => useMeetings.setState({ meetings }));

/** The Mind's Meeting, if it has one. */
export const useMeetingOf = (mindId: string): Meeting | undefined =>
  useMeetings((state) => state.meetings.find((each) => each.mindId === mindId));

/** A Meeting is recording or paused: its Mind shows the meeting bar. */
export const isLive = (meeting: Meeting | undefined): meeting is Meeting =>
  meeting?.status === "recording" || meeting?.status === "paused";

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
  return meeting.recordedMs + (since === null ? 0 : Math.max(0, now - Date.parse(since)));
}
