import { useEffect, useState } from "react";
import { create } from "zustand";
import type { AudioSource, Meeting, Settings } from "../../core/api";
import { meetingTimeAt } from "../../shared/blockTimes";
import { mindMade } from "./changes";
import { core } from "./core";
import { errorMessage } from "./errors";
import { translateNow } from "./i18n";
import { currentMindIdOf } from "./layoutSlice";
import type { Translate } from "./objects";
import { useAppStore } from "./store";
import { UNDO_SHOWN_MS, type Undoable, useUndo } from "./undo";

interface MeetingsState {
  meetings: Meeting[];
  /** What went wrong with the last thing done to a Meeting, by its id, until the next try. */
  problems: Readonly<Record<string, string>>;
  /** The Meeting whose Start waits for the User to read the one-time recording notice. */
  noticeFor: { meetingId: string; systemAudio: boolean } | null;
  /** Record was chosen where this copy can't record audio: the sidebar's footer says so until it is dismissed. */
  cantRecord: boolean;
  /** The gaps the User has dismissed, by Meeting: how many of its gaps they have seen. */
  gapsSeen: Readonly<Record<string, number>>;
  /** The Meetings whose recording details are shown (#161). */
  detailsShown: Readonly<Record<string, boolean>>;
  /** Lists the Meetings, at startup; the core's events keep the list after that. */
  load(): Promise<void>;
  /**
   * "Record a meeting": the Meeting recording now, opened, or else a new Mind in a tab of its
   * own, in the Folder of the Mind in view; it starts recording at once once the User has read
   * the one-time notice, and shows the start block before.
   */
  record(): Promise<void>;
  clearCantRecord(): void;
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
    cantRecord: false,
    gapsSeen: {},
    detailsShown: {},

    async load() {
      try {
        set({ meetings: await core.listMeetings() });
      } catch (error) {
        useAppStore.getState().reportError(error);
      }
    },

    async record() {
      const recording = get().meetings.find((each) => each.status === "recording");
      if (recording) {
        useAppStore.getState().openMind(recording.mindId);
        return;
      }
      const { settings, layout, minds } = useAppStore.getState();
      if (!settings) return;
      if (settings.meetingAudio === "none") {
        set({ cantRecord: true });
        return;
      }
      try {
        const inView = currentMindIdOf(layout);
        const folderId = minds.find((each) => each.id === inView)?.folderId ?? null;
        const meeting = await core.createMeeting(folderId ? { folderId } : undefined);
        const starts = settings.device.meetingNoticeAccepted;
        takeBackable(meeting, starts);
        // The notes come first in a meeting: the text has the cursor, as in a Mind made with ⌘T.
        useAppStore.setState({ bodyToFocus: meeting.mindId });
        // A tab of its own, as New Mind opens one: never in place of the Mind the User is in.
        await useAppStore.getState().changeLayout({
          type: "open",
          item: { kind: "mind", mindId: meeting.mindId },
          where: { in: "group" },
        });
        if (starts) await get().start(meeting.id, meeting.systemAudio);
      } catch (error) {
        useAppStore.getState().reportError(error);
      }
    },

    clearCantRecord: () => set({ cantRecord: false }),

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

/** The Record whose new Mind Undo can still take back, by Meeting. */
const takeBackableRecords = new Map<
  string,
  { change: Undoable; seen: boolean; timer?: ReturnType<typeof setTimeout> }
>();

const capturedAnything = (meeting: Meeting) =>
  meeting.audio.kept ||
  meeting.recordedMs > 0 ||
  meeting.status === "paused" ||
  meeting.status === "stopped";

function dropTakeBack(meetingId: string): void {
  const pending = takeBackableRecords.get(meetingId);
  if (!pending) return;
  clearTimeout(pending.timer);
  takeBackableRecords.delete(meetingId);
  useUndo.getState().withdraw(pending.change);
}

/**
 * Offers Undo for the Mind a Record just made: it takes the Mind back while nothing has been
 * recorded, or just after a recording began. Once audio is saved, or the line has gone, it
 * doesn't offer it any more: Undo never deletes a recorded Meeting or its audio.
 */
function takeBackable(meeting: Meeting, recording: boolean): void {
  const takeBack = async () => {
    const now = useMeetings.getState().meetings.find((each) => each.id === meeting.id);
    if (now && capturedAnything(now)) return;
    await core.deleteMind(meeting.mindId);
  };
  const bringBack = async () => {
    await core.restore([{ kind: "mind", id: meeting.mindId }]);
    useAppStore.getState().openMind(meeting.mindId);
  };
  const change = recording
    ? {
        said: (t: Translate) => t("meetings.record.undo"),
        name: (t: Translate) => t("meetings.record"),
        undo: takeBack,
        redo: bringBack,
      }
    : mindMade("", takeBack, bringBack);
  takeBackableRecords.set(meeting.id, { change, seen: false });
  useUndo.getState().push(change);
}

function settleTakeBacks(meetings: readonly Meeting[]): void {
  for (const [meetingId, pending] of takeBackableRecords) {
    const meeting = meetings.find((each) => each.id === meetingId);
    if (!meeting) {
      if (pending.seen) dropTakeBack(meetingId);
      continue;
    }
    pending.seen = true;
    if (capturedAnything(meeting)) dropTakeBack(meetingId);
    else if (meeting.recordingSince !== null && !pending.timer) {
      pending.timer = setTimeout(() => dropTakeBack(meetingId), UNDO_SHOWN_MS);
    }
  }
}

core.on("meetings.changed", (meetings) => {
  useMeetings.setState({ meetings });
  settleTakeBacks(meetings);
});

/** This copy can record audio. */
export const canRecordMeetings = (settings: Settings | null): boolean =>
  !!settings && settings.meetingAudio !== "none";

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
  return meetingTimeAt(meeting, now);
}
