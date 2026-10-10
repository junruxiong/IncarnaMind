/**
 * Meetings (#135): the recording's state, and its audio on disk.
 *
 * A Meeting belongs to a Mind. It starts "ready" (the Mind shows the start
 * block), then records, pauses and stops. The audio comes from the host's
 * `AudioCapture` and goes to disk in 10-second parts per source as it comes
 * (./audioFiles), so a crash, sleep or device change loses at most the part
 * still filling.
 *
 * The Meeting's time is recording time: each stretch recorded (a period)
 * starts where the one before ended, so pauses and gaps take none of it. A
 * stretch that ended because the computer slept or IncarnaMind closed leaves a
 * gap, marked at that point with how long was missed. A Meeting found
 * recording at startup was cut off by a crash or a quit: its stretch ends
 * where its last saved part ends, and it waits, paused, for the User to resume.
 *
 * Its audio is kept for at most a day after the last stretch, then deleted;
 * the User can delete it sooner. While audio is being captured, background
 * work holds back (`onCapturingChange`): live capture comes first.
 */
import { randomUUID } from "node:crypto";
import { translate } from "../../shared/i18n";
import type { AudioCapture, CaptureSession, CaptureState, Logger } from "../adapters";
import {
  type AudioSource,
  MEETING_AUDIO_KEPT_MS,
  type Meeting,
  type MeetingGap,
  type MeetingSourceState,
  type Mind,
} from "../api";
import { InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { Language } from "../language";
import type { Database } from "../storage";
import {
  type AudioPart,
  listParts,
  meetingFolder,
  msOf,
  PART_SAMPLES,
  removeAudio,
  removeUnfinished,
  writePart,
} from "./audioFiles";

/** Samples arriving this much later than the ones before them could have: a new run starts where they belong. */
const DISCONTINUITY_MS = 1000;

/** The computer's sound, all silence for this long, is said to be silent: macOS gives silence when it isn't allowed. */
const SILENT_SAMPLES = 16_000 * 60;

/** How often audio past its day is looked for. */
const SWEEP_INTERVAL_MS = 60_000;

type EndedBy = "pause" | "stop" | "sleep" | "interrupted";

/** One stretch recorded. `durationMs` is null while it lasts. */
interface Period {
  start: string;
  offsetMs: number;
  durationMs: number | null;
  endedBy: EndedBy | null;
}

interface MeetingRow {
  id: string;
  mind_id: string;
  status: Meeting["status"];
  paused_by: Meeting["pausedBy"];
  system_audio: number;
  periods: string;
  started_at: string | null;
  stopped_at: string | null;
  audio_expires_at: string | null;
  audio_deleted_at: string | null;
  created_at: string;
  updated_at: string;
}

const COLUMNS =
  "id, mind_id, status, paused_by, system_audio, periods, started_at, stopped_at, audio_expires_at, audio_deleted_at, created_at, updated_at";

/** One source's samples since its run began: what is saved, and what waits to fill a part. */
interface Run {
  /** Where the run starts, in the Meeting's time. */
  startMs: number;
  /** Samples in it so far, saved or waiting. */
  samples: number;
  saved: number;
  waiting: Int16Array[];
  waitingSamples: number;
}

/** The Meeting whose audio is being captured now. */
interface Live {
  meetingId: string;
  folder: string;
  period: Period;
  session: CaptureSession | null;
  sources: Record<AudioSource, MeetingSourceState>;
  runs: Map<AudioSource, Run>;
  /** Where each source's saved audio ends, so a new run never overlaps it. */
  ends: Map<AudioSource, number>;
  /** All-silent samples in a row from the computer's sound. */
  silent: number;
  saveError: string | null;
}

export interface MeetingsOptions {
  db: Database;
  dataDir: string;
  now: () => string;
  capture: AudioCapture | undefined;
  enabled(): boolean;
  noticeAccepted(): boolean;
  /** The "What is recorded" choice in Settings. */
  systemAudioByDefault(): boolean;
  language(): Language;
  createMind(title: string): Mind;
  mindExists(mindId: string): boolean;
  changed(meetings: Meeting[]): void;
  /** Whether audio is being captured, each time that changes. */
  onCapturingChange(capturing: boolean): void;
  log: Logger;
}

export type Meetings = ReturnType<typeof createMeetings>;

export function createMeetings(options: MeetingsOptions) {
  const { db, now, capture, log } = options;
  const clock = () => Date.parse(now());
  const folderOf = (meetingId: string) => meetingFolder(options.dataDir, meetingId);
  let live: Live | null = null;
  /** Whether each Meeting has audio on disk, looked up once. */
  const kept = new Map<string, boolean>();

  const rowOf = (meetingId: unknown): MeetingRow => {
    if (typeof meetingId !== "string" || meetingId === "") {
      throw new InvalidInputError("A Meeting id must be a non-empty string.");
    }
    const row = db.get<MeetingRow>(
      `SELECT ${COLUMNS} FROM meetings WHERE id = ? AND deleted_at IS NULL`,
      [meetingId],
    );
    if (!row) throw new NotFoundError("That Meeting doesn't exist or has been discarded.");
    return row;
  };

  const periodsOf = (row: MeetingRow): Period[] => JSON.parse(row.periods) as Period[];

  const hasAudio = (meetingId: string): boolean => {
    let value = kept.get(meetingId);
    if (value === undefined) {
      value = listParts(folderOf(meetingId)).length > 0;
      kept.set(meetingId, value);
    }
    return value;
  };

  /** Where the stretch ends in wall-clock time, in ms. */
  const endOf = (period: Period) => Date.parse(period.start) + (period.durationMs ?? 0);

  function toMeeting(row: MeetingRow): Meeting {
    const periods = periodsOf(row);
    const last = periods.at(-1);
    const open = last && last.durationMs === null ? last : null;
    const gaps: MeetingGap[] = [];
    for (let i = 1; i < periods.length; i++) {
      const before = periods[i - 1] as Period;
      const after = periods[i] as Period;
      if (before.endedBy !== "sleep" && before.endedBy !== "interrupted") continue;
      gaps.push({
        atMs: after.offsetMs,
        missedMs: Math.max(0, Date.parse(after.start) - endOf(before)),
        reason: before.endedBy,
      });
    }
    const current = live?.meetingId === row.id ? live : null;
    return {
      id: row.id,
      mindId: row.mind_id,
      status: row.status,
      pausedBy: row.paused_by,
      systemAudio: row.system_audio === 1,
      sources: current ? { ...current.sources } : { microphone: "off", system: "off" },
      recordedMs: open ? open.offsetMs : last ? last.offsetMs + (last.durationMs ?? 0) : 0,
      recordingSince: open ? open.start : null,
      gaps,
      startedAt: row.started_at,
      stoppedAt: row.stopped_at,
      audio: {
        kept: hasAudio(row.id),
        expiresAt: row.audio_expires_at,
        deletedAt: row.audio_deleted_at,
      },
      saveError: current?.saveError ?? null,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }

  const list = (): Meeting[] =>
    db
      .all<MeetingRow>(
        `SELECT ${COLUMNS} FROM meetings WHERE deleted_at IS NULL ORDER BY created_at DESC, rowid DESC`,
      )
      .map(toMeeting);

  const changed = () => options.changed(list());

  function save(
    meetingId: string,
    change: Partial<Omit<MeetingRow, "id" | "mind_id" | "created_at" | "updated_at">>,
  ): void {
    const entries = Object.entries(change);
    const at = now();
    db.run(
      `UPDATE meetings SET ${entries.map(([column]) => `${column} = ?`).join(", ")}, updated_at = ? WHERE id = ?`,
      [...entries.map(([, value]) => value as string | number | null), at, meetingId],
    );
  }

  /** How far into the Meeting's time it is now, while audio is being captured. */
  const meetingTime = (current: Live) =>
    current.period.offsetMs + (clock() - Date.parse(current.period.start));

  // ---- Audio to disk ----

  /** Takes the first `count` samples waiting in a run. */
  function take(run: Run, count: number): Int16Array {
    const out = new Int16Array(count);
    let filled = 0;
    while (filled < count) {
      const first = run.waiting[0] as Int16Array;
      const needed = count - filled;
      if (first.length <= needed) {
        out.set(first, filled);
        filled += first.length;
        run.waiting.shift();
      } else {
        out.set(first.subarray(0, needed), filled);
        run.waiting[0] = first.subarray(needed);
        filled += needed;
      }
    }
    run.waitingSamples -= count;
    return out;
  }

  function savePart(current: Live, source: AudioSource, run: Run, samples: Int16Array): void {
    const offsetMs = run.startMs + msOf(run.saved);
    run.saved += samples.length;
    current.ends.set(source, offsetMs + msOf(samples.length));
    try {
      writePart(current.folder, source, offsetMs, samples);
    } catch (error) {
      // The part is lost; the Meeting says so until a part is saved again.
      const code = (error as NodeJS.ErrnoException).code ?? "unknown";
      log.error("meeting.saveFailed", { meetingId: current.meetingId, source, code });
      if (current.saveError === null) {
        current.saveError = error instanceof Error ? error.message : String(error);
        changed();
      }
      return;
    }
    const first = !hasAudio(current.meetingId);
    kept.set(current.meetingId, true);
    if (first || current.saveError !== null) {
      current.saveError = null;
      changed();
    }
  }

  /** Saves what a run has waiting, however short, and ends it. */
  function endRun(current: Live, source: AudioSource): void {
    const run = current.runs.get(source);
    if (!run) return;
    current.runs.delete(source);
    if (run.waitingSamples > 0) savePart(current, source, run, take(run, run.waitingSamples));
  }

  function hear(current: Live, source: AudioSource, samples: Int16Array): void {
    if (live !== current || samples.length === 0) return;
    const at = meetingTime(current);
    let run = current.runs.get(source);
    // Samples that come much later than the run could have produced them (a device changed,
    // the capture stalled) start a new run where they belong, so the parts keep the right times.
    if (run && at - (run.startMs + msOf(run.samples + samples.length)) > DISCONTINUITY_MS) {
      endRun(current, source);
      run = undefined;
    }
    if (!run) {
      const ends = current.ends.get(source) ?? current.period.offsetMs;
      run = {
        startMs: Math.max(ends, current.period.offsetMs, at - msOf(samples.length)),
        samples: 0,
        saved: 0,
        waiting: [],
        waitingSamples: 0,
      };
      current.runs.set(source, run);
    }
    run.waiting.push(samples);
    run.waitingSamples += samples.length;
    run.samples += samples.length;
    while (run.waitingSamples >= PART_SAMPLES) {
      savePart(current, source, run, take(run, PART_SAMPLES));
    }
    if (source === "system") {
      current.silent = samples.every((sample) => sample === 0)
        ? current.silent + samples.length
        : 0;
    }
    setSource(
      current,
      source,
      source === "system" && current.silent >= SILENT_SAMPLES ? "silent" : "recording",
    );
  }

  function setSource(current: Live, source: AudioSource, state: MeetingSourceState): void {
    if (current.sources[source] === state) return;
    current.sources[source] = state;
    log.info("meeting.source", { meetingId: current.meetingId, source, state });
    changed();
  }

  function heardState(current: Live, source: AudioSource, state: CaptureState): void {
    if (live !== current) return;
    if (state !== "recording") endRun(current, source);
    if (state === "recording" && current.sources[source] === "silent") return;
    setSource(current, source, state);
  }

  /** Starts capturing a Meeting's audio, in a stretch that begins now. */
  function begin(row: MeetingRow, period: Period): void {
    if (!capture) throw new InvalidInputError("This copy of IncarnaMind can't record audio.");
    const systemAudio = row.system_audio === 1;
    const current: Live = {
      meetingId: row.id,
      folder: folderOf(row.id),
      period,
      session: null,
      sources: {
        microphone: "starting",
        system: !systemAudio ? "off" : capture.systemAudio ? "starting" : "unavailable",
      },
      runs: new Map(),
      ends: new Map(),
      silent: 0,
      saveError: null,
    };
    for (const part of listParts(current.folder)) {
      const end = part.offsetMs + part.durationMs;
      if (end > (current.ends.get(part.source) ?? 0)) current.ends.set(part.source, end);
    }
    live = current;
    options.onCapturingChange(true);
    current.session = capture.start(
      { systemAudio: systemAudio && capture.systemAudio },
      {
        audio: (source, samples) => hear(current, source, samples),
        state: (source, state) => heardState(current, source, state),
      },
    );
  }

  /**
   * Stops capturing: what is waiting is saved now, and the stretch ends where
   * its audio does, or now if that is later. Returns the stretch's length.
   */
  function end(current: Live): number {
    live = null;
    endRun(current, "microphone");
    endRun(current, "system");
    current.session?.stop().catch((error: unknown) => {
      log.warn("meeting.captureStopFailed", {
        meetingId: current.meetingId,
        message: error instanceof Error ? error.message : String(error),
      });
    });
    options.onCapturingChange(false);
    const audioEnd = Math.max(0, ...current.ends.values());
    return Math.max(clock() - Date.parse(current.period.start), audioEnd - current.period.offsetMs);
  }

  /** Ends the Meeting's open stretch, if it has one, and returns its stretches. */
  function closePeriod(row: MeetingRow, endedBy: EndedBy): Period[] {
    const periods = periodsOf(row);
    const open = periods.at(-1);
    if (!open || open.durationMs !== null) return periods;
    open.durationMs =
      live?.meetingId === row.id ? end(live) : Math.max(0, clock() - Date.parse(open.start));
    open.endedBy = endedBy;
    return periods;
  }

  /** When the Meeting's audio goes: a day after its last stretch, if it has any. */
  const expiryOf = (meetingId: string, periods: Period[]) => {
    const last = periods.at(-1);
    return last && hasAudio(meetingId)
      ? new Date(endOf(last) + MEETING_AUDIO_KEPT_MS).toISOString()
      : null;
  };

  const recordedOf = (periods: Period[]) => {
    const last = periods.at(-1);
    return last ? last.offsetMs + (last.durationMs ?? 0) : 0;
  };

  function assertNoneCapturing(meetingId: string): void {
    if (live && live.meetingId !== meetingId) {
      throw new InvalidInputError("Another Meeting is being recorded. Stop it first.");
    }
  }

  // ---- Retention ----

  function deleteAudio(meetingId: string, reason: "user" | "expired" | "discarded"): void {
    if (live?.meetingId === meetingId) {
      live.runs.clear();
    }
    removeAudio(folderOf(meetingId));
    kept.set(meetingId, false);
    log.info("meeting.audioDeleted", { meetingId, reason });
  }

  /** Deletes audio a day past its Meeting's last stretch. */
  function sweep(): void {
    const at = now();
    const due = db.all<{ id: string }>(
      `SELECT id FROM meetings
       WHERE deleted_at IS NULL AND audio_expires_at IS NOT NULL AND audio_expires_at <= ?
         AND status IN ('paused', 'stopped')`,
      [at],
    );
    if (due.length === 0) return;
    for (const { id } of due) {
      deleteAudio(id, "expired");
      save(id, { audio_deleted_at: at, audio_expires_at: null });
    }
    changed();
  }

  // ---- After a crash or a quit ----

  /**
   * A Meeting still recording was cut off: its stretch ends where its saved
   * audio does, and it waits, paused, for the User. Parts half written go.
   */
  function recover(): void {
    const cut = db.all<MeetingRow>(
      `SELECT ${COLUMNS} FROM meetings WHERE deleted_at IS NULL AND status = 'recording'`,
    );
    for (const row of cut) {
      const folder = folderOf(row.id);
      const unfinished = removeUnfinished(folder);
      const parts = listParts(folder);
      const periods = periodsOf(row);
      const last = periods.at(-1);
      if (last && last.durationMs === null) {
        const audioEnd = Math.max(
          last.offsetMs,
          ...parts.map((part) => part.offsetMs + part.durationMs),
        );
        last.durationMs = audioEnd - last.offsetMs;
      }
      // Asleep when IncarnaMind closed, or cut off: either way it didn't record until now.
      if (last) last.endedBy = "interrupted";
      save(row.id, {
        status: "paused",
        paused_by: "interrupted",
        periods: JSON.stringify(periods),
        audio_expires_at: expiryOf(row.id, periods),
      });
      log.warn("meeting.recovered", { meetingId: row.id, parts: parts.length, unfinished });
    }
  }

  recover();
  sweep();
  const sweeping = setInterval(() => {
    try {
      sweep();
    } catch (error) {
      log.error("meeting.sweepFailed", {
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }, SWEEP_INTERVAL_MS);
  sweeping.unref?.();

  const meetingTitle = () => {
    const language = options.language();
    const date = new Intl.DateTimeFormat(language === "zh-CN" ? "zh-CN" : "en-GB", {
      day: "numeric",
      month: "long",
    }).format(new Date(clock()));
    return translate(language, "meetings.mindTitle", { date });
  };

  const parseInput = (input: unknown): Record<string, unknown> => {
    if (input === undefined) return {};
    if (!isRecord(input)) throw new InvalidInputError("Expected an object.");
    return input;
  };

  return {
    list,

    create(input: unknown): Meeting {
      if (!options.enabled()) throw new InvalidInputError("Meetings are turned off in Settings.");
      if (!capture) throw new InvalidInputError("This copy of IncarnaMind can't record audio.");
      const { mindId } = parseInput(input);
      if (mindId !== undefined && (typeof mindId !== "string" || !options.mindExists(mindId))) {
        throw new NotFoundError("That Mind doesn't exist or has been deleted.");
      }
      if (
        typeof mindId === "string" &&
        db.get("SELECT 1 FROM meetings WHERE mind_id = ? AND deleted_at IS NULL", [mindId])
      ) {
        throw new InvalidInputError("That Mind has a Meeting already.");
      }
      const id = randomUUID();
      const at = now();
      db.transaction(() => {
        const mind = typeof mindId === "string" ? mindId : options.createMind(meetingTitle()).id;
        db.run(
          `INSERT INTO meetings (id, mind_id, status, system_audio, created_at, updated_at)
           VALUES (?, ?, 'ready', ?, ?, ?)`,
          [id, mind, options.systemAudioByDefault() ? 1 : 0, at, at],
        );
      });
      changed();
      return toMeeting(rowOf(id));
    },

    discard(meetingId: unknown): void {
      const row = rowOf(meetingId);
      if (row.status !== "ready") {
        throw new InvalidInputError("Only a Meeting that hasn't started can be discarded.");
      }
      const at = now();
      db.run("UPDATE meetings SET deleted_at = ?, updated_at = ? WHERE id = ?", [at, at, row.id]);
      changed();
    },

    start(input: unknown): Meeting {
      const { meetingId, systemAudio } = parseInput(input);
      const row = rowOf(meetingId);
      if (row.status !== "ready") throw new InvalidInputError("That Meeting has started already.");
      if (systemAudio !== undefined && typeof systemAudio !== "boolean") {
        throw new InvalidInputError("systemAudio must be true or false.");
      }
      if (!options.enabled()) throw new InvalidInputError("Meetings are turned off in Settings.");
      if (!capture) throw new InvalidInputError("This copy of IncarnaMind can't record audio.");
      if (!options.noticeAccepted()) {
        throw new InvalidInputError("Read the recording notice before recording a Meeting.");
      }
      assertNoneCapturing(row.id);
      const at = now();
      const period: Period = { start: at, offsetMs: 0, durationMs: null, endedBy: null };
      const system = systemAudio ?? row.system_audio === 1;
      save(row.id, {
        status: "recording",
        system_audio: system ? 1 : 0,
        periods: JSON.stringify([period]),
        started_at: at,
      });
      begin(rowOf(row.id), period);
      log.info("meeting.started", { meetingId: row.id, systemAudio: system });
      changed();
      return toMeeting(rowOf(row.id));
    },

    pause(meetingId: unknown): Meeting {
      const row = rowOf(meetingId);
      if (row.status !== "recording") throw new InvalidInputError("That Meeting isn't recording.");
      const periods = closePeriod(row, "pause");
      save(row.id, {
        status: "paused",
        paused_by: "user",
        periods: JSON.stringify(periods),
        audio_expires_at: expiryOf(row.id, periods),
      });
      log.info("meeting.paused", { meetingId: row.id });
      changed();
      return toMeeting(rowOf(row.id));
    },

    resume(meetingId: unknown): Meeting {
      const row = rowOf(meetingId);
      if (row.status !== "paused") throw new InvalidInputError("That Meeting isn't paused.");
      assertNoneCapturing(row.id);
      const periods = periodsOf(row);
      const period: Period = {
        start: now(),
        offsetMs: recordedOf(periods),
        durationMs: null,
        endedBy: null,
      };
      save(row.id, {
        status: "recording",
        paused_by: null,
        periods: JSON.stringify([...periods, period]),
        audio_expires_at: null,
      });
      begin(rowOf(row.id), period);
      log.info("meeting.resumed", { meetingId: row.id });
      changed();
      return toMeeting(rowOf(row.id));
    },

    stop(meetingId: unknown): Meeting {
      const row = rowOf(meetingId);
      if (row.status !== "recording" && row.status !== "paused") {
        throw new InvalidInputError("That Meeting isn't being recorded.");
      }
      const periods = closePeriod(row, "stop");
      save(row.id, {
        status: "stopped",
        paused_by: null,
        periods: JSON.stringify(periods),
        stopped_at: now(),
        audio_expires_at: expiryOf(row.id, periods),
      });
      log.info("meeting.stopped", { meetingId: row.id });
      changed();
      return toMeeting(rowOf(row.id));
    },

    deleteAudio(meetingId: unknown): Meeting {
      const row = rowOf(meetingId);
      deleteAudio(row.id, "user");
      save(row.id, { audio_deleted_at: now(), audio_expires_at: null });
      changed();
      return toMeeting(rowOf(row.id));
    },

    async openSettings(source: unknown): Promise<void> {
      if (source !== "microphone" && source !== "system") {
        throw new InvalidInputError('The source must be "microphone" or "system".');
      }
      if (!capture) throw new InvalidInputError("This copy of IncarnaMind can't record audio.");
      await capture.openSettings(source);
    },

    /** The Mind was deleted: its Meeting stops, and its audio goes with it. */
    mindDeleted(mindId: string): void {
      const rows = db.all<MeetingRow>(
        `SELECT ${COLUMNS} FROM meetings WHERE mind_id = ? AND deleted_at IS NULL`,
        [mindId],
      );
      if (rows.length === 0) return;
      const at = now();
      for (const row of rows) {
        if (live?.meetingId === row.id) end(live);
        deleteAudio(row.id, "discarded");
        db.run("UPDATE meetings SET deleted_at = ?, updated_at = ? WHERE id = ?", [
          at,
          at,
          row.id,
        ]);
      }
      changed();
    },

    /** The computer is going to sleep: what is waiting is saved, and capture stops until it wakes. */
    sleeping(): void {
      if (!live) return;
      const row = rowOf(live.meetingId);
      save(row.id, { periods: JSON.stringify(closePeriod(row, "sleep")) });
      log.info("meeting.asleep", { meetingId: row.id });
      changed();
    },

    /** The computer woke: a Meeting it put to sleep records again, the sleep marked as a gap. */
    awake(): void {
      if (live) return;
      const row = db.get<MeetingRow>(
        `SELECT ${COLUMNS} FROM meetings WHERE deleted_at IS NULL AND status = 'recording'`,
      );
      if (!row) return;
      const periods = periodsOf(row);
      const period: Period = {
        start: now(),
        offsetMs: recordedOf(periods),
        durationMs: null,
        endedBy: null,
      };
      save(row.id, { periods: JSON.stringify([...periods, period]) });
      begin(rowOf(row.id), period);
      log.info("meeting.awake", { meetingId: row.id });
      changed();
    },

    /** The saved parts of a Meeting's audio, in time order. */
    audioParts(meetingId: string): AudioPart[] {
      return listParts(folderOf(rowOf(meetingId).id));
    },

    sweep,

    /**
     * IncarnaMind is closing: what is waiting is saved. A Meeting recording
     * stays so in the database, so the next start finds it cut off.
     */
    close(): void {
      clearInterval(sweeping);
      if (live) end(live);
    },
  };
}
