/**
 * A Meeting's Transcript (#136), made on this computer from its audio as it
 * is captured: the microphone, with the computer's echo taken out, and the
 * computer's sound, as two streams (./tracks) to the `Transcriber`, whose
 * lines arrive a few seconds behind. A line from the microphone is "Me" and
 * one from the computer's sound "Them" (#941); in a room with no computer
 * sound, "Speaker ?". The Transcript is a Document from its first line, in the
 * data folder's "Transcripts" while the Meeting records, indexed again as
 * lines come; at Stop it is finished and saved as a WebVTT file in the Output
 * location, and the Document follows it there.
 *
 * Its audio stays on disk, so nothing is lost when live text can't keep up:
 * - while the model downloads or starts, the audio waits, and what waited
 *   too long is transcribed after the Meeting stops;
 * - when live text falls too far behind, it stops, the meeting bar says
 *   "Recording…", and the rest is transcribed after the Meeting stops;
 * - when the engine's process crashes, it is started again and the stretch
 *   it lost is marked as a gap, which is filled in after the Meeting stops
 *   if its audio is still there.
 *
 * Where the User chose a cloud service (#141), the live audio goes to it
 * instead, counted against the monthly cap as it goes. A lost connection, or
 * the service refusing, hands the live audio to this computer: what the cloud
 * hadn't given text for is a gap, filled in after Stop. At the cap, the
 * cloud's streams finish what they were sent, and this computer goes on. The
 * meeting bar says so, and recording never stops.
 *
 * What is needed to pick up after a crash is in files beside the Transcript,
 * as the audio's parts are (ADR-0015): its lines and gaps in the WebVTT file,
 * and in "<meeting>.json" its Document, how far live text got and what is
 * left to transcribe.
 */
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  isTheirs,
  numberedSpeaker,
  SPEAKER_ME,
  SPEAKER_THEM,
  SPEAKER_UNKNOWN,
} from "../../shared/speakers";
import type {
  CloudLoss,
  CloudStream,
  EngineSwitch,
  Logger,
  SpeechSegment,
  Transcriber,
} from "../adapters";
import type {
  AudioSource,
  CloudTranscriptionRoute,
  Document,
  ExternalService,
  MeetingTranscript,
  SavedFile,
  TranscriptProblem,
  TranscriptState,
} from "../api";
import { hashBytes } from "../documents/files";
import {
  type Cue,
  type GapReason,
  readVtt,
  type TranscriptGap,
  writeVtt,
} from "../documents/formats/vtt";
import type { SnapshotFile } from "../documents/web";
import { FileWriteCancelledError } from "../errors";
import type { NewFile } from "../outputFiles";
import type { CloudAudioMeter, CloudTranscription } from "../transcription/cloud/routes";
import type { TranscriptionModel } from "../transcription/model";
import type { AudioPart } from "./audioFiles";
import { echoedBy, isEcho, repeats } from "./echoLines";
import type { CaptureHealthRecords } from "./health";
import { createTracks, type TrackedAudio, type Tracks } from "./tracks";

/** Live text this far behind the audio has fallen behind. */
export const BEHIND_MS = 30_000;
/** While the engine starts, at most this much audio waits for it; older audio is transcribed after Stop. */
const HELD_MS = 60_000;
/**
 * The Document is indexed again at most this often while lines come: every
 * 20 seconds at first, then less often as it grows, to two minutes, as each
 * indexing keeps a version until the next start (unless a Citation quotes it).
 */
const indexEvery = (lines: number) => Math.min(120_000, Math.max(20_000, lines * 100));
/** The engine started again this many times within `RESTARTS_WITHIN_MS` isn't started again until Stop. */
const MAX_RESTARTS = 3;
const RESTARTS_WITHIN_MS = 5 * 60_000;
/** Both streams move to another engine together: switches this close are one switch of the Meeting. */
const SWITCH_TOGETHER_MS = 10_000;
/** Audio read back from disk goes to the engine this much at a time. */
const CATCH_UP_CHUNK_MS = 20_000;
/** The speed probe counts once this much audio has been decoded live. */
const PROBE_MS = 30_000;
/** Partial text reaches the window at most this often. */
const PARTIAL_EVERY_MS = 500;
/** Cloud audio is counted against the cap this much ahead of what is sent. */
const COUNTED_AHEAD_MS = 10_000;

const SAMPLE_RATE_MS = 16;
const WAV_HEADER_BYTES = 44;

interface Range {
  fromMs: number;
  toMs: number;
}

/** What is kept in "<meeting>.json". */
interface Saved {
  documentId: string | null;
  /** How far live text got, in the Meeting's time: everything before has its final text. */
  doneMs: number;
  /** Stretches to transcribe once the Meeting stops. */
  pending: Range[];
  savedPath: string | null;
  /** Its save took the default Output location (`SavedFile.defaulted`). */
  savedDefaulted?: boolean;
  /** Stopped with nothing transcribed. */
  empty?: boolean;
  /** The number each speaker a cloud service told apart has, by source and its own label. */
  voices?: Record<string, number>;
}

interface LiveStream {
  baseMs: number;
  doneMs: number;
}

interface Live {
  stream: string;
  /** Where the live audio goes: this computer, or a cloud service. */
  engine: "local" | CloudTranscriptionRoute;
  /** The cloud service's streams and what has been counted against the cap, while `engine` is one. */
  cloud: {
    stream: CloudStream;
    meter: CloudAudioMeter;
    aheadMs: number;
    service: ExternalService;
  } | null;
  /** Where to transcribe is being worked out: the audio waits meanwhile. */
  choosing: boolean;
  tracks: Tracks;
  /** The streams opened, each with its source's first audio: the computer's sound may come later, or never. */
  streams: Partial<Record<AudioSource, LiveStream>>;
  started: boolean;
  /** Audio waiting for the engine to start. */
  held: TrackedAudio[];
  /** How far audio has gone to the engine, and how far its text is final on every stream, in the Meeting's time. */
  pushedMs: number;
  doneMs: number;
  /** Audio from here on isn't transcribed live: it is transcribed after Stop. Null while it is. */
  pendingFrom: number | null;
  /**
   * Where the audio that waited for the engine to start ends: until the engine
   * is done with it, only audio after it counts towards falling behind.
   */
  backlogEnd: number;
  callSound(): boolean;
}

interface Reading {
  baseMs: number;
  source: AudioSource;
  speaker(): string;
  /** A cloud stream lost before its last lines came. */
  lost?: boolean;
}

interface Heard {
  startMs: number;
  text: string;
  id: number;
  speaker: string;
}

interface State extends Saved {
  meetingId: string;
  cues: Cue[];
  gaps: TranscriptGap[];
  state: TranscriptState;
  problem: TranscriptProblem | null;
  partials: Map<string, Heard>;
  live: Live | null;
  /** The live streams last ended, until their last lines are in. */
  ending: Promise<void> | null;
  /** Streams reading audio back from disk, and live streams ending. */
  catchUps: Map<string, Reading>;
  restarts: number[];
  lastSwitch: (Pick<EngineSwitch, "from" | "to" | "reason"> & { atMs: number }) | null;
  fellBack: MeetingTranscript["fellBack"];
  probe: { audioMs: number; decodeMs: number; recorded: boolean };
  written: boolean;
  lastIndexed: number;
  indexTimer: ReturnType<typeof setTimeout> | null;
  partialTimer: ReturnType<typeof setTimeout> | null;
  finishing: Promise<void> | null;
}

export interface TranscriptDocuments {
  add(id: string, name: string, file: SnapshotFile): Document;
  /** Files the Document in the Mind's Folder. */
  file(documentId: string, folderId: string): void;
  check(path: string): Promise<void>;
  moveFile(id: string, path: string): Promise<void>;
  /** The live Document, or null once deleted. */
  find(id: string): Document | null;
}

export interface TranscriptsOptions {
  dataDir: string;
  transcriber: Transcriber;
  model: TranscriptionModel;
  documents: TranscriptDocuments;
  saveFile(file: NewFile): Promise<SavedFile>;
  /** "Transcript · {meeting}", in the language of the interface. */
  name(meetingId: string): string;
  /** The Folder of the Meeting's Mind, if it is in one. */
  folderOf(meetingId: string): string | null;
  /** The Meeting's audio parts on disk. */
  partsOf(meetingId: string): AudioPart[];
  /** Keeps how long decoding took for the audio it decoded, by model. */
  recordSpeed(modelId: string, ratio: number): void;
  /** Each Meeting's capture health (#161): how fast live text came, restarts, falling behind, switches. */
  health: CaptureHealthRecords;
  /** The cloud service the User chose for meeting audio (#141). Absent: always this computer. */
  cloud?: Pick<CloudTranscription, "start" | "allowed">;
  changed(): void;
  log: Logger;
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

const SOURCES: readonly AudioSource[] = ["microphone", "system"];

const streamName = (prefix: string, source: AudioSource) => `${prefix}:${source}`;

const speakerOf = (source: AudioSource, callSound: boolean) =>
  source === "system" ? SPEAKER_THEM : callSound ? SPEAKER_ME : SPEAKER_UNKNOWN;

/** The audio a stretch of tracks holds, in ms. */
const lengthOf = (audio: TrackedAudio) => audio.microphone.length / SAMPLE_RATE_MS;

/** The engine couldn't start, or stopped, while what was left was being transcribed. */
class EngineError extends Error {}

/** Adds a stretch to a list of stretches, merging those it touches. */
function addRange(ranges: Range[], range: Range): Range[] {
  if (range.toMs <= range.fromMs) return ranges;
  const all = [...ranges, range].sort((a, b) => a.fromMs - b.fromMs);
  const merged: Range[] = [];
  for (const each of all) {
    const last = merged.at(-1);
    if (last && each.fromMs <= last.toMs) last.toMs = Math.max(last.toMs, each.toMs);
    else merged.push({ ...each });
  }
  return merged;
}

/** A part's samples, from our own WAV files. */
async function samplesOf(part: AudioPart): Promise<Int16Array> {
  const bytes = await readFile(part.path);
  const body = bytes.subarray(WAV_HEADER_BYTES);
  const copy = new Uint8Array(body.byteLength - (body.byteLength % 2));
  copy.set(body.subarray(0, copy.byteLength));
  return new Int16Array(copy.buffer);
}

export function createTranscripts(options: TranscriptsOptions) {
  const { transcriber, model, documents, health, log } = options;
  const folder = join(options.dataDir, "Transcripts");
  const states = new Map<string, State>();
  let streams = 0;
  let closed = false;

  const vttPath = (meetingId: string) => join(folder, `${meetingId}.vtt`);
  const jsonPath = (meetingId: string) => join(folder, `${meetingId}.json`);

  // ---- Files ----

  function load(meetingId: string): State {
    const known = states.get(meetingId);
    if (known) return known;
    let saved: Saved = { documentId: null, doneMs: 0, pending: [], savedPath: null };
    let cues: Cue[] = [];
    let gaps: TranscriptGap[] = [];
    try {
      if (existsSync(jsonPath(meetingId))) {
        saved = { ...saved, ...(JSON.parse(readFileSync(jsonPath(meetingId), "utf8")) as Saved) };
      }
      const vtt =
        saved.savedPath && existsSync(saved.savedPath) ? saved.savedPath : vttPath(meetingId);
      if (existsSync(vtt)) ({ cues, gaps } = readVtt(readFileSync(vtt, "utf8")));
    } catch (error) {
      log.error("transcript.unreadable", { meetingId, message: messageOf(error) });
    }
    const state: State = {
      ...saved,
      meetingId,
      cues,
      gaps,
      state: saved.savedPath
        ? "saved"
        : saved.empty
          ? "empty"
          : cues.length > 0
            ? "behind"
            : "waiting",
      problem: null,
      partials: new Map(),
      live: null,
      ending: null,
      catchUps: new Map(),
      restarts: [],
      lastSwitch: null,
      fellBack: null,
      probe: { audioMs: 0, decodeMs: 0, recorded: false },
      written: cues.length > 0,
      lastIndexed: 0,
      indexTimer: null,
      partialTimer: null,
      finishing: null,
    };
    states.set(meetingId, state);
    return state;
  }

  /** Writes a file whole or not at all. */
  function writeWhole(path: string, data: string): SnapshotFile {
    mkdirSync(folder, { recursive: true });
    const bytes = Buffer.from(data, "utf8");
    const partial = `${path}.${randomUUID()}.part`;
    writeFileSync(partial, bytes);
    renameSync(partial, path);
    return { path, contentHash: hashBytes(bytes), size: bytes.byteLength, mtimeMs: Date.now() };
  }

  function persist(state: State): void {
    const saved: Saved = {
      documentId: state.documentId,
      doneMs: state.doneMs,
      pending: state.pending,
      savedPath: state.savedPath,
      ...(state.savedDefaulted && { savedDefaulted: true }),
      ...(state.state === "empty" && { empty: true }),
      ...(state.voices && { voices: state.voices }),
    };
    try {
      writeWhole(jsonPath(state.meetingId), JSON.stringify(saved));
    } catch (error) {
      log.error("transcript.saveFailed", { meetingId: state.meetingId, message: messageOf(error) });
    }
  }

  const transcriptOf = (state: State) => writeVtt({ cues: state.cues, gaps: state.gaps });

  /** Writes the lines to the data folder's file, makes the Document on the first, and indexes it again now and then. */
  function writeLines(state: State, { now = false } = {}): void {
    if (state.cues.length === 0 || state.savedPath) return;
    let file: SnapshotFile;
    try {
      file = writeWhole(vttPath(state.meetingId), transcriptOf(state));
    } catch (error) {
      log.error("transcript.writeFailed", {
        meetingId: state.meetingId,
        message: messageOf(error),
      });
      return;
    }
    state.written = true;
    persist(state);
    if (!state.documentId) {
      const id = randomUUID();
      try {
        documents.add(id, options.name(state.meetingId), file);
        const folderId = options.folderOf(state.meetingId);
        if (folderId) documents.file(id, folderId);
        state.documentId = id;
        state.lastIndexed = Date.now();
        persist(state);
      } catch (error) {
        log.error("transcript.documentFailed", {
          meetingId: state.meetingId,
          message: messageOf(error),
        });
      }
      return;
    }
    if (!documents.find(state.documentId)) return;
    const wait = state.lastIndexed + indexEvery(state.cues.length) - Date.now();
    if (now || wait <= 0) {
      if (state.indexTimer) clearTimeout(state.indexTimer);
      state.indexTimer = null;
      index(state);
    } else {
      state.indexTimer ??= setTimeout(() => {
        state.indexTimer = null;
        index(state);
      }, wait);
    }
  }

  function index(state: State): void {
    if (closed || state.savedPath) return;
    state.lastIndexed = Date.now();
    documents.check(vttPath(state.meetingId)).catch((error: unknown) => {
      log.error("transcript.indexFailed", {
        meetingId: state.meetingId,
        message: messageOf(error),
      });
    });
  }

  // ---- What the engine says ----

  function owner(stream: string): {
    state: State;
    baseMs: number;
    live: boolean;
    source: AudioSource;
    speaker: string;
    reading?: Reading;
  } | null {
    for (const state of states.values()) {
      const live = state.live;
      for (const source of SOURCES) {
        const own = live?.streams[source];
        if (live && own && stream === streamName(live.stream, source)) {
          return {
            state,
            baseMs: own.baseMs,
            live: true,
            source,
            speaker: speakerOf(source, live.callSound()),
          };
        }
      }
      const reading = state.catchUps.get(stream);
      if (reading) {
        return {
          state,
          live: false,
          baseMs: reading.baseMs,
          source: reading.source,
          speaker: reading.speaker(),
          reading,
        };
      }
    }
    return null;
  }

  /**
   * Who said a line a cloud service told apart (#141): someone numbered in the order heard in
   * the Meeting, except the User's own microphone in a call, which stays "Me".
   */
  function voiceOf(state: State, source: AudioSource, speaker: string, label: string): string {
    if (speaker === SPEAKER_ME) return speaker;
    const voices = state.voices ?? {};
    state.voices = voices;
    const key = `${source}:${label}`;
    const number = voices[key] ?? Object.keys(voices).length + 1;
    voices[key] = number;
    return numberedSpeaker(number);
  }

  function addCue(state: State, cue: Cue): void {
    if (isEcho(cue, state.cues)) return;
    // Lines caught up from disk fill in before later ones: keep them in time order.
    let at = state.cues.length;
    while (at > 0 && (state.cues[at - 1] as Cue).startMs > cue.startMs) at--;
    state.cues.splice(at, 0, cue);
    // Theirs may come after the echo of it.
    for (const echo of echoedBy(cue, state.cues)) state.cues.splice(state.cues.indexOf(echo), 1);
  }

  function shownPartial(state: State): MeetingTranscript["partial"] {
    const all = [...state.partials.values()];
    const theirs = all.filter((each) => isTheirs(each.speaker)).map((each) => each.text);
    const newest = all
      .filter((each) => each.speaker !== SPEAKER_ME || !repeats(each.text, theirs))
      .sort((a, b) => b.startMs - a.startMs)[0];
    return newest ? { startMs: newest.startMs, text: newest.text, speaker: newest.speaker } : null;
  }

  function heardSegment(segment: SpeechSegment): void {
    if (closed) return;
    const found = owner(segment.stream);
    if (!found) return;
    const { state, baseMs, live } = found;
    const speaker = segment.speaker
      ? voiceOf(state, found.source, found.speaker, segment.speaker)
      : found.speaker;
    const startMs = baseMs + segment.startMs;
    if (!segment.final) {
      if (!live || state.state !== "live") return;
      state.partials.set(segment.stream, { startMs, text: segment.text, id: segment.id, speaker });
      state.partialTimer ??= setTimeout(() => {
        state.partialTimer = null;
        options.changed();
      }, PARTIAL_EVERY_MS);
      return;
    }
    if (live && state.partials.get(segment.stream)?.id === segment.id) {
      state.partials.delete(segment.stream);
    }
    if (segment.text) {
      addCue(state, { startMs, endMs: baseMs + segment.endMs, text: segment.text, speaker });
    }
    if (segment.failed) {
      health.add(state.meetingId, {
        kind: "notTranscribed",
        atMs: startMs,
        durationMs: segment.endMs - segment.startMs,
      });
    } else if (live && state.live?.engine === "local") {
      probe(state, segment);
    }
    writeLines(state);
    options.changed();
  }

  /** The speed probe: how long decoding takes for the audio it decodes, kept for the next Meeting's choice of model. */
  function probe(state: State, segment: SpeechSegment): void {
    state.probe.audioMs += segment.endMs - segment.startMs;
    state.probe.decodeMs += segment.decodeMs;
    health.decoded(state.meetingId, segment.endMs - segment.startMs, segment.decodeMs);
    if (state.probe.recorded || state.probe.audioMs < PROBE_MS) return;
    state.probe.recorded = true;
    const ratio = state.probe.decodeMs / state.probe.audioMs;
    options.recordSpeed(model.definition.id, ratio);
    log.info("transcript.speed", {
      meetingId: state.meetingId,
      model: model.definition.id,
      ratio: Math.round(ratio * 100) / 100,
    });
  }

  function heardProgress(stream: string, doneMs: number): void {
    const found = owner(stream);
    if (!found?.live || !found.state.live) return;
    const { state } = found;
    const live = found.state.live;
    const own = live.streams[found.source];
    if (!own) return;
    own.doneMs = Math.max(own.doneMs, found.baseMs + doneMs);
    live.doneMs = Math.max(
      live.doneMs,
      Math.min(...Object.values(live.streams).map((each) => each.doneMs)),
    );
    if (live.pendingFrom === null) state.doneMs = live.doneMs;
  }

  /** The engine's process stopped: started again, the stretch it lost marked, unless it keeps stopping. */
  function engineStopped(reason: string): void {
    if (closed) return;
    log.warn("transcript.engineStopped", { reason });
    for (const state of states.values()) {
      const live = state.live;
      if (!live?.started) continue;
      const lost = { fromMs: live.doneMs, toMs: Math.max(live.doneMs, live.pushedMs) };
      if (live.pendingFrom === null && lost.toMs > lost.fromMs) {
        state.gaps.push({ ...lost, reason: "restarted" });
        state.pending = addRange(state.pending, lost);
      }
      state.partials.clear();
      const now = Date.now();
      state.restarts = [...state.restarts.filter((at) => now - at < RESTARTS_WITHIN_MS), now];
      live.started = false;
      live.stream = `${state.meetingId}:${++streams}`;
      live.streams = {};
      live.doneMs = live.pushedMs;
      if (state.restarts.length > MAX_RESTARTS) {
        live.pendingFrom ??= live.pushedMs;
        state.state = "behind";
        state.problem = { kind: "engine", message: reason };
        health.add(state.meetingId, { kind: "behind", atMs: lost.fromMs });
      } else {
        startEngine(state);
        health.add(state.meetingId, {
          kind: "restarted",
          atMs: lost.fromMs,
          gapMs: lost.toMs - lost.fromMs,
        });
      }
      writeLines(state, { now: true });
      persist(state);
    }
    options.changed();
  }

  /** The transcriber moved a stream to another engine (#141, #462): the Meeting's health says when, and why. */
  function heardSwitch(change: EngineSwitch): void {
    const found = owner(change.stream);
    if (!found) return;
    const { state } = found;
    const atMs = found.live && state.live ? state.live.pushedMs : found.baseMs;
    const { from, to, reason } = change;
    const last = state.lastSwitch;
    if (
      last?.from === from &&
      last.to === to &&
      last.reason === reason &&
      Math.abs(atMs - last.atMs) < SWITCH_TOGETHER_MS
    ) {
      return;
    }
    state.lastSwitch = { from, to, reason, atMs };
    health.add(state.meetingId, { kind: "switched", atMs, from, to, reason });
  }

  /** A cloud stream ended before it was finished: the live audio goes to this computer. */
  function heardLost(stream: string, loss: CloudLoss): void {
    if (closed) return;
    const found = owner(stream);
    if (!found) return;
    log.warn("transcript.cloudLost", { meetingId: found.state.meetingId, loss });
    if (found.reading) {
      found.reading.lost = true;
      return;
    }
    const live = found.state.live;
    if (live) leaveCloud(found.state, live, loss);
  }

  /**
   * The live audio goes to this computer from now on (#141). "dropped" and "refused": what the
   * cloud hadn't given text for is a gap, transcribed on this computer after Stop. "cap" and
   * "user": its streams finish with what they were sent. The bar says why, except where the
   * User chose.
   */
  function leaveCloud(state: State, live: Live, reason: CloudLoss | "cap" | "user"): void {
    if (live.engine === "local") return;
    const route = live.engine;
    const atMs = live.pushedMs;
    if (reason === "dropped" || reason === "refused") {
      dropStreams(live);
      const lost = { fromMs: live.doneMs, toMs: Math.max(live.doneMs, live.pushedMs) };
      if (live.started && live.pendingFrom === null && lost.toMs > lost.fromMs) {
        state.gaps.push({ ...lost, reason: "cloud" });
        state.pending = addRange(state.pending, lost);
      }
    } else if (live.started) {
      finishStreams(state, live, atMs);
    }
    live.engine = "local";
    live.cloud = null;
    live.started = false;
    live.stream = `${state.meetingId}:${++streams}`;
    live.streams = {};
    live.doneMs = live.pushedMs;
    live.backlogEnd = live.pushedMs;
    state.partials.clear();
    if (reason !== "user") state.fellBack = { route, atMs, reason };
    health.add(state.meetingId, {
      kind: "switched",
      atMs,
      from: "cloud",
      to: "local",
      reason: reason === "dropped" || reason === "refused" ? "failed" : reason,
    });
    startEngine(state);
    writeLines(state, { now: true });
    persist(state);
    options.changed();
  }

  transcriber.listen({
    segment: heardSegment,
    progress: heardProgress,
    // Not the engine's message: one from a cloud service could quote what it was sent.
    failed: (stream) => log.warn("transcript.decodeFailed", { stream }),
    stopped: engineStopped,
    switched: heardSwitch,
    lost: heardLost,
  });

  // ---- Live ----

  /** Starts the engine for the live stream, once the model is downloaded; the audio waits meanwhile. */
  function startEngine(state: State): void {
    const live = state.live;
    if (!live || live.started || live.choosing || live.pendingFrom !== null) return;
    if (live.engine !== "local") {
      startCloud(state, live);
      return;
    }
    if (model.status().state !== "ready") {
      state.state = "preparing";
      model.ensure();
      return;
    }
    if (state.state !== "behind") state.state = "preparing";
    void model.load().then((loaded) => {
      if (closed || state.live !== live) return;
      if (!loaded) {
        live.pendingFrom ??= live.held[0]?.atMs ?? live.pushedMs;
        live.held = [];
        state.state = "behind";
        const message = model.status().error?.message ?? "";
        log.warn("transcript.engineFailed", { meetingId: state.meetingId, message });
        state.problem = { kind: "engine", message };
        options.changed();
        return;
      }
      live.started = true;
      state.state = "live";
      state.problem = null;
      const held = live.held;
      live.held = [];
      for (const audio of held) pushTracks(state, live, audio);
      live.backlogEnd = live.pushedMs;
      options.changed();
    });
  }

  /** The cloud service takes the live audio: its streams open as each source's audio first comes. */
  function startCloud(state: State, live: Live): void {
    live.started = true;
    state.state = "live";
    state.problem = null;
    const held = live.held;
    live.held = [];
    for (const audio of held) {
      if (!live.started) {
        live.held.push(audio);
        continue;
      }
      pushTracks(state, live, audio);
    }
    live.backlogEnd = live.pushedMs;
    options.changed();
  }

  /** Counts cloud audio against the cap ahead of sending it; false at the cap. */
  function counted(live: Live, audio: TrackedAudio): boolean {
    const cloud = live.cloud;
    if (!cloud) return true;
    const sources = SOURCES.filter((source) => audio[source]).length;
    const ms = lengthOf(audio) * sources;
    while (cloud.aheadMs < ms) {
      if (!cloud.meter.add(COUNTED_AHEAD_MS)) return false;
      cloud.aheadMs += COUNTED_AHEAD_MS;
    }
    cloud.aheadMs -= ms;
    return true;
  }

  function pushTracks(state: State, live: Live, audio: TrackedAudio): void {
    if (live.engine !== "local") {
      const allowed = (live.cloud && options.cloud?.allowed(live.cloud.service)) ?? false;
      if (!allowed || !counted(live, audio)) {
        leaveCloud(state, live, allowed ? "cap" : "user");
        if (live.started) pushTracks(state, live, audio);
        else live.held.push(audio);
        return;
      }
    }
    for (const source of SOURCES) {
      const samples = audio[source];
      if (!samples) continue;
      const name = streamName(live.stream, source);
      if (!live.streams[source]) {
        live.streams[source] = { baseMs: audio.atMs, doneMs: audio.atMs };
        if (live.cloud) transcriber.open(name, live.cloud.stream);
      }
      transcriber.push(name, samples);
    }
    live.pushedMs = audio.atMs + lengthOf(audio);
  }

  function dropStreams(live: Live): void {
    for (const source of SOURCES) {
      if (live.streams[source]) transcriber.drop(streamName(live.stream, source));
    }
  }

  function send(state: State, live: Live, audio: TrackedAudio): void {
    if (!live.started || live.pendingFrom !== null) return;
    pushTracks(state, live, audio);
    if (!live.started) return;
    if (live.pushedMs - Math.max(live.doneMs, live.backlogEnd) > BEHIND_MS) fallBehind(state, live);
  }

  /** Live text stops; the rest is transcribed after Stop. */
  function fallBehind(state: State, live: Live): void {
    dropStreams(live);
    live.stream = `${state.meetingId}:${++streams}`;
    live.streams = {};
    live.started = false;
    live.pendingFrom = live.doneMs;
    state.state = "behind";
    state.partials.clear();
    health.add(state.meetingId, { kind: "behind", atMs: live.doneMs });
    persist(state);
    options.changed();
  }

  /** Works out where a stretch of recording is transcribed, the audio waiting meanwhile. */
  function choose(state: State, live: Live, cloud: NonNullable<TranscriptsOptions["cloud"]>): void {
    cloud.start().then(
      (start) => {
        if (closed || state.live !== live) return;
        live.choosing = false;
        if (start && "refused" in start) {
          state.fellBack = { route: start.route, atMs: live.pushedMs, reason: "cap" };
          health.add(state.meetingId, {
            kind: "switched",
            atMs: live.pushedMs,
            from: "cloud",
            to: "local",
            reason: "cap",
          });
        } else if (start) {
          live.engine = start.route;
          live.cloud = {
            stream: start.stream,
            meter: start.meter,
            aheadMs: 0,
            service: start.service,
          };
          log.info("transcript.cloud", { meetingId: state.meetingId, route: start.route });
        }
        startEngine(state);
        options.changed();
      },
      (error: unknown) => {
        if (closed || state.live !== live) return;
        live.choosing = false;
        log.warn("transcript.cloudUnavailable", {
          meetingId: state.meetingId,
          message: messageOf(error),
        });
        startEngine(state);
        options.changed();
      },
    );
  }

  function feed(state: State, audio: TrackedAudio | null): void {
    const live = state.live;
    if (!audio || !live || live.pendingFrom !== null) return;
    if (live.started) {
      send(state, live, audio);
      return;
    }
    live.held.push(audio);
    // Audio that waited too long is transcribed after Stop: live text starts from what is left.
    const newest = audio.atMs + audio.microphone.length / SAMPLE_RATE_MS;
    while ((live.held[0]?.atMs ?? newest) < newest - HELD_MS) {
      const old = live.held.shift() as TrackedAudio;
      const end = old.atMs + old.microphone.length / SAMPLE_RATE_MS;
      state.pending = addRange(state.pending, { fromMs: old.atMs, toMs: end });
      live.doneMs = end;
      live.pushedMs = end;
    }
  }

  model.onReady(() => {
    for (const state of states.values()) {
      if (state.live && !state.live.started) startEngine(state);
      if (state.state === "finishing" && !state.finishing) state.finishing = finish(state);
    }
    options.changed();
  });

  // ---- Finishing ----

  /** Transcribes a stretch from the audio on disk, each source on its own; the part with no audio left is a gap. */
  async function catchUp(state: State, range: Range): Promise<void> {
    const parts = options
      .partsOf(state.meetingId)
      .filter(
        (part) => part.offsetMs < range.toMs && part.offsetMs + part.durationMs > range.fromMs,
      );
    let covered: Range[] = [];
    for (const part of parts) {
      covered = addRange(covered, { fromMs: part.offsetMs, toMs: part.offsetMs + part.durationMs });
    }
    let at = range.fromMs;
    for (const each of [...covered, { fromMs: range.toMs, toMs: range.toMs }]) {
      if (each.fromMs > at)
        state.gaps.push({ fromMs: at, toMs: Math.min(each.fromMs, range.toMs), reason: "lost" });
      at = Math.max(at, each.toMs);
    }
    if (parts.length === 0) return;
    const prefix = `${state.meetingId}:catch-up:${++streams}`;
    let callSound = false;
    const opened: string[] = [];
    try {
      const tracks = createTracks(range.fromMs);
      const end = Math.min(
        range.toMs,
        Math.max(...parts.map((part) => part.offsetMs + part.durationMs)),
      );
      for (let from = range.fromMs; from < end; from += CATCH_UP_CHUNK_MS) {
        const to = Math.min(end, from + CATCH_UP_CHUNK_MS);
        for (const part of parts) {
          const partEnd = part.offsetMs + part.durationMs;
          if (part.offsetMs >= to || partEnd <= from) continue;
          const samples = await samplesOf(part);
          const first = Math.max(0, Math.round((from - part.offsetMs) * SAMPLE_RATE_MS));
          const last = Math.min(samples.length, Math.round((to - part.offsetMs) * SAMPLE_RATE_MS));
          if (last > first) {
            tracks.put(
              part.source,
              samples.subarray(first, last),
              part.offsetMs + first / SAMPLE_RATE_MS,
            );
          }
        }
        const audio = tracks.flush();
        callSound ||= audio?.system?.some((sample) => sample !== 0) ?? false;
        for (const source of SOURCES) {
          const track = audio?.[source];
          if (!audio || !track) continue;
          const stream = streamName(prefix, source);
          if (!state.catchUps.has(stream)) {
            state.catchUps.set(stream, {
              baseMs: audio.atMs,
              source,
              speaker: () => speakerOf(source, callSound),
            });
            opened.push(stream);
          }
          transcriber.push(stream, track);
        }
      }
      await Promise.all(opened.map((stream) => transcriber.finish(stream))).catch(
        (error: unknown) => {
          throw new EngineError(messageOf(error));
        },
      );
    } finally {
      for (const stream of opened) state.catchUps.delete(stream);
    }
  }

  /** Whether a gap's stretch has been transcribed since: its lines fill it. */
  const filled = (gap: TranscriptGap, done: readonly Range[]) =>
    (gap.reason === "restarted" || gap.reason === "cloud") &&
    done.some((range) => range.fromMs <= gap.fromMs && range.toMs >= gap.toMs);

  /**
   * After Stop: the live stream's last lines, then what was left to
   * transcribe, then the WebVTT file saved to the Output location, where the
   * Document follows it.
   */
  async function finish(state: State): Promise<void> {
    try {
      if (state.pending.length > 0) {
        if (model.status().state !== "ready") {
          model.ensure();
          state.finishing = null;
          options.changed();
          return; // goes on when the model is ready (`onReady`)
        }
        if (!(await model.load())) throw new EngineError(model.status().error?.message ?? "");
        const done: Range[] = [];
        for (const range of [...state.pending]) {
          await catchUp(state, range);
          done.push(range);
          state.pending = state.pending.filter((each) => each !== range);
          state.gaps = state.gaps.filter((gap) => !filled(gap, done));
          writeLines(state);
          persist(state);
          options.changed();
        }
      }
      await save(state);
    } catch (error) {
      state.state = "failed";
      state.problem = {
        kind: error instanceof EngineError ? "engine" : "save",
        message: error instanceof FileWriteCancelledError ? "" : messageOf(error),
      };
      log.warn("transcript.finishFailed", {
        meetingId: state.meetingId,
        message: messageOf(error),
      });
    } finally {
      state.finishing = null;
      if (!closed) {
        persist(state);
        options.changed();
      }
    }
  }

  async function save(state: State): Promise<void> {
    if (state.cues.length === 0) {
      state.state = "empty";
      return;
    }
    const document = state.documentId ? documents.find(state.documentId) : null;
    const name = document?.name ?? options.name(state.meetingId);
    const data = new TextEncoder().encode(transcriptOf(state));
    const saved = await options.saveFile({ name: `${name}.vtt`, data, ownFile: null });
    if (document) await documents.moveFile(document.id, saved.path);
    rmSync(vttPath(state.meetingId), { force: true });
    state.savedPath = saved.path;
    state.savedDefaulted = saved.defaulted;
    state.state = "saved";
    state.problem = null;
    log.info("transcript.saved", { meetingId: state.meetingId, lines: state.cues.length });
  }

  /** The live streams give their last lines, up to `endMs`; they are the Meeting's until then. */
  function finishStreams(state: State, live: Live, endMs: number): void {
    const callSound = live.callSound();
    const fromMs = live.doneMs;
    const ending = SOURCES.flatMap((source) => {
      const own = live.streams[source];
      return own
        ? [
            {
              stream: streamName(live.stream, source),
              reading: {
                baseMs: own.baseMs,
                source,
                speaker: () => speakerOf(source, callSound),
              } as Reading,
            },
          ]
        : [];
    });
    for (const { stream, reading } of ending) state.catchUps.set(stream, reading);
    const before = state.ending;
    const done: Promise<void> = Promise.all([
      before,
      ...ending.map(({ stream }) => transcriber.finish(stream)),
    ])
      .then(
        () => {
          if (ending.some(({ reading }) => reading.lost)) {
            state.pending = addRange(state.pending, { fromMs, toMs: endMs });
          } else state.doneMs = Math.max(state.doneMs, endMs);
        },
        (error: unknown) => {
          log.warn("transcript.endFailed", {
            meetingId: state.meetingId,
            message: messageOf(error),
          });
          state.pending = addRange(state.pending, { fromMs, toMs: endMs });
        },
      )
      .finally(() => {
        for (const { stream } of ending) state.catchUps.delete(stream);
        if (state.ending === done) state.ending = null;
        if (closed) return;
        writeLines(state, { now: true });
        persist(state);
      });
    state.ending = done;
  }

  function endLive(state: State, endMs: number): void {
    const live = state.live;
    if (!live) return;
    feed(state, live.tracks.flush());
    state.live = null;
    state.partials.clear();
    if (live.started && live.pendingFrom === null) {
      finishStreams(state, live, endMs);
    } else {
      state.pending = addRange(state.pending, {
        fromMs: live.pendingFrom ?? live.doneMs,
        toMs: endMs,
      });
    }
    persist(state);
    options.changed();
  }

  return {
    /**
     * Indexes a recording Meeting's Transcript again now, for a Question asked during the call
     * (#144): what was said up to now can be found and cited. Resolves once its file is checked
     * (a changed one is then processed). Nothing for a Meeting with no lines or one saved.
     */
    async refresh(meetingId: string): Promise<void> {
      const state = states.get(meetingId);
      if (closed || !state?.documentId || state.savedPath || state.cues.length === 0) return;
      if (state.indexTimer) clearTimeout(state.indexTimer);
      state.indexTimer = null;
      state.lastIndexed = Date.now();
      await documents.check(vttPath(meetingId));
    },

    /**
     * The newest line said, by `speaker` when given (a live line's "Me" or "Them", which is
     * anyone told apart too), for the heading over help during the call; null before there is one.
     */
    newestLine(
      meetingId: string,
      speaker?: string,
    ): { text: string; speaker: string | null } | null {
      const cues = states.get(meetingId)?.cues ?? [];
      const said = (each: Cue) =>
        speaker === undefined ||
        each.speaker === speaker ||
        (speaker === SPEAKER_THEM && isTheirs(each.speaker));
      const cue = cues.findLast(said);
      return cue ? { text: cue.text, speaker: cue.speaker } : null;
    },

    /** The Meeting's Transcript, as the window shows it. */
    of(meetingId: string): MeetingTranscript {
      const state = load(meetingId);
      const status = model.status();
      return {
        documentId: state.documentId,
        state: state.state,
        lines: state.cues.length,
        partial: shownPartial(state),
        model:
          (state.state === "preparing" ||
            state.state === "finishing" ||
            state.problem?.kind === "download") &&
          status.state !== "ready"
            ? status
            : null,
        cloud:
          state.live && state.live.engine !== "local" && state.live.started
            ? state.live.engine
            : null,
        fellBack: state.fellBack,
        savedPath: state.savedPath,
        savedDefaulted: state.savedDefaulted === true,
        problem:
          state.problem ??
          (status.state === "failed" && (state.state === "preparing" || state.state === "finishing")
            ? {
                kind: status.error?.kind === "load" ? "engine" : "download",
                message: status.error?.message ?? "",
              }
            : null),
      };
    },

    /**
     * Audio starts being captured, at `offsetMs` in the Meeting's time. `callSound` says,
     * whenever asked, whether the computer's sound is being recorded.
     */
    begin(meetingId: string, offsetMs: number, callSound: () => boolean): void {
      const state = load(meetingId);
      if (state.savedPath) return;
      const live: Live = {
        stream: `${meetingId}:${++streams}`,
        engine: "local",
        cloud: null,
        choosing: options.cloud !== undefined,
        tracks: createTracks(offsetMs),
        streams: {},
        started: false,
        held: [],
        pushedMs: offsetMs,
        doneMs: offsetMs,
        pendingFrom: null,
        backlogEnd: offsetMs,
        callSound,
      };
      state.live = live;
      state.fellBack = null;
      state.problem = state.problem?.kind === "engine" ? state.problem : null;
      state.state = "preparing";
      if (options.cloud) choose(state, live, options.cloud);
      else startEngine(state);
      options.changed();
    },

    /** Samples from a source, starting at `atMs` in the Meeting's time. */
    hear(meetingId: string, source: AudioSource, samples: Int16Array, atMs: number): void {
      const state = states.get(meetingId);
      if (state?.live) feed(state, state.live.tracks.add(source, samples, atMs));
    },

    /** Audio stopped being captured, at `endMs` in the Meeting's time: paused, asleep or stopped. */
    end(meetingId: string, endMs: number): void {
      const state = states.get(meetingId);
      if (state) endLive(state, endMs);
    },

    /** The Meeting stopped: what is left is transcribed, and the file saved. */
    stop(meetingId: string): void {
      const state = load(meetingId);
      if (state.savedPath || state.finishing) return;
      if (state.cues.length === 0 && state.pending.length === 0 && !state.ending) {
        state.state = "empty";
        persist(state);
        options.changed();
        return;
      }
      state.state = "finishing";
      state.finishing = (state.ending ?? Promise.resolve()).then(() => finish(state));
      options.changed();
    },

    /**
     * IncarnaMind closed while the Meeting recorded: what was recorded after
     * live text got is transcribed once it stops.
     */
    recovered(meetingId: string, recordedMs: number): void {
      const state = load(meetingId);
      if (state.savedPath) return;
      const fromMs = Math.max(state.doneMs, state.cues.at(-1)?.endMs ?? 0);
      state.pending = addRange(state.pending, { fromMs, toMs: recordedMs });
      if (state.cues.length > 0 || state.pending.length > 0) state.state = "behind";
      persist(state);
    },

    /** A stopped Meeting whose Transcript wasn't finished, when IncarnaMind started: it is finished now. */
    resume(meetingId: string): void {
      const state = load(meetingId);
      if (state.savedPath || state.state === "empty") return;
      if (state.cues.length === 0 && state.pending.length === 0) return;
      state.state = "finishing";
      state.finishing = finish(state);
    },

    retry(meetingId: string): void {
      const state = load(meetingId);
      const status = model.status();
      if (status.state === "failed") model.retry();
      state.problem = null;
      state.restarts = [];
      const live = state.live;
      if (live && !live.started) {
        if (live.pendingFrom !== null) {
          state.pending = addRange(state.pending, {
            fromMs: live.pendingFrom,
            toMs: live.tracks.atMs,
          });
          live.stream = `${meetingId}:${++streams}`;
          live.streams = {};
          live.pushedMs = live.tracks.atMs;
          live.doneMs = live.tracks.atMs;
          live.pendingFrom = null;
        }
        startEngine(state);
      } else if ((state.state === "failed" || state.state === "finishing") && !state.finishing) {
        state.state = "finishing";
        state.finishing = finish(state);
      }
      options.changed();
    },

    /** The Meeting's Mind was deleted: its Transcript's working files go; its Document stays. */
    forget(meetingId: string): void {
      const state = states.get(meetingId);
      if (state?.live) {
        dropStreams(state.live);
        state.live = null;
      }
      if (state?.indexTimer) clearTimeout(state.indexTimer);
      states.delete(meetingId);
      rmSync(jsonPath(meetingId), { force: true });
    },

    close(): void {
      closed = true;
      for (const state of states.values()) {
        if (state.indexTimer) clearTimeout(state.indexTimer);
        if (state.partialTimer) clearTimeout(state.partialTimer);
        if (state.live) persist(state);
      }
      model.close();
    },
  };
}

export type Transcripts = ReturnType<typeof createTranscripts>;

/** Why a stretch has no text, as the reasons of the gaps a Transcript marks. */
export type { GapReason };
