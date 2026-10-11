/**
 * A Meeting's Transcript (#136), made on this computer from its audio as it
 * is captured: the microphone and the computer's sound mixed into one track
 * (./mixer) and streamed to the `Transcriber`, whose lines arrive a few
 * seconds behind. The Transcript is a Document from its first line, in the
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
 * What is needed to pick up after a crash is in files beside the Transcript,
 * as the audio's parts are (ADR-0015): its lines and gaps in the WebVTT file,
 * and in "<meeting>.json" its Document, how far live text got and what is
 * left to transcribe.
 */
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { Logger, SpeechSegment, Transcriber } from "../adapters";
import type {
  Document,
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
import type { TranscriptionModel } from "../transcription/model";
import type { AudioPart } from "./audioFiles";
import { createMixer, type MixedAudio, type Mixer } from "./mixer";

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
/** Audio read back from disk goes to the engine this much at a time. */
const CATCH_UP_CHUNK_MS = 20_000;
/** The speed probe counts once this much audio has been decoded live. */
const PROBE_MS = 30_000;
/** Partial text reaches the window at most this often. */
const PARTIAL_EVERY_MS = 500;

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
  /** Stopped with nothing transcribed. */
  empty?: boolean;
}

interface Live {
  stream: string;
  /** The Meeting's time at the stream's first sample. */
  baseMs: number;
  mixer: Mixer;
  /** The engine takes this stream's audio. */
  started: boolean;
  /** Audio waiting for the engine to start. */
  held: MixedAudio[];
  /** How far audio has gone to the engine, and how far its text is final, in the Meeting's time. */
  pushedMs: number;
  doneMs: number;
  /** Audio from here on isn't transcribed live: it is transcribed after Stop. Null while it is. */
  pendingFrom: number | null;
  /**
   * Where the audio that waited for the engine to start ends: until the engine
   * is done with it, only audio after it counts towards falling behind.
   */
  backlogEnd: number;
}

interface State extends Saved {
  meetingId: string;
  cues: Cue[];
  gaps: TranscriptGap[];
  state: TranscriptState;
  problem: TranscriptProblem | null;
  partial: { startMs: number; text: string; id: number } | null;
  live: Live | null;
  /** The live stream last ended, until its last lines are in. */
  ending: Promise<void> | null;
  /** Streams reading audio back from disk, and live streams ending: their start in the Meeting's time. */
  catchUps: Map<string, number>;
  restarts: number[];
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
  changed(): void;
  log: Logger;
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

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
  const { transcriber, model, documents, log } = options;
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
      partial: null,
      live: null,
      ending: null,
      catchUps: new Map(),
      restarts: [],
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
      ...(state.state === "empty" && { empty: true }),
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

  /** The stream's Meeting, and where the stream starts in the Meeting's time. */
  function owner(stream: string): { state: State; baseMs: number; live: boolean } | null {
    for (const state of states.values()) {
      if (state.live?.stream === stream) return { state, baseMs: state.live.baseMs, live: true };
      const baseMs = state.catchUps.get(stream);
      if (baseMs !== undefined) return { state, baseMs, live: false };
    }
    return null;
  }

  function addCue(state: State, cue: Cue): void {
    // Lines caught up from disk fill in before later ones: keep them in time order.
    let at = state.cues.length;
    while (at > 0 && (state.cues[at - 1] as Cue).startMs > cue.startMs) at--;
    state.cues.splice(at, 0, cue);
  }

  function heardSegment(segment: SpeechSegment): void {
    if (closed) return;
    const found = owner(segment.stream);
    if (!found) return;
    const { state, baseMs, live } = found;
    const startMs = baseMs + segment.startMs;
    if (!segment.final) {
      if (!live || state.state !== "live") return;
      state.partial = { startMs, text: segment.text, id: segment.id };
      state.partialTimer ??= setTimeout(() => {
        state.partialTimer = null;
        options.changed();
      }, PARTIAL_EVERY_MS);
      return;
    }
    if (live && state.partial?.id === segment.id) state.partial = null;
    if (segment.text)
      addCue(state, { startMs, endMs: baseMs + segment.endMs, text: segment.text, speaker: null });
    if (live) probe(state, segment);
    writeLines(state);
    options.changed();
  }

  /** The speed probe: how long decoding takes for the audio it decodes, kept for the next Meeting's choice of model. */
  function probe(state: State, segment: SpeechSegment): void {
    state.probe.audioMs += segment.endMs - segment.startMs;
    state.probe.decodeMs += segment.decodeMs;
    if (state.probe.recorded || state.probe.audioMs < PROBE_MS) return;
    state.probe.recorded = true;
    const ratio = state.probe.decodeMs / state.probe.audioMs;
    options.recordSpeed(model.definition.id, ratio);
    log.info("transcript.speed", {
      model: model.definition.id,
      ratio: Math.round(ratio * 100) / 100,
    });
  }

  function heardProgress(stream: string, doneMs: number): void {
    const found = owner(stream);
    if (!found?.live || !found.state.live) return;
    const { state } = found;
    const live = found.state.live;
    live.doneMs = Math.max(live.doneMs, found.baseMs + doneMs);
    if (live.pendingFrom === null) state.doneMs = live.doneMs;
  }

  /** The engine's process stopped: started again, the stretch it lost marked, unless it keeps stopping. */
  function engineStopped(reason: string): void {
    if (closed) return;
    log.warn("transcript.engineStopped", { reason });
    for (const state of states.values()) {
      const live = state.live;
      if (!live?.started) continue;
      if (live.pendingFrom === null) {
        const lost = { fromMs: live.doneMs, toMs: Math.max(live.doneMs, live.pushedMs) };
        if (lost.toMs > lost.fromMs) {
          state.gaps.push({ ...lost, reason: "restarted" });
          state.pending = addRange(state.pending, lost);
        }
      }
      state.partial = null;
      const now = Date.now();
      state.restarts = [...state.restarts.filter((at) => now - at < RESTARTS_WITHIN_MS), now];
      live.started = false;
      live.stream = `${state.meetingId}:${++streams}`;
      live.baseMs = live.pushedMs;
      live.doneMs = live.pushedMs;
      if (state.restarts.length > MAX_RESTARTS) {
        live.pendingFrom ??= live.pushedMs;
        state.state = "behind";
        state.problem = { kind: "engine", message: reason };
      } else {
        startEngine(state);
      }
      writeLines(state, { now: true });
      persist(state);
    }
    options.changed();
  }

  transcriber.listen({
    segment: heardSegment,
    progress: heardProgress,
    failed: (stream, message) => log.warn("transcript.decodeFailed", { stream, message }),
    stopped: engineStopped,
  });

  // ---- Live ----

  /** Starts the engine for the live stream, once the model is downloaded; the audio waits meanwhile. */
  function startEngine(state: State): void {
    const live = state.live;
    if (!live || live.started || live.pendingFrom !== null) return;
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
        state.problem = { kind: "engine", message: model.status().error?.message ?? "" };
        options.changed();
        return;
      }
      live.started = true;
      state.state = "live";
      state.problem = null;
      const held = live.held;
      live.held = [];
      for (const audio of held) {
        transcriber.push(live.stream, audio.samples);
        live.pushedMs = audio.atMs + audio.samples.length / SAMPLE_RATE_MS;
      }
      live.backlogEnd = live.pushedMs;
      options.changed();
    });
  }

  function send(state: State, live: Live, audio: MixedAudio): void {
    if (!live.started || live.pendingFrom !== null) return;
    transcriber.push(live.stream, audio.samples);
    live.pushedMs = audio.atMs + audio.samples.length / SAMPLE_RATE_MS;
    if (live.pushedMs - Math.max(live.doneMs, live.backlogEnd) > BEHIND_MS) fallBehind(state, live);
  }

  /** Live text stops; the rest is transcribed after Stop. */
  function fallBehind(state: State, live: Live): void {
    transcriber.drop(live.stream);
    live.stream = `${state.meetingId}:${++streams}`;
    live.started = false;
    live.pendingFrom = live.doneMs;
    state.state = "behind";
    state.partial = null;
    log.info("transcript.behind", { meetingId: state.meetingId });
    persist(state);
    options.changed();
  }

  function feed(state: State, audio: MixedAudio | null): void {
    const live = state.live;
    if (!audio || !live || live.pendingFrom !== null) return;
    if (live.started) {
      send(state, live, audio);
      return;
    }
    live.held.push(audio);
    // Audio that waited too long is transcribed after Stop: live text starts from what is left.
    const newest = audio.atMs + audio.samples.length / SAMPLE_RATE_MS;
    while ((live.held[0]?.atMs ?? newest) < newest - HELD_MS) {
      const old = live.held.shift() as MixedAudio;
      const end = old.atMs + old.samples.length / SAMPLE_RATE_MS;
      state.pending = addRange(state.pending, { fromMs: old.atMs, toMs: end });
      live.doneMs = end;
      live.pushedMs = end;
      live.baseMs = end;
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

  /** Transcribes a stretch from the audio on disk, both sources mixed; the part with no audio left is a gap. */
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
    const stream = `${state.meetingId}:catch-up:${++streams}`;
    state.catchUps.set(stream, range.fromMs);
    try {
      const mixer = createMixer(range.fromMs);
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
            mixer.put(samples.subarray(first, last), part.offsetMs + first / SAMPLE_RATE_MS);
          }
        }
        const audio = mixer.flush();
        if (audio) transcriber.push(stream, audio.samples);
      }
      await transcriber.finish(stream).catch((error: unknown) => {
        throw new EngineError(messageOf(error));
      });
    } finally {
      state.catchUps.delete(stream);
    }
  }

  /** Whether a gap's stretch has been transcribed since: its lines fill it. */
  const filled = (gap: TranscriptGap, done: readonly Range[]) =>
    gap.reason === "restarted" &&
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
    state.state = "saved";
    state.problem = null;
    log.info("transcript.saved", { meetingId: state.meetingId, lines: state.cues.length });
  }

  function endLive(state: State, endMs: number): void {
    const live = state.live;
    if (!live) return;
    feed(state, live.mixer.flush());
    state.live = null;
    state.partial = null;
    if (live.started && live.pendingFrom === null) {
      // The stream is the Meeting's until its last lines are in.
      state.catchUps.set(live.stream, live.baseMs);
      state.ending = transcriber
        .finish(live.stream)
        .then(
          () => {
            state.doneMs = Math.max(state.doneMs, endMs);
          },
          (error: unknown) => {
            log.warn("transcript.endFailed", {
              meetingId: state.meetingId,
              message: messageOf(error),
            });
            state.pending = addRange(state.pending, { fromMs: live.doneMs, toMs: endMs });
          },
        )
        .finally(() => {
          state.catchUps.delete(live.stream);
          state.ending = null;
          if (closed) return;
          writeLines(state, { now: true });
          persist(state);
        });
    } else {
      state.pending = addRange(state.pending, {
        fromMs: live.pendingFrom ?? live.baseMs,
        toMs: endMs,
      });
    }
    persist(state);
    options.changed();
  }

  return {
    /** The Meeting's Transcript, as the window shows it. */
    of(meetingId: string): MeetingTranscript {
      const state = load(meetingId);
      const status = model.status();
      return {
        documentId: state.documentId,
        state: state.state,
        lines: state.cues.length,
        partial: state.partial
          ? { startMs: state.partial.startMs, text: state.partial.text }
          : null,
        model:
          (state.state === "preparing" ||
            state.state === "finishing" ||
            state.problem?.kind === "download") &&
          status.state !== "ready"
            ? status
            : null,
        savedPath: state.savedPath,
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

    /** Audio starts being captured, at `offsetMs` in the Meeting's time. */
    begin(meetingId: string, offsetMs: number): void {
      const state = load(meetingId);
      if (state.savedPath) return;
      state.live = {
        stream: `${meetingId}:${++streams}`,
        baseMs: offsetMs,
        mixer: createMixer(offsetMs),
        started: false,
        held: [],
        pushedMs: offsetMs,
        doneMs: offsetMs,
        pendingFrom: null,
        backlogEnd: offsetMs,
      };
      state.problem = state.problem?.kind === "engine" ? state.problem : null;
      state.state = "preparing";
      startEngine(state);
      options.changed();
    },

    /** Samples from a source, starting at `atMs` in the Meeting's time. */
    hear(meetingId: string, samples: Int16Array, atMs: number): void {
      const state = states.get(meetingId);
      if (state?.live) feed(state, state.live.mixer.add(samples, atMs));
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
            toMs: live.mixer.atMs,
          });
          live.stream = `${meetingId}:${++streams}`;
          live.baseMs = live.mixer.atMs;
          live.pushedMs = live.mixer.atMs;
          live.doneMs = live.mixer.atMs;
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
        transcriber.drop(state.live.stream);
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
