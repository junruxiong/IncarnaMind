/**
 * Telling the voices in a recording apart (#138), with sherpa-onnx-node's speaker diarization:
 * pyannote's segmentation model finds where each voice speaks, 3D-Speaker's CAM++ gives each
 * stretch a voiceprint, and the stretches are clustered into voices. Its `process` holds its
 * thread until it is done, so the desktop app runs it in a process of its own
 * (src/main/speakerProcess.ts).
 */
import { readFileSync } from "node:fs";
import type { SpeakerEmbeddingExtractor } from "sherpa-onnx-node";
import { MEETING_SAMPLE_RATE, type SpeakerModelFiles, type VoiceTurn } from "../adapters";
import { loadSherpa } from "./sherpa";

/** Voiceprints closer than this are one voice: set on the first 15 minutes of the English cases of eval/meetings (eval/speakers/README.md). */
export const CLUSTER_THRESHOLD = 1.05;

/**
 * Voices whose voiceprints are at least this alike are taken for one person afterwards: over a
 * long meeting the engine's clustering splits people into several voices.
 */
export const SAME_PERSON = 0.6;
/** A voice with less of the speech than this is someone else's stray stretch: it joins the voice most like it. */
const STRAY_SHARE = 0.02;
/** A voiceprint is taken from a voice's longest stretches, up to this much of them. */
const PRINT_SECONDS = 30;

const WAV_HEADER_BYTES = 44;
const SAMPLES_PER_MS = MEETING_SAMPLE_RATE / 1000;

export interface SeparationRequest {
  parts: readonly { path: string; offsetMs: number }[];
  model: SpeakerModelFiles;
}

interface Span {
  /** Where it starts in the joined samples. */
  at: number;
  length: number;
  offsetMs: number;
}

/** A part's samples, from our own WAV files, as the engine reads them: -1 to 1. */
function samplesOf(path: string): Float32Array {
  const bytes = readFileSync(path);
  const count = Math.floor((bytes.byteLength - WAV_HEADER_BYTES) / 2);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    samples[i] = bytes.readInt16LE(WAV_HEADER_BYTES + i * 2) / 32768;
  }
  return samples;
}

/** The parts' samples one after the other, in the Meeting's order, and where each came from. */
function joinParts(parts: SeparationRequest["parts"]): { samples: Float32Array; spans: Span[] } {
  const read = [...parts]
    .sort((a, b) => a.offsetMs - b.offsetMs)
    .map((part) => ({ offsetMs: part.offsetMs, samples: samplesOf(part.path) }));
  const samples = new Float32Array(read.reduce((sum, part) => sum + part.samples.length, 0));
  const spans: Span[] = [];
  let at = 0;
  for (const part of read) {
    samples.set(part.samples, at);
    spans.push({ at, length: part.samples.length, offsetMs: part.offsetMs });
    at += part.samples.length;
  }
  return { samples, spans };
}

/**
 * The engine's segments (seconds of the joined samples, its own speaker numbers) in the
 * Meeting's time, cut where the recording paused, with voices numbered in the order first heard.
 */
export function toMeetingTime(
  segments: readonly { start: number; end: number; speaker: number }[],
  spans: readonly Span[],
): VoiceTurn[] {
  const order = new Map<number, number>();
  const turns: VoiceTurn[] = [];
  for (const segment of [...segments].sort((a, b) => a.start - b.start)) {
    const voice = order.get(segment.speaker) ?? order.size;
    order.set(segment.speaker, voice);
    const from = segment.start * MEETING_SAMPLE_RATE;
    const to = segment.end * MEETING_SAMPLE_RATE;
    for (const span of spans) {
      const start = Math.max(from, span.at);
      const end = Math.min(to, span.at + span.length);
      if (end <= start) continue;
      const startMs = span.offsetMs + (start - span.at) / SAMPLES_PER_MS;
      const endMs = span.offsetMs + (end - span.at) / SAMPLES_PER_MS;
      const last = turns.at(-1);
      if (last && last.voice === voice && Math.abs(last.endMs - startMs) < 1) last.endMs = endMs;
      else turns.push({ startMs, endMs, voice });
    }
  }
  return turns;
}

interface Segment {
  start: number;
  end: number;
  speaker: number;
}

const cosine = (a: Float32Array, b: Float32Array) => {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    aa += x * x;
    bb += y * y;
  }
  return aa === 0 || bb === 0 ? 0 : dot / Math.sqrt(aa * bb);
};

interface Group {
  ids: number[];
  print: Float32Array;
  seconds: number;
}

/** `from` joins `into`: its voiceprint becomes their speech-weighted mean. */
function join(into: Group, from: Group): void {
  const total = into.seconds + from.seconds || 1;
  into.print = into.print.map(
    (value, i) => (value * into.seconds + (from.print[i] ?? 0) * from.seconds) / total,
  );
  into.seconds += from.seconds;
  into.ids.push(...from.ids);
}

/**
 * Which voice each voice becomes once alike voices are one person: the two most alike join
 * while they are at least `samePerson` alike; then each voice with less than `STRAY_SHARE` of
 * the speech joins the voice most like it.
 */
export function joinedVoices(
  voices: ReadonlyMap<number, { print: Float32Array; seconds: number }>,
  samePerson = SAME_PERSON,
): Map<number, number> {
  let groups: Group[] = [...voices].map(([id, voice]) => ({
    ids: [id],
    print: Float32Array.from(voice.print),
    seconds: voice.seconds,
  }));
  for (;;) {
    let best: { into: Group; from: Group; alike: number } | null = null;
    for (const [at, a] of groups.entries()) {
      for (const b of groups.slice(at + 1)) {
        const alike = cosine(a.print, b.print);
        if (alike >= samePerson && (!best || alike > best.alike))
          best = { into: a, from: b, alike };
      }
    }
    if (!best) break;
    join(best.into, best.from);
    const gone = best.from;
    groups = groups.filter((group) => group !== gone);
  }
  const speech = groups.reduce((sum, group) => sum + group.seconds, 0);
  const kept = groups.filter((group) => group.seconds >= STRAY_SHARE * speech);
  for (const stray of groups.filter((group) => !kept.includes(group))) {
    const nearest = kept.reduce<Group | null>(
      (most, group) =>
        !most || cosine(stray.print, group.print) > cosine(stray.print, most.print) ? group : most,
      null,
    );
    if (nearest) join(nearest, stray);
  }
  const joined = new Map<number, number>();
  for (const group of kept.length > 0 ? kept : groups) {
    for (const id of group.ids) joined.set(id, group.ids[0] ?? id);
  }
  return joined;
}

/** Each voice's voiceprint, from its longest stretches, and how much it speaks. */
function voiceprintsOf(
  extractor: SpeakerEmbeddingExtractor,
  samples: Float32Array,
  segments: readonly Segment[],
): Map<number, { print: Float32Array; seconds: number }> {
  const bySpeaker = new Map<number, Segment[]>();
  for (const segment of segments) {
    bySpeaker.set(segment.speaker, [...(bySpeaker.get(segment.speaker) ?? []), segment]);
  }
  const voices = new Map<number, { print: Float32Array; seconds: number }>();
  for (const [speaker, own] of bySpeaker) {
    const longest = [...own].sort((a, b) => b.end - b.start - (a.end - a.start));
    const stream = extractor.createStream();
    let taken = 0;
    for (const segment of longest) {
      if (taken >= PRINT_SECONDS) break;
      const seconds = Math.min(segment.end - segment.start, PRINT_SECONDS - taken);
      const from = Math.round(segment.start * MEETING_SAMPLE_RATE);
      stream.acceptWaveform({
        samples: samples.subarray(from, from + Math.round(seconds * MEETING_SAMPLE_RATE)),
        sampleRate: MEETING_SAMPLE_RATE,
      });
      taken += seconds;
    }
    stream.inputFinished();
    voices.set(speaker, {
      print: Float32Array.from(extractor.compute(stream, false)),
      seconds: own.reduce((sum, segment) => sum + segment.end - segment.start, 0),
    });
  }
  return voices;
}

/**
 * Separates the voices in a request's parts. The models load for each request, as each is one
 * Meeting. `threshold` is for the evaluation's tuning.
 */
export async function createSherpaSeparator({
  threshold = CLUSTER_THRESHOLD,
  samePerson = SAME_PERSON,
}: {
  threshold?: number;
  samePerson?: number;
} = {}): Promise<(request: SeparationRequest) => VoiceTurn[]> {
  const sherpa = await loadSherpa();
  return (request) => {
    const { model } = request;
    const engine = new sherpa.OfflineSpeakerDiarization({
      segmentation: {
        pyannote: { model: model.segmentation },
        numThreads: model.threads,
        debug: 0,
      },
      embedding: { model: model.embedding, numThreads: model.threads, debug: 0 },
      clustering: { numClusters: -1, threshold },
      minDurationOn: 0.3,
      minDurationOff: 0.5,
    });
    const { samples, spans } = joinParts(request.parts);
    if (samples.length === 0) return [];
    const segments = engine.process(samples);
    const extractor = new sherpa.SpeakerEmbeddingExtractor({
      model: model.embedding,
      numThreads: model.threads,
      debug: 0,
    });
    const joined = joinedVoices(voiceprintsOf(extractor, samples, segments), samePerson);
    return toMeetingTime(
      segments.map((segment) => ({
        ...segment,
        speaker: joined.get(segment.speaker) ?? segment.speaker,
      })),
      spans,
    );
  };
}
