/**
 * Telling the voices in a recording apart (#138), with sherpa-onnx-node's speaker diarization:
 * pyannote's segmentation model finds where each voice speaks, 3D-Speaker's CAM++ gives each
 * stretch a voiceprint, and the stretches are clustered into voices. Its `process` holds its
 * thread until it is done, so the desktop app runs it in a process of its own
 * (src/main/speakerProcess.ts).
 */
import { readFileSync } from "node:fs";
import { MEETING_SAMPLE_RATE, type SpeakerModelFiles, type VoiceTurn } from "../adapters";
import { loadSherpa } from "./sherpa";

/** Voiceprints closer than this are one voice: tuned on AMI (eval/speakers/README.md). */
export const CLUSTER_THRESHOLD = 0.5;

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

/**
 * Separates the voices in a request's parts. The models load for each request, as each is one
 * Meeting. `threshold` is for the evaluation's tuning.
 */
export async function createSherpaSeparator({
  threshold = CLUSTER_THRESHOLD,
}: {
  threshold?: number;
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
    return toMeetingTime(engine.process(samples), spans);
  };
}
