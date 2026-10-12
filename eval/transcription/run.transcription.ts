/**
 * `npm run eval:transcription` (eval/transcription/README.md): the real
 * speech-to-text engine (Qwen3-ASR-0.6B int8 and Silero VAD, through
 * sherpa-onnx-node) on a stream made of FLEURS clips, one per language, fed
 * in 100 ms frames at the pace a Meeting is captured, through the same code
 * as the transcription process (`createSpeechHost`). Reports WER (English),
 * CER (Chinese), how long decoding took for the audio it decoded, and how far
 * live text was behind when the audio ended.
 *
 * INCARNAMIND_EVAL_CLIPS: the folder fetch-fleurs.py wrote (required).
 * INCARNAMIND_EVAL_MODELS: a folder holding the model's files as the data
 * folder's `models/` does; by default ~/.cache/incarnamind-eval/models, where
 * they are downloaded and checked the first time.
 */
import { mkdir, mkdtemp, readFile, symlink, writeFile } from "node:fs/promises";
import { homedir, loadavg, tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { MEETING_SAMPLE_RATE, type SpeechSegment } from "../../src/core/adapters";
import { downloadModel } from "../../src/core/embedding/download";
import {
  QWEN3_ASR_0_6B,
  SILERO_VAD,
  transcriptionModelFiles,
} from "../../src/core/transcription/model";
import { createSherpaEngine } from "../../src/core/transcription/sherpa";
import { createSpeechHost } from "../../src/core/transcription/speech";
import { characters, type ErrorRate, errorRate, words } from "./scoring";

const CLIPS = process.env.INCARNAMIND_EVAL_CLIPS;
const MODELS =
  process.env.INCARNAMIND_EVAL_MODELS ?? join(homedir(), ".cache", "incarnamind-eval", "models");
/** Silence before each clip and after the last, as pauses between speakers. */
const PAUSE_MS = 1500;
const FRAME = MEETING_SAMPLE_RATE / 10;

interface Clip {
  file: string;
  reference: string;
}

/** A FLEURS clip's samples, mono at 16 kHz: 32-bit float, as FLEURS keeps them, or 16-bit PCM. */
async function samplesOf(path: string): Promise<Float32Array> {
  const bytes = await readFile(path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 12;
  let format: { code: number; channels: number; rate: number; bits: number } | null = null;
  while (at + 8 <= bytes.length) {
    const id = bytes.toString("ascii", at, at + 4);
    const size = view.getUint32(at + 4, true);
    if (id === "fmt ") {
      format = {
        code: view.getUint16(at + 8, true),
        channels: view.getUint16(at + 10, true),
        rate: view.getUint32(at + 12, true),
        bits: view.getUint16(at + 22, true),
      };
    }
    if (id === "data") {
      const float = format?.code === 3 && format.bits === 32;
      const pcm = format?.code === 1 && format.bits === 16;
      if (format?.channels !== 1 || format.rate !== MEETING_SAMPLE_RATE || !(float || pcm)) {
        throw new Error(`${path} isn't mono 16 kHz audio we read: ${JSON.stringify(format)}`);
      }
      const width = float ? 4 : 2;
      const out = new Float32Array(Math.floor(size / width));
      for (let i = 0; i < out.length; i++) {
        const offset = at + 8 + i * width;
        out[i] = float ? view.getFloat32(offset, true) : view.getInt16(offset, true) / 32768;
      }
      return out;
    }
    at += 8 + size + (size % 2);
  }
  throw new Error(`${path} has no audio.`);
}

async function clipsOf(language: string): Promise<{ clips: Clip[]; folder: string }> {
  const folder = join(CLIPS as string, language);
  const rows = (await readFile(join(folder, "clips.tsv"), "utf8")).trim().split("\n");
  return {
    folder,
    clips: rows.map((row) => {
      const [file, raw] = row.split("\t");
      return { file: file as string, reference: raw as string };
    }),
  };
}

/** The model's files, checked against their pinned SHA-256, downloaded the first time. */
async function modelData(): Promise<string> {
  const dataDir = await mkdtemp(join(tmpdir(), "incarnamind-transcription-"));
  await mkdir(MODELS, { recursive: true });
  await symlink(MODELS, join(dataDir, "models"));
  const signal = new AbortController().signal;
  for (const [folder, source] of [
    [SILERO_VAD.folder, SILERO_VAD.source],
    [QWEN3_ASR_0_6B.folder, QWEN3_ASR_0_6B.source],
  ] as const) {
    await downloadModel({ directory: join(MODELS, folder), source, signal, onProgress: () => {} });
  }
  return dataDir;
}

interface LanguageResult {
  language: string;
  clips: number;
  audioSeconds: number;
  speechSeconds: number;
  score: ErrorRate & { unit: "word" | "character" };
  /** Decoding time for the final text over the audio it decoded. */
  decodeRatio: number;
  /** All decoding, partial text included, over the stream's length. */
  busyRatio: number;
  partials: number;
  /** How long after the audio ended its last final text came, in seconds. */
  behindAtEnd: number;
  hypothesis: string;
}

describe.runIf(CLIPS)("Transcription on this computer", () => {
  test("WER, CER and how long decoding takes, on FLEURS clips fed as a Meeting is", async () => {
    const dataDir = await modelData();
    const files = transcriptionModelFiles(dataDir, QWEN3_ASR_0_6B);
    const loadStarted = Date.now();
    const engine = await createSherpaEngine(files);
    const loadSeconds = (Date.now() - loadStarted) / 1000;
    const results: LanguageResult[] = [];

    for (const language of ["en", "zh"]) {
      const { clips, folder } = await clipsOf(language);
      const pause = new Float32Array((MEETING_SAMPLE_RATE * PAUSE_MS) / 1000);
      const parts: Float32Array[] = [];
      for (const clip of clips) parts.push(pause, await samplesOf(join(folder, clip.file)));
      parts.push(pause);
      const stream = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
      let offset = 0;
      for (const part of parts) {
        stream.set(part, offset);
        offset += part.length;
      }

      const segments: SpeechSegment[] = [];
      let lastFinalAt = 0;
      const host = createSpeechHost(engine, {
        segment(segment) {
          segments.push(segment);
          if (segment.final) lastFinalAt = Date.now();
        },
        progress: () => {},
        failed: (_stream, message) => console.error(`A stretch couldn't be decoded: ${message}`),
      });
      const started = Date.now();
      for (let at = 0; at < stream.length; at += FRAME) {
        host.push(language, stream.subarray(at, at + FRAME));
        const due = started + ((at + FRAME) * 1000) / MEETING_SAMPLE_RATE;
        await new Promise((resolve) => setTimeout(resolve, Math.max(0, due - Date.now())));
      }
      const audioEnded = Date.now();
      await host.finish(language);
      const finals = segments.filter((segment) => segment.final);
      const hypothesis = finals.map((segment) => segment.text).join(language === "zh" ? "" : " ");
      const reference = clips.map((clip) => clip.reference).join(language === "zh" ? "" : " ");
      const split = language === "zh" ? characters : words;
      const speechMs = finals.reduce((sum, segment) => sum + segment.endMs - segment.startMs, 0);
      const audioMs = (stream.length * 1000) / MEETING_SAMPLE_RATE;
      results.push({
        language,
        clips: clips.length,
        audioSeconds: Math.round(audioMs / 100) / 10,
        speechSeconds: Math.round(speechMs / 100) / 10,
        score: {
          ...errorRate(split(reference), split(hypothesis)),
          unit: language === "zh" ? "character" : "word",
        },
        decodeRatio:
          finals.reduce((sum, segment) => sum + segment.decodeMs, 0) / Math.max(1, speechMs),
        busyRatio: segments.reduce((sum, segment) => sum + segment.decodeMs, 0) / audioMs,
        partials: segments.length - finals.length,
        behindAtEnd: Math.max(0, lastFinalAt - audioEnded) / 1000,
        hypothesis,
      });
    }
    engine.close();

    const report = {
      date: new Date().toISOString(),
      model: QWEN3_ASR_0_6B.name,
      threads: files.threads,
      loadSeconds,
      /** The machine's load while it ran: the ratios are only as good as the machine was free. */
      loadAverage: loadavg().map((value) => Math.round(value * 10) / 10),
      results,
    };
    const out = join(process.cwd(), "eval", "results");
    await mkdir(out, { recursive: true });
    await writeFile(
      join(out, `transcription-${report.date.slice(0, 10)}.json`),
      JSON.stringify(report, null, 2),
    );
    console.log(
      [
        `${report.model}, ${report.threads} threads, loaded in ${loadSeconds.toFixed(1)} s, load average ${report.loadAverage.join(" ")}`,
        ...results.map(
          (result) =>
            `${result.language}: ${result.clips} clips, ${result.audioSeconds} s of audio; ${result.score.unit === "word" ? "WER" : "CER"} ${(result.score.rate * 100).toFixed(1)}% (${result.score.errors}/${result.score.total}); decoding ${result.decodeRatio.toFixed(2)}× the speech it decoded, ${result.busyRatio.toFixed(2)}× the stream with partial text; ${result.partials} partials; last line ${result.behindAtEnd.toFixed(1)} s after the audio ended`,
        ),
      ].join("\n"),
    );
    expect(results).toHaveLength(2);
  });
});
