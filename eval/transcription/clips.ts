/**
 * The transcription evaluations' audio (eval/transcription/README.md): FLEURS clips as
 * fetch-fleurs.py wrote them, joined into one stream with pauses between, and the
 * speech-to-text model's files.
 */
import { mkdir, mkdtemp, readFile, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MEETING_SAMPLE_RATE } from "../../src/core/adapters";
import { downloadModel } from "../../src/core/embedding/download";
import { QWEN3_ASR_0_6B, SILERO_VAD } from "../../src/core/transcription/model";
import { evalCache } from "../lib/cache";

/** Silence before each clip and after the last, as pauses between speakers. */
export const PAUSE_MS = 1500;
/** Audio goes to an engine 100 ms at a time, as a Meeting's capture sends it. */
export const FRAME = MEETING_SAMPLE_RATE / 10;

export interface Clip {
  file: string;
  reference: string;
  language: string;
}

/** A FLEURS clip's samples, mono at 16 kHz: 32-bit float, as FLEURS keeps them, or 16-bit PCM. */
export async function samplesOf(path: string): Promise<Float32Array> {
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

/** One language's clips, in the order fetched. */
export async function clipsOf(clipsDir: string, language: string): Promise<Clip[]> {
  const folder = join(clipsDir, language);
  const rows = (await readFile(join(folder, "clips.tsv"), "utf8")).trim().split("\n");
  return rows.map((row) => {
    const [file, raw] = row.split("\t");
    return { file: join(folder, file as string), reference: raw as string, language };
  });
}

/** The clips one after the other, `PAUSE_MS` of silence before each and after the last. */
export async function streamOf(clips: readonly Clip[]): Promise<Float32Array> {
  const pause = new Float32Array((MEETING_SAMPLE_RATE * PAUSE_MS) / 1000);
  const parts: Float32Array[] = [];
  for (const clip of clips) parts.push(pause, await samplesOf(clip.file));
  parts.push(pause);
  const stream = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    stream.set(part, offset);
    offset += part.length;
  }
  return stream;
}

/**
 * A data folder whose models/ holds the model's files, checked against their pinned SHA-256,
 * downloaded the first time into `models` (by default the evaluation's cache).
 */
export async function modelData(
  models = process.env.INCARNAMIND_EVAL_MODELS ?? join(evalCache(), "models"),
): Promise<string> {
  const dataDir = await mkdtemp(join(tmpdir(), "incarnamind-transcription-"));
  await mkdir(models, { recursive: true });
  await symlink(models, join(dataDir, "models"));
  const signal = new AbortController().signal;
  for (const [folder, source] of [
    [SILERO_VAD.folder, SILERO_VAD.source],
    [QWEN3_ASR_0_6B.folder, QWEN3_ASR_0_6B.source],
  ] as const) {
    await downloadModel({ directory: join(models, folder), source, signal, onProgress: () => {} });
  }
  return dataDir;
}
