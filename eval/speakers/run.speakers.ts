/**
 * `npm run eval:speakers` (eval/speakers/README.md): the speaker pass's own separator
 * (`createSherpaSeparator` in src/core/transcription/diarization.ts) on AMI meetings written as
 * a Meeting's 10-second parts, scored for DER against AMI's speakers.
 *
 * INCARNAMIND_EVAL_SPEAKER_CLIPS: the folder fetch-ami.py wrote (online/ and room/).
 * INCARNAMIND_EVAL_SPEAKER_THRESHOLDS: clustering thresholds to try, e.g. "0.5,0.7" (default: the app's).
 * INCARNAMIND_EVAL_SPEAKER_HOUR=1: also joins the online meetings into an hour, for the real-time factor.
 */
import { readdirSync, readFileSync } from "node:fs";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { loadavg, tmpdir } from "node:os";
import { basename, join } from "node:path";
import { describe, expect, test } from "vitest";
import { silentLogger } from "../../src/core/activityLog";
import { downloadModel } from "../../src/core/embedding/download";
import { writePart } from "../../src/core/meetings/audioFiles";
import { labelled } from "../../src/core/meetings/speakerPass";
import { CLUSTER_THRESHOLD, createSherpaSeparator } from "../../src/core/transcription/diarization";
import { SILERO_VAD } from "../../src/core/transcription/model";
import { createSileroDetectors } from "../../src/core/transcription/sherpa";
import { createSpeakerModel } from "../../src/core/transcription/speakerModel";
import { evalCache } from "../lib/cache";
import { diarizationError, readRttm, type Turn } from "./der";

const CLIPS = process.env.INCARNAMIND_EVAL_SPEAKER_CLIPS ?? "";
const THRESHOLDS = (process.env.INCARNAMIND_EVAL_SPEAKER_THRESHOLDS ?? String(CLUSTER_THRESHOLD))
  .split(",")
  .map(Number);
const PART_SAMPLES = 160_000;

interface Clip {
  name: string;
  setting: "online" | "room";
  samples: Int16Array;
  reference: Turn[];
}

/** A WAV file's 16-bit samples, wherever its data chunk is. */
function samplesOf(path: string): Int16Array {
  const bytes = readFileSync(path);
  let at = 12;
  while (at + 8 <= bytes.length) {
    const id = bytes.toString("ascii", at, at + 4);
    const size = bytes.readUInt32LE(at + 4);
    if (id === "data") {
      const body = bytes.subarray(at + 8, at + 8 + size);
      const copy = new Uint8Array(body.byteLength - (body.byteLength % 2));
      copy.set(body.subarray(0, copy.byteLength));
      return new Int16Array(copy.buffer);
    }
    at += 8 + size + (size % 2);
  }
  throw new Error(`${path} has no data chunk.`);
}

function clipsIn(setting: Clip["setting"]): Clip[] {
  const folder = join(CLIPS, setting);
  return readdirSync(folder)
    .filter((name) => name.endsWith(".wav"))
    .sort()
    .map((name) => ({
      name: basename(name, ".wav"),
      setting,
      samples: samplesOf(join(folder, name)),
      reference: readRttm(readFileSync(join(folder, name.replace(/\.wav$/, ".rttm")), "utf8")),
    }));
}

/** The clip as a Meeting keeps its audio: 10-second parts. */
async function partsOf(samples: Int16Array) {
  const folder = await mkdtemp(join(tmpdir(), "incarnamind-speakers-"));
  const parts = [];
  for (let at = 0; at < samples.length; at += PART_SAMPLES) {
    const part = writePart(
      folder,
      "microphone",
      (at * 1000) / 16_000,
      samples.subarray(at, at + PART_SAMPLES),
    );
    parts.push({ path: part.path, offsetMs: part.offsetMs });
  }
  return { folder, parts };
}

async function modelFiles() {
  const dataDir = await mkdtemp(join(tmpdir(), "incarnamind-speaker-models-"));
  const models = join(evalCache(), "models");
  await mkdir(models, { recursive: true });
  await symlink(models, join(dataDir, "models"));
  const vad = join(models, SILERO_VAD.folder);
  await downloadModel({
    directory: vad,
    source: SILERO_VAD.source,
    signal: new AbortController().signal,
    onProgress: () => {},
  });
  return {
    speakers: await createSpeakerModel({ dataDir, log: silentLogger }).files(),
    vad: join(vad, SILERO_VAD.file),
  };
}

/**
 * The Transcript's lines as the app would cut them, with Silero VAD as the transcription
 * process runs it: the final labels are scored on them, each with the voice it overlaps most.
 */
async function linesOf(samples: Int16Array, vadFile: string): Promise<Turn[]> {
  const detector = (await createSileroDetectors(vadFile))();
  const lines: Turn[] = [];
  const take = () => {
    for (let found = detector.next(); found; found = detector.next()) {
      const startMs = (found.start * 1000) / 16_000;
      lines.push({
        startMs,
        endMs: startMs + (found.samples.length * 1000) / 16_000,
        speaker: "Speaker ?",
      });
    }
  };
  for (let at = 0; at < samples.length; at += 1600) {
    const chunk = samples.subarray(at, at + 1600);
    const floats = new Float32Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) floats[i] = (chunk[i] as number) / 32768;
    detector.accept(floats);
    take();
  }
  detector.flush();
  take();
  return lines;
}

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

describe.runIf(CLIPS)("Telling speakers apart after the call", () => {
  test("DER and the real-time factor on AMI meetings, online and in a room", async () => {
    const { speakers: model, vad } = await modelFiles();
    const clips = [...clipsIn("online"), ...clipsIn("room")];
    if (process.env.INCARNAMIND_EVAL_SPEAKER_HOUR === "1") {
      const online = clips.filter((clip) => clip.setting === "online");
      const joined: Clip = {
        name: "an hour of online meetings",
        setting: "online",
        samples: new Int16Array(0),
        reference: [],
      };
      const pieces: Int16Array[] = [];
      let at = 0;
      for (const clip of online) {
        if (at >= 3_600_000) break;
        pieces.push(clip.samples);
        joined.reference.push(
          ...clip.reference.map((turn) => ({
            startMs: turn.startMs + at,
            endMs: turn.endMs + at,
            speaker: `${clip.name}:${turn.speaker}`,
          })),
        );
        at += (clip.samples.length * 1000) / 16_000;
      }
      joined.samples = new Int16Array(pieces.reduce((sum, piece) => sum + piece.length, 0));
      let offset = 0;
      for (const piece of pieces) {
        joined.samples.set(piece, offset);
        offset += piece.length;
      }
      clips.push(joined);
    }
    const rows: string[] = [];
    for (const threshold of THRESHOLDS) {
      const separate = await createSherpaSeparator({ threshold });
      for (const clip of clips) {
        const { folder, parts } = await partsOf(clip.samples);
        const load = loadavg()[0] ?? 0;
        const started = performance.now();
        const turns = separate({ parts, model });
        const seconds = (performance.now() - started) / 1000;
        await rm(folder, { recursive: true, force: true });
        const hypothesis = turns.map((turn) => ({ ...turn, speaker: String(turn.voice) }));
        const exact = diarizationError(clip.reference, hypothesis);
        const collared = diarizationError(clip.reference, hypothesis, { collarMs: 250 });
        const lines = await linesOf(clip.samples, vad);
        const named = labelled(
          lines.map((line) => ({ ...line, text: "" })),
          turns,
          false,
        );
        const final = diarizationError(
          clip.reference,
          (named ?? lines).map((line) => ({ ...line, speaker: line.speaker ?? "Speaker ?" })),
        );
        const audio = clip.samples.length / 16_000;
        const row = [
          threshold,
          clip.setting,
          clip.name,
          `${(audio / 60).toFixed(1)} min`,
          `${exact.referenceSpeakers} / ${exact.hypothesisSpeakers}`,
          pct(exact.der),
          pct(collared.der),
          `${pct(final.der)} (${final.hypothesisSpeakers})`,
          `${pct(exact.missedMs / exact.speechMs)} / ${pct(exact.falseAlarmMs / exact.speechMs)} / ${pct(exact.confusionMs / exact.speechMs)}`,
          `${(seconds / audio).toFixed(3)}×`,
          load.toFixed(1),
        ].join(" | ");
        console.log(row);
        rows.push(`| ${row} |`);
      }
    }
    const out = join(process.cwd(), "eval", "results");
    await mkdir(out, { recursive: true });
    await writeFile(
      join(out, `speakers-${new Date().toISOString().replace(/[:.]/g, "-")}.md`),
      [
        `Model threads: ${model.threads}. Load average at the end: ${loadavg()
          .map((value) => value.toFixed(1))
          .join(", ")}.`,
        "",
        "| Threshold | Setting | Meeting | Audio | Speakers (AMI / found) | DER | DER, 0.25 s collar | DER on the lines' labels (voices) | Missed / false alarm / confusion | Time / audio | Load |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        ...rows,
      ].join("\n"),
    );
    expect(rows.length).toBe(clips.length * THRESHOLDS.length);
  });
});
