/**
 * sherpa-onnx-node, which transcribes Meetings (#136), bundles an ONNX
 * Runtime of its own; the reranker and the embedding model run on
 * onnxruntime-node's. The app runs each in a utility process of its own, so
 * they never share one; here both run in one process, in either order, the
 * harder case. Each runs Silero VAD (a 213 KB fixture), so no model is downloaded.
 */
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";

const MODEL = fileURLToPath(
  new URL("../fixtures/transcription/silero_vad.int8.onnx", import.meta.url),
);
const ROOT = fileURLToPath(new URL("../..", import.meta.url));

/** Runs the model with onnxruntime-node on 512 samples of silence: its outputs' names. */
const ORT_RUN = `
  const ort = await import("onnxruntime-node");
  const session = await ort.InferenceSession.create(process.env.MODEL);
  const state = () => new ort.Tensor("float32", new Float32Array(128), [2, 1, 64]);
  const out = await session.run({
    x: new ort.Tensor("float32", new Float32Array(512), [1, 512]),
    h: state(),
    c: state(),
  });
  const prob = out.prob.data[0];
  if (!(prob >= 0 && prob < 0.5)) throw new Error("Silence read as speech: " + prob);
  results.push("ort:" + Object.keys(out).sort().join(","));
`;

/** Runs Silero VAD through sherpa-onnx-node on three seconds of quiet noise. */
const SHERPA_RUN = `
  const sherpa = (await import("sherpa-onnx-node")).default;
  const vad = new sherpa.Vad({
    sileroVad: { model: process.env.MODEL, threshold: 0.5, minSpeechDuration: 0.25, minSilenceDuration: 0.5, maxSpeechDuration: 20, windowSize: 512 },
    sampleRate: 16000, numThreads: 1, debug: 0,
  }, 30);
  const samples = new Float32Array(16000 * 3).map(() => (Math.random() - 0.5) * 0.001);
  for (let at = 0; at + 512 <= samples.length; at += 512) vad.acceptWaveform(samples.subarray(at, at + 512));
  vad.flush();
  results.push("sherpa:" + (vad.isEmpty() ? "no speech" : "speech"));
`;

function inOneProcess(...steps: string[]): string[] {
  const blocks = steps.map((step) => `{${step}}`).join("\n");
  const script = `const results = []; ${blocks} console.log(JSON.stringify(results));`;
  const output = execFileSync(process.execPath, ["--input-type=module", "-e", script], {
    cwd: ROOT,
    env: { ...process.env, MODEL },
    encoding: "utf8",
  });
  return JSON.parse(output.trim().split("\n").at(-1) ?? "[]") as string[];
}

test("onnxruntime-node and sherpa-onnx-node's ONNX Runtime work in one process, onnxruntime-node loaded first", () => {
  expect(inOneProcess(ORT_RUN, SHERPA_RUN, ORT_RUN)).toEqual([
    "ort:new_c,new_h,prob",
    "sherpa:no speech",
    "ort:new_c,new_h,prob",
  ]);
});

test("onnxruntime-node and sherpa-onnx-node's ONNX Runtime work in one process, sherpa-onnx-node loaded first", () => {
  expect(inOneProcess(SHERPA_RUN, ORT_RUN, SHERPA_RUN)).toEqual([
    "sherpa:no speech",
    "ort:new_c,new_h,prob",
    "sherpa:no speech",
  ]);
});
