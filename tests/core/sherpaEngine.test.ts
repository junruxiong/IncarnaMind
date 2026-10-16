import { describe, expect, test, vi } from "vitest";
import { MEETING_SAMPLE_RATE } from "../../src/core/adapters";

/*
 * The engine's recognizer when sherpa-onnx's own reader refuses a result (#981):
 * a stand-in for the package whose `decodeAsync` reads JSON the way the real
 * one does, and an addon that still holds the result.
 */
const RESULTS: string[] = [];
const latest = () => RESULTS.at(-1) ?? "";

vi.mock("sherpa-onnx-node", () => ({
  OfflineRecognizer: {
    createAsync: async () => ({
      createStream: () => ({ acceptWaveform: () => {}, handle: { id: RESULTS.length } }),
      decodeAsync: async () => JSON.parse(latest()),
    }),
  },
  Vad: class {},
}));
vi.mock("sherpa-onnx-node/addon.js", () => ({
  default: { getOfflineStreamResultAsJson: latest },
}));

const { createSherpaEngine } = await import("../../src/core/transcription/sherpa");

const model = {
  engine: "qwen3-asr",
  files: { convFrontend: "a", encoder: "b", decoder: "c", tokenizer: "d" },
  vad: "v",
  threads: 1,
};
const samples = new Float32Array(MEETING_SAMPLE_RATE);

describe("The sherpa-onnx recognizer", () => {
  test("keeps the words of a stretch whose result has a control character, which the package's reader refuses", async () => {
    const engine = await createSherpaEngine(model);
    RESULTS.push('{"lang": "", "text": "The launch\nmoves to the eighteenth.", "tokens": []}');
    expect(() => JSON.parse(latest())).toThrow(SyntaxError);

    expect(await engine.recognizer.decode(samples, "meeting:1:microphone")).toBe(
      "The launch\nmoves to the eighteenth.",
    );
  });

  test("leaves out the model's header that sherpa-onnx left in the text, with its control characters too", async () => {
    const engine = await createSherpaEngine(model);
    RESULTS.push(
      '{"lang": "", "text": "提uggestion\nlanguage English<asr_text>Local authorities are warning.", "tokens": []}',
    );

    expect(await engine.recognizer.decode(samples, "meeting:1:microphone")).toBe(
      "Local authorities are warning.",
    );
  });

  test("gives the text of an ordinary result as the package reads it", async () => {
    const engine = await createSherpaEngine(model);
    RESULTS.push('{"lang": "", "text": "Plain words.", "tokens": []}');

    expect(await engine.recognizer.decode(samples, "meeting:1:microphone")).toBe("Plain words.");
  });

  test("fails for a result it can't read at all, as before", async () => {
    const engine = await createSherpaEngine(model);
    RESULTS.push("not json");

    await expect(engine.recognizer.decode(samples, "meeting:1:microphone")).rejects.toThrow(
      SyntaxError,
    );
  });
});
