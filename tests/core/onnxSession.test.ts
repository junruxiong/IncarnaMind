import { describe, expect, test, vi } from "vitest";
import { createOnnxCrossEncoder } from "../../src/core/reranking/onnx";

const create = vi.hoisted(() => vi.fn(async () => ({ release: async () => {} })));

vi.mock("onnxruntime-node", () => ({ InferenceSession: { create } }));
vi.mock("@huggingface/tokenizers", () => ({ Tokenizer: class {} }));
vi.mock("node:fs/promises", () => ({ readFile: async () => "{}" }));

describe("The built-in reranking model's session", () => {
  // Turning spinning off or using 4 threads was 16 to 29% slower on a calm computer: see #487.
  test("runs on the CPU with onnxruntime's default threads", async () => {
    await createOnnxCrossEncoder().load({
      model: "m.onnx",
      tokenizer: "t.json",
      tokenizerConfig: "c.json",
      maxTokens: 512,
    });

    expect(create).toHaveBeenCalledWith("m.onnx", {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });
  });
});
