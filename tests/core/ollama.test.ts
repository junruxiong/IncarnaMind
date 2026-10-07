import { describe, expect, test } from "vitest";
import {
  type OllamaPullProgress,
  RECOMMENDED_OLLAMA_DOWNLOAD_GB,
  RECOMMENDED_OLLAMA_MODEL,
} from "../../src/core";
import { translate } from "../../src/shared/i18n";
import { createTempDataFolder, startCore } from "../helpers/core";
import { startOllamaStub, unusedLocalUrl } from "../helpers/ollama";

const PULL_LINES = [
  { status: "pulling manifest" },
  { status: "pulling 2bada8a74506", digest: "sha256:2bada8a74506", total: 2_000, completed: 0 },
  { status: "pulling 2bada8a74506", digest: "sha256:2bada8a74506", total: 2_000, completed: 1_000 },
  { status: "pulling 2bada8a74506", digest: "sha256:2bada8a74506", total: 2_000, completed: 2_000 },
  { status: "verifying sha256 digest" },
  { status: "writing manifest" },
  { status: "success" },
];

describe("Ollama", () => {
  test("is detected on a running server, with the models it already has", async () => {
    const ollama = await startOllamaStub({ models: ["llama3.2:latest", "qwen3:8b"] });
    const core = startCore(await createTempDataFolder());

    expect(await core.detectOllama({ baseUrl: ollama.baseUrl })).toEqual({
      running: true,
      baseUrl: ollama.baseUrl,
      models: ["llama3.2:latest", "qwen3:8b"],
      recommendedModel: RECOMMENDED_OLLAMA_MODEL,
    });
  });

  test("is reported as not running when nothing answers", async () => {
    const baseUrl = await unusedLocalUrl();
    const core = startCore(await createTempDataFolder());

    expect(await core.detectOllama({ baseUrl })).toEqual({ running: false, baseUrl });
  });

  test("one click pulls the recommended model with progress and selects it", async () => {
    const ollama = await startOllamaStub({ pullLines: PULL_LINES });
    const core = startCore(await createTempDataFolder());
    const progress: OllamaPullProgress[] = [];
    core.on("ollama.pullProgress", (update) => progress.push(update));

    const provider = await core.selectOllama({ baseUrl: ollama.baseUrl });

    expect(ollama.pulls).toEqual([{ model: RECOMMENDED_OLLAMA_MODEL, stream: true }]);
    expect(progress.map((update) => update.status)).toEqual(PULL_LINES.map((line) => line.status));
    expect(progress[2]).toEqual({
      model: RECOMMENDED_OLLAMA_MODEL,
      status: "pulling 2bada8a74506",
      completed: 1_000,
      total: 2_000,
    });
    expect(provider).toEqual({
      id: expect.any(String),
      kind: "ollama",
      baseUrl: ollama.baseUrl,
      hasApiKey: false,
      service: null,
    });
    expect(await core.getChatReadiness()).toEqual({
      ready: true,
      provider,
      modelId: RECOMMENDED_OLLAMA_MODEL,
      consent: "not-required",
    });
  });

  test("a model Ollama already has is selected without pulling it", async () => {
    const ollama = await startOllamaStub({ models: ["llama3.2:latest"] });
    const core = startCore(await createTempDataFolder());

    await core.selectOllama({ baseUrl: ollama.baseUrl, model: "llama3.2" });

    expect(ollama.pulls).toEqual([]);
    expect(await core.getChatReadiness()).toMatchObject({ ready: true, modelId: "llama3.2" });
  });

  test("a failed pull reports Ollama's error and selects nothing", async () => {
    const ollama = await startOllamaStub({
      pullLines: [
        { status: "pulling manifest" },
        { error: "pull model manifest: file does not exist" },
      ],
    });
    const core = startCore(await createTempDataFolder());

    await expect(
      core.selectOllama({ baseUrl: ollama.baseUrl, model: "no-such-model" }),
    ).rejects.toThrow(/file does not exist/);
    expect(await core.listChatProviders()).toEqual([]);
    expect(await core.getChatReadiness()).toEqual({ ready: false, reason: "no-provider" });
  });

  test("one click offers Qwen3.5-4B, naming its download size in English and in Chinese", () => {
    expect(RECOMMENDED_OLLAMA_MODEL).toBe("qwen3.5:4b");
    const size = `${RECOMMENDED_OLLAMA_DOWNLOAD_GB} GB`;
    for (const language of ["en", "zh-CN"] as const) {
      const model = { model: RECOMMENDED_OLLAMA_MODEL };
      expect(translate(language, "providers.ollama.use", model)).toContain(size);
      expect(translate(language, "providers.ollama.pulling", model)).toContain(size);
    }
  });

  test("one click fails clearly when Ollama isn't running", async () => {
    const baseUrl = await unusedLocalUrl();
    const core = startCore(await createTempDataFolder());

    await expect(core.selectOllama({ baseUrl })).rejects.toThrow(/isn't running/);
    expect(await core.listChatProviders()).toEqual([]);
  });
});
