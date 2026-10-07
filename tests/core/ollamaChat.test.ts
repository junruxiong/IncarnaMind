/**
 * Chat with local models through Ollama's own API (see
 * src/core/providers/ollamaChat.ts and ollamaModels.ts): the context window
 * chosen per model from its KV cache and this computer's memory, the settings
 * every request carries, the models offered, and Ollama's address. Against a
 * stub of Ollama's API: nothing here needs Ollama.
 */
import { describe, expect, test } from "vitest";
import {
  defaultOllamaUrl,
  OLLAMA_DEFAULT_URL,
  ollamaHostUrl,
} from "../../src/core/providers/kinds";
import { chatOptions, shapeChatRequest } from "../../src/core/providers/ollamaChat";
import {
  CONTEXT_SIZES,
  chooseNumCtx,
  citingMode,
  DEFAULT_OLLAMA_SETTINGS,
  kvBytesPerToken,
  kvCacheBytes,
  profileOf,
  thinkFor,
} from "../../src/core/providers/ollamaModels";
import { askAndEnd, BGE_M3, GIB, MISTRAL, QWEN35, setUpLocalModel } from "../helpers/localModels";
import { question, writeMind } from "../helpers/minds";
import { startOllamaServer } from "../helpers/ollama";

describe("The context window, num_ctx", () => {
  /** Qwen3-4B's model_info, as `/api/show` gives it. */
  const QWEN3_INFO = {
    "general.architecture": "qwen3",
    "qwen3.block_count": 36,
    "qwen3.attention.head_count": 32,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128,
    "qwen3.attention.value_length": 128,
    "qwen3.context_length": 262_144,
  };

  test("the KV cache is estimated from the layers, KV heads and head size /api/show gives", () => {
    // 36 layers × 8 KV heads × (128 + 128) values × 2 bytes.
    expect(kvBytesPerToken(QWEN3_INFO)).toBe(147_456);
    // At the model's own 262,144 tokens, about 38.7 GB: more than a 32 GB computer has.
    expect(kvCacheBytes(kvBytesPerToken(QWEN3_INFO), 262_144) / 1e9).toBeCloseTo(38.65, 1);
    // At 16,384, 2.4 GB.
    expect(kvCacheBytes(kvBytesPerToken(QWEN3_INFO), 16_384) / 1e9).toBeCloseTo(2.42, 1);
  });

  test("only layers with full attention keep a KV cache; the head size falls back to the embedding's share", () => {
    // Llama 3.2 3B: 28 layers × 8 KV heads × (128 + 128) values × 2 bytes.
    expect(
      kvBytesPerToken({
        "general.architecture": "llama",
        "llama.block_count": 28,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 128,
        "llama.attention.value_length": 128,
      }),
    ).toBe(114_688);
    // Mistral 7B gives no head size: 4096 / 32 heads = 128.
    expect(kvBytesPerToken(MISTRAL.modelInfo ?? {})).toBe(32 * 8 * 256 * 2);
    expect(
      kvBytesPerToken({
        "general.architecture": "qwen35",
        "qwen35.block_count": 32,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.full_attention_interval": 4,
      }),
    ).toBe(8 * 4 * 512 * 2);
  });

  test("the KV cache stays within a tenth of the memory and half of what is free, the model within half", () => {
    const qwen35 = { weightsBytes: 3_973_305_013, kvBytesPerToken: kvBytesPerToken({}) };
    const window = (totalGb: number, freeGb = totalGb) =>
      chooseNumCtx({
        ...qwen35,
        contextLength: 262_144,
        memoryBytes: totalGb * GIB,
        freeBytes: freeGb * GIB,
      });
    expect(window(64)).toBe(32_768);
    expect(window(32)).toBe(16_384);
    expect(window(16)).toBe(8_192);
    expect(window(8)).toBe(8_192);
    // Little free memory, e.g. other apps or another model loaded: the smallest window.
    expect(window(32, 2)).toBe(8_192);
    // A model trained for less keeps its own.
    expect(chooseNumCtx({ ...qwen35, contextLength: 4_096, memoryBytes: 32 * GIB })).toBe(4_096);
  });

  test("never inherits a huge context: a 256K model on a 256 GB computer gets 32,768", () => {
    const tag = { name: "qwen3:4b", size: 2_497_293_931, digest: "d" };
    const show = { capabilities: ["completion", "tools", "thinking"], model_info: QWEN3_INFO };
    expect(profileOf(tag, show, { totalBytes: 256 * GIB }).settings.numCtx).toBe(
      Math.max(...CONTEXT_SIZES),
    );
    // On a 32 GB Mac with 7 GB free: 16,384.
    expect(profileOf(tag, show, { totalBytes: 32 * GIB, freeBytes: 7 * GIB }).settings.numCtx).toBe(
      16_384,
    );
    expect(DEFAULT_OLLAMA_SETTINGS.numCtx).toBe(8_192);
  });
});

describe("What a local model can do, from its capabilities", () => {
  test("it cites with Tools when it can call them, else with structured output", () => {
    expect(citingMode(["completion", "tools", "thinking"])).toBe("tools");
    expect(citingMode(["completion"])).toBe("structured-output");
    expect(citingMode(["embedding"])).toBe("none");
    expect(citingMode(null)).toBeNull();
  });

  test("thinking stays off, unless the model can only think", () => {
    expect(thinkFor({ thinking: { values: [false, true], default: true } })).toBe(false);
    expect(thinkFor({ thinking: { values: [true], default: true } })).toBe(true);
    expect(thinkFor({})).toBe(false);
    const thinkingOnly = profileOf(
      { name: "qwen3:4b", size: 2_497_293_931 },
      {
        capabilities: ["completion", "tools", "thinking"],
        thinking: { values: [true], default: true },
      },
      { totalBytes: 32 * GIB },
    );
    expect(thinkingOnly.settings.think).toBe(true);
    // Its reasoning counts against the output cap: it gets half the window for it.
    expect(thinkingOnly.settings.outputTokens).toBe(thinkingOnly.settings.numCtx / 2);
  });
});

describe("Every chat request to Ollama", () => {
  test("carries the model's window, the Answer's temperature, an output cap, keep_alive, think: false and truncate: false", async () => {
    const ollama = await startOllamaServer({
      models: [QWEN35],
      reply: () => ({ content: "A Mind is a notebook." }),
    });
    const { core, mind, client } = await setUpLocalModel(ollama, QWEN35.name);
    const asked = question("What is a Mind?");
    writeMind(client, [asked]);

    const { ended } = await askAndEnd(core, client, mind.id, asked.attrs.id);

    expect(ended.event).toBe("finished");
    const body = ollama.chats[0]?.body ?? {};
    expect(body).toMatchObject({
      model: QWEN35.name,
      stream: true,
      think: false,
      truncate: false,
      keep_alive: "30m",
      // A 32 GB computer: 16,384 (see above), a quarter of it for the output.
      options: { num_ctx: 16_384, temperature: 0.2, num_predict: 4_096 },
    });
    // Fields `/api/chat` doesn't read aren't sent.
    for (const field of ["temperature", "top_p", "max_output_tokens", "tool_choice"]) {
      expect(body).not.toHaveProperty(field);
    }
  });

  test("a request that may call no Tool goes without them; the output limit never passes the model's cap", () => {
    const settings = { ...DEFAULT_OLLAMA_SETTINGS };
    const shaped = shapeChatRequest(
      { model: "m", tools: [{ type: "function" }], tool_choice: "none", temperature: 0.2 },
      settings,
    );
    expect(shaped).toMatchObject({ keep_alive: "30m", truncate: false });
    expect(shaped).not.toHaveProperty("tools");
    expect(shaped).not.toHaveProperty("tool_choice");
    expect(chatOptions(settings, { maxOutputTokens: 64, temperature: 0.2 })).toEqual({
      num_ctx: 8_192,
      num_predict: 64,
      temperature: 0.2,
    });
    expect(chatOptions(settings, { maxOutputTokens: 100_000 }).num_predict).toBe(
      settings.outputTokens,
    );
  });
});

describe("Ollama's models and address", () => {
  test("the picker leaves out models that can't chat, such as embedding models", async () => {
    const ollama = await startOllamaServer({
      models: [QWEN35, MISTRAL, BGE_M3],
      reply: () => ({ content: "" }),
    });
    const { core } = await setUpLocalModel(ollama, QWEN35.name);

    const [group] = await core.listChatModels();

    expect(group?.models).toEqual([QWEN35.name, "mistral:latest"]);
  });

  test("an embedding model can't be asked: the Answer says the model can't answer, and nothing is sent", async () => {
    const ollama = await startOllamaServer({ models: [BGE_M3], reply: () => ({ content: "" }) });
    const { core, mind, client } = await setUpLocalModel(ollama, "bge-m3");
    const asked = question("What is a Mind?");
    writeMind(client, [asked]);

    const { ended } = await askAndEnd(core, client, mind.id, asked.attrs.id);

    expect(ended.payload).toMatchObject({ error: { kind: "model" } });
    expect(ollama.chats).toHaveLength(0);
  });

  test("OLLAMA_HOST is read as Ollama reads it", () => {
    expect(ollamaHostUrl(undefined)).toBeNull();
    expect(ollamaHostUrl("")).toBeNull();
    expect(ollamaHostUrl("0.0.0.0")).toBe("http://127.0.0.1:11434");
    expect(ollamaHostUrl("0.0.0.0:11500")).toBe("http://127.0.0.1:11500");
    expect(ollamaHostUrl("gpu-box.local")).toBe("http://gpu-box.local:11434");
    // With a scheme and no port, the scheme's own port (which a URL leaves out).
    expect(ollamaHostUrl("http://gpu-box.local")).toBe("http://gpu-box.local");
    expect(ollamaHostUrl("https://ollama.example.com/base/")).toBe(
      "https://ollama.example.com/base",
    );
    expect(ollamaHostUrl("[::1]:11435")).toBe("http://[::1]:11435");
    expect(ollamaHostUrl("ftp://nope")).toBeNull();
    expect(defaultOllamaUrl({ OLLAMA_HOST: "127.0.0.1:11500" })).toBe("http://127.0.0.1:11500");
    expect(defaultOllamaUrl({})).toBe(OLLAMA_DEFAULT_URL);
  });
});
