/**
 * Answers from real local models through the Ollama on this computer, with
 * the app's engine, prompts and Ollama wrapper. Skipped unless
 * INCARNAMIND_REAL_OLLAMA=1: it needs Ollama running with the models already
 * pulled (it pulls nothing). By default it runs the small qwen3:0.6b, which
 * checks the plumbing, not the quality of Answers. Set OLLAMA_URL for another
 * server, INCARNAMIND_OLLAMA_MODELS for other models (comma-separated, run
 * one at a time), and INCARNAMIND_OLLAMA_CITING ("tools", "structured-output"
 * or "none") to try another citing mode than the one the capabilities choose.
 *
 * Whatever this computer's memory, every request's window is at most
 * `MAX_NUM_CTX`, so a test never loads a model with a large KV cache. It
 * checks that every request carries the window, `keep_alive`, `think: false`
 * and `truncate: false` and fits, by Ollama's own count, and logs per Answer:
 * the citing mode, the time, the phases, the Citations, and the markers the
 * engine placed. It unloads each model (`keep_alive: 0`) when it is done with it.
 */
import { describe, expect, test } from "vitest";
import {
  type CitationSupport,
  type Core,
  type CoreEvents,
  createAiSdkChatModel,
  createOllamaModels,
  type OllamaModels,
} from "../../src/core";
import { createOllamaChatModel } from "../../src/core/providers/ollamaChat";
import { DEFAULT_OLLAMA_SETTINGS, outputTokensFor } from "../../src/core/providers/ollamaModels";
import { answerEnded } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind } from "../helpers/mindClient";
import { answerText, question, writeMind } from "../helpers/minds";
import { waitForTagging } from "../helpers/tags";

const BASE_URL = (process.env.OLLAMA_URL ?? "http://127.0.0.1:11434").replace(/\/+$/, "");
const CITING = process.env.INCARNAMIND_OLLAMA_CITING as CitationSupport | undefined;
/** The largest window a test request may have. */
const MAX_NUM_CTX = 8_192;
const MODELS = (process.env.INCARNAMIND_OLLAMA_MODELS ?? "qwen3:0.6b")
  .split(",")
  .map((each) => each.trim())
  .filter(Boolean);

const DOCUMENTS = {
  "Attention.md": [
    "# Attention Is All You Need",
    "",
    "The Transformer is a model architecture that relies entirely on an attention mechanism to draw global dependencies between input and output, dispensing with recurrence and convolutions.",
    "",
    "We trained our models on one machine with 8 NVIDIA P100 GPUs. Each training step for the base models took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. The big models were trained for 300,000 steps (3.5 days).",
    "",
    "On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4.",
    "",
    "We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ε = 10^-9, and varied the learning rate over the course of training, increasing it linearly for the first 4000 warmup steps.",
  ].join("\n"),
  "梯度下降.md": [
    "# 梯度下降",
    "",
    "梯度下降法是一种一阶迭代优化算法，用于寻找可微函数的局部最小值。每一步都沿着当前点梯度的反方向移动。",
    "",
    "学习率决定每一步移动的距离。学习率过大时，算法可能在最小值附近来回震荡，甚至发散；学习率过小时，收敛会非常缓慢。",
    "",
    "随机梯度下降每次只用一个样本或一小批样本来估计梯度，因此每一步的计算量很小，适合大规模数据集。",
  ].join("\n"),
};

const QUESTIONS = [
  "How long were the big Transformer models trained, and on what hardware?",
  "What BLEU score did the big Transformer reach on English-to-German?",
  "学习率过大会怎样？",
  "什么是随机梯度下降？",
];

/** One chat request to Ollama, as sent and as counted. */
interface Logged {
  model: string;
  numCtx: number | null;
  numPredict: number | null;
  truncate: unknown;
  think: unknown;
  keepAlive: unknown;
  status: number;
  promptTokens: number | null;
  answer: boolean;
}

/** A `fetch` that records each chat request's settings and Ollama's count of it. */
function loggingFetch(log: Logged[]): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const response = await fetch(input, init);
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (!url.endsWith("/api/chat") || typeof init?.body !== "string") return response;
    const body = JSON.parse(init.body) as Record<string, unknown>;
    const options = (body.options ?? {}) as Record<string, unknown>;
    const system = ((body.messages as { role: string; content: string }[]) ?? []).find(
      (message) => message.role === "system",
    );
    const entry: Logged = {
      model: String(body.model),
      numCtx: typeof options.num_ctx === "number" ? options.num_ctx : null,
      numPredict: typeof options.num_predict === "number" ? options.num_predict : null,
      truncate: body.truncate,
      think: body.think,
      keepAlive: body.keep_alive,
      status: response.status,
      promptTokens: null,
      answer: system?.content.startsWith("You answer Questions") ?? false,
    };
    log.push(entry);
    void response
      .clone()
      .text()
      .then((text) => {
        for (const line of text.split("\n").reverse()) {
          if (!line.includes('"done":true')) continue;
          const done = JSON.parse(line) as { prompt_eval_count?: number };
          entry.promptTokens = done.prompt_eval_count ?? null;
          break;
        }
      })
      .catch(() => undefined);
    return response;
  }) as typeof fetch;
}

/**
 * The models' lookups for a test: the window at most `MAX_NUM_CTX`, and the
 * citing mode replaced when one is given.
 */
function forTest(models: OllamaModels, citing: CitationSupport | undefined): OllamaModels {
  return {
    ...models,
    describe: async (baseUrl, model) => {
      const profile = await models.describe(baseUrl, model);
      if (!profile) return null;
      const numCtx = Math.min(profile.settings.numCtx, MAX_NUM_CTX);
      return {
        ...profile,
        support: citing ?? profile.support,
        settings: {
          ...profile.settings,
          numCtx,
          outputTokens: outputTokensFor(numCtx, profile.settings.think),
        },
      };
    },
  };
}

async function unload(model: string) {
  await fetch(`${BASE_URL}/api/generate`, {
    method: "POST",
    body: JSON.stringify({ model, keep_alive: 0 }),
  }).catch(() => undefined);
}

describe.runIf(process.env.INCARNAMIND_REAL_OLLAMA === "1")(
  "Real local models through Ollama",
  { timeout: 20 * 60_000 },
  () => {
    test.each(MODELS)("%s answers within its window, citing", async (modelId) => {
      const log: Logged[] = [];
      const core: Core = startCore(await createTempDataFolder(), {
        createChatModel: (spec) =>
          spec.kind === "ollama" && spec.baseUrl
            ? createOllamaChatModel({
                baseUrl: spec.baseUrl,
                modelId: spec.modelId,
                settings: spec.ollama ?? DEFAULT_OLLAMA_SETTINGS,
                fetch: loggingFetch(log),
              })
            : createAiSdkChatModel(spec),
        ollamaModels: forTest(createOllamaModels(), CITING),
      });
      try {
        await core.saveChatProvider({ kind: "ollama", baseUrl: BASE_URL, modelId });
        const sources = await createTempDataFolder();
        const paths = await Promise.all(
          Object.entries(DOCUMENTS).map(([name, text]) => writeSourceFile(sources, name, text)),
        );
        const added = await addAndProcess(core, paths, 120_000);
        await waitForTagging(
          core,
          added.map((document) => document.id),
          (document) => document.tagging !== "pending" && document.tagging !== "tagging",
          300_000,
        );

        // Tagging loaded the model: unload it, so the first Answer shows it loading.
        await unload(modelId);
        const rows: Record<string, unknown>[] = [];
        for (const text of QUESTIONS) {
          const mind = await core.createMind({ title: text.slice(0, 20) });
          const client = await connectToMind(core, mind.id);
          const asked = question(text);
          writeMind(client, [asked]);
          await client.settled();
          const phases: string[] = [];
          const stop = core.on("answer.phase", ({ phase }) => phases.push(phase));
          const started = Date.now();
          const result = await core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id });
          if (!result.asked) throw new Error(JSON.stringify(result));
          const ended = await answerEnded(core, result.answerId);
          stop();
          const seconds = (Date.now() - started) / 1000;
          const payload = ended.payload as Partial<CoreEvents["answer.finished"]> &
            Partial<CoreEvents["answer.failed"]>;
          rows.push({
            question: text.slice(0, 40),
            ended: ended.event,
            error: payload.error?.message.slice(0, 120),
            mode: payload.citationSupport,
            seconds: Number(seconds.toFixed(1)),
            citations: payload.citations?.length,
            found: payload.citations?.filter((each) => each.check === "found").length,
            placed: payload.placedMarkers,
            droppedRecords: payload.droppedRecords,
            droppedMarkers: payload.droppedMarkers,
            phases: phases.join(">"),
            text: answerText(client, result.answerId).slice(0, 300),
            quotes: payload.citations?.map(
              (citation) => `${citation.check} ${citation.checkReason ?? ""}: ${citation.quote}`,
            ),
          });
        }
        // Let the last responses' counts arrive.
        await new Promise((resolve) => setTimeout(resolve, 500));
        console.log(`\n${modelId}: Answers\n${JSON.stringify(rows, null, 1)}`);
        console.log(`${modelId}: chat requests\n${JSON.stringify(log, null, 1)}`);

        expect(rows.every((row) => row.ended === "finished")).toBe(true);
        expect(log.length).toBeGreaterThan(0);
        for (const entry of log) {
          expect(entry.truncate).toBe(false);
          expect(entry.status).toBe(200);
          expect(entry.numCtx).toBeLessThanOrEqual(MAX_NUM_CTX);
          expect(entry.keepAlive).toBe("30m");
          expect(entry.think).toBe(false);
          // Every request fits its window with room for the output, by Ollama's own count.
          if (entry.promptTokens !== null && entry.numCtx !== null && entry.numPredict !== null) {
            expect(entry.promptTokens + entry.numPredict).toBeLessThanOrEqual(entry.numCtx);
          }
        }
        // Every request to a model carries the same window, so Ollama never reloads it.
        expect(new Set(log.map((entry) => entry.numCtx)).size).toBe(1);
      } finally {
        core.close();
        await unload(modelId);
      }
    });
  },
);
