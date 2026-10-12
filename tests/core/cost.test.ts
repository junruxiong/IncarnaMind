import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { APICallError, generateText, streamText } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import type { ChatProvider, Core, MonthlyCap } from "../../src/core";
import { createAiSdkChatModel, InvalidInputError } from "../../src/core";
import { DATABASE_FILE } from "../../src/core/core";
import { costOf, inCurrency, rateAt } from "../../src/core/cost/price";
import { catalogFacts } from "../../src/core/providers/catalog";
import { ollamaCloudPrice } from "../../src/core/providers/catalog/billing";
import type { Price } from "../../src/core/providers/catalog/types";
import { classifyProviderError } from "../../src/core/providers/providerErrors";
import { runItems } from "../../src/core/runs/stop";
import { openDatabase } from "../../src/core/storage";
import { createFakeBrowser, startFakeOpenAI } from "../helpers/chatgpt";
import { createTempDataFolder, manualClock, queryDatabase, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { answerIn, question, writeMind } from "../helpers/minds";

type StreamResult = Awaited<ReturnType<MockLanguageModelV4["doStream"]>>;
type StreamPart = StreamResult["stream"] extends ReadableStream<infer Part> ? Part : never;
type Usage = Extract<StreamPart, { type: "finish" }>["usage"];

/** What a provider reports for one request. */
interface Counted {
  input: number;
  output: number;
  cached?: number;
  written?: number;
}

const usageOf = ({ input, output, cached = 0, written }: Counted): Usage => ({
  inputTokens: { total: input, noCache: input - cached, cacheRead: cached, cacheWrite: written },
  outputTokens: { total: output, text: output, reasoning: undefined },
});

/**
 * A model that answers "OK" to each request and reports the next of `counts`
 * (the last one again after that). With `held`, a streamed reply waits for it
 * before it finishes, so that several requests are open at once.
 */
function reportingModel(counts: readonly Counted[], held?: Promise<void>) {
  let calls = 0;
  const next = () =>
    usageOf(counts[Math.min(calls++, counts.length - 1)] ?? { input: 0, output: 0 });
  return new MockLanguageModelV4({
    doGenerate: async () => ({
      content: [{ type: "text", text: "OK" }],
      finishReason: { unified: "stop", raw: undefined },
      usage: next(),
      warnings: [],
    }),
    doStream: async () => {
      const usage = next();
      const parts: StreamPart[] = [
        { type: "stream-start", warnings: [] },
        { type: "text-start", id: "t" },
        { type: "text-delta", id: "t", delta: "OK" },
        { type: "text-end", id: "t" },
      ];
      return {
        stream: new ReadableStream<StreamPart>({
          async pull(controller) {
            const part = parts.shift();
            if (part) return controller.enqueue(part);
            await held;
            controller.enqueue({
              type: "finish",
              finishReason: { unified: "stop", raw: undefined },
              usage,
            });
            controller.close();
          },
        }),
      };
    },
  });
}

const HAIKU = "claude-haiku-5-5";
const haikuPrice = (): Price => {
  const price = catalogFacts("anthropic", HAIKU)?.price;
  if (!price) throw new Error("The catalog has no price for Claude Haiku 5.5.");
  return price;
};

/** A core whose chat model is `model` on a provider of `kind`, with Questions to its service allowed. */
async function cloudCore(
  model: MockLanguageModelV4,
  provider: { kind: "anthropic" | "openai" | "ollama"; modelId: string } = {
    kind: "anthropic",
    modelId: HAIKU,
  },
  dataDir?: string,
) {
  const folder = dataDir ?? (await createTempDataFolder());
  const clock = manualClock("2026-10-12T09:00:00.000Z");
  const core = startCore(folder, { createChatModel: () => model, now: clock.now });
  // An Ollama cloud model's service is known only once it is asked with: allow it then.
  core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
  const saved = await core.saveChatProvider(
    provider.kind === "ollama" ? provider : { ...provider, apiKey: "sk-test" },
  );
  if (saved.service) await core.allowDataFlow("chat", saved.service.id);
  return { core, provider: saved, dataDir: folder, clock };
}

const setCap = (core: Core, monthlyCap: MonthlyCap | null) =>
  core.updateSettings({ user: { monthlyCap } });

/** One request through the model Answers get, streamed as Answers stream; throws what stopped it. */
async function request(core: Core, maxOutputTokens?: number) {
  const { model } = await core.prepareChatModel();
  const streamed = streamText({
    model,
    prompt: "What is attention?",
    maxRetries: 0,
    maxOutputTokens,
    onError: () => {},
  });
  let text = "";
  for await (const part of streamed.fullStream) {
    if (part.type === "error") throw part.error;
    if (part.type === "text-delta") text += part.text;
  }
  return text;
}

const rows = (dataDir: string) =>
  queryDatabase<{
    billing: string;
    status: string;
    currency: string | null;
    reserved: number | null;
    cost: number | null;
    input_tokens: number | null;
    output_tokens: number | null;
    deleted_at: string | null;
    flow: string;
  }>(dataDir, "SELECT * FROM model_requests ORDER BY created_at, rowid");

const spentUsd = async (core: Core) =>
  (await core.getCostSummary()).spent.find((each) => each.currency === "USD")?.amount ?? 0;

describe("Recording", () => {
  test("the cap is off by default, and the month-to-date figure is what the recorded requests cost", async () => {
    const counts = [
      { input: 12_000, cached: 4_800, output: 750 },
      { input: 9_000, output: 400 },
      { input: 15_000, cached: 9_000, output: 1_200 },
    ];
    const { core, provider, dataDir } = await cloudCore(reportingModel(counts));
    expect((await core.getSettings()).user.monthlyCap).toBeNull();

    for (const _ of counts) await request(core);

    const price = haikuPrice();
    const expected = counts.reduce(
      (sum, each) =>
        sum +
        ((each.input - (each.cached ?? 0)) * price.input +
          (each.cached ?? 0) * (price.cacheRead ?? price.input) +
          each.output * price.output) /
          1_000_000,
      0,
    );
    const summary = await core.getCostSummary();
    expect(summary.spent).toHaveLength(1);
    expect(summary.spent[0]?.currency).toBe("USD");
    expect(summary.spent[0]?.amount).toBeCloseTo(expected, 10);
    expect(summary.cap).toBeNull();
    expect(summary.spentTowardCap).toBeNull();
    expect(summary.providers).toEqual([
      {
        providerId: provider.id,
        billing: "per-token",
        requests: 3,
        spent: [{ currency: "USD", amount: expect.closeTo(expected, 10) }],
        unpricedTokens: 0,
      },
    ]);
    expect(rows(dataDir).map((row) => [row.status, row.input_tokens, row.output_tokens])).toEqual(
      counts.map((each) => ["settled", each.input, each.output]),
    );
  });

  test("an Answer's calls are all recorded, and its cost is what its reported tokens cost", async () => {
    const counts = [{ input: 3_000, output: 200 }];
    const { core, dataDir } = await cloudCore(reportingModel(counts));
    const mind = await core.createMind({ title: "Tides" });
    const client = await connectToMind(core, mind.id);
    const asked = question("What is attention?");
    writeMind(client, [asked]);
    await client.settled();
    const finished = new Promise<{ inputTokens: number; outputTokens: number }>((resolve) =>
      core.on("answer.finished", ({ usage }) => resolve(usage)),
    );

    await core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id });
    const usage = await finished;

    const price = haikuPrice();
    expect(await spentUsd(core)).toBeCloseTo(
      (usage.inputTokens * price.input + usage.outputTokens * price.output) / 1_000_000,
      10,
    );
    expect(rows(dataDir).every((row) => row.flow === "chat" && row.status === "settled")).toBe(
      true,
    );
  });

  test("a request the provider refuses before answering costs nothing, and isn't counted", async () => {
    const refusing = new MockLanguageModelV4({
      doStream: async () => {
        throw new APICallError({
          message: "Invalid key",
          url: "https://api.anthropic.com/v1/messages",
          requestBodyValues: {},
          statusCode: 401,
          isRetryable: false,
        });
      },
    });
    const { core, dataDir } = await cloudCore(refusing);

    await expect(request(core)).rejects.toThrow();

    expect((await core.getCostSummary()).providers).toEqual([]);
    expect(rows(dataDir)).toEqual([expect.objectContaining({ deleted_at: expect.any(String) })]);
  });

  test("a request whose provider reports nothing settles at what came, never above its reservation", async () => {
    const silent = new MockLanguageModelV4({
      doGenerate: async () => ({
        content: [{ type: "text", text: "A short answer." }],
        finishReason: { unified: "stop", raw: undefined },
        usage: {
          inputTokens: {
            total: undefined,
            noCache: undefined,
            cacheRead: undefined,
            cacheWrite: undefined,
          },
          outputTokens: { total: undefined, text: undefined, reasoning: undefined },
        },
        warnings: [],
      }),
    });
    const { core, dataDir } = await cloudCore(silent);
    const { model } = await core.prepareChatModel();

    await generateText({
      model,
      prompt: "What is attention?",
      maxRetries: 0,
      maxOutputTokens: 4_000,
    });

    const [row] = rows(dataDir);
    expect(row?.status).toBe("settled");
    expect(row?.cost).toBeGreaterThan(0);
    expect(row?.cost).toBeLessThan((row?.reserved ?? 0) / 10);
  });

  test("a reply stopped halfway settles at what came, not at all it could have cost", async () => {
    const hanging = new MockLanguageModelV4({
      doStream: async ({ abortSignal }) => {
        const parts: StreamPart[] = [
          { type: "stream-start", warnings: [] },
          { type: "text-start", id: "t" },
          { type: "text-delta", id: "t", delta: "Spring tides come " },
        ];
        return {
          stream: new ReadableStream<StreamPart>({
            pull(controller) {
              const part = parts.shift();
              if (part) return controller.enqueue(part);
              return new Promise<void>((_, reject) => {
                abortSignal?.addEventListener("abort", () =>
                  reject(new DOMException("The request was aborted.", "AbortError")),
                );
              });
            },
          }),
        };
      },
    });
    const { core, dataDir } = await cloudCore(hanging);
    const { model } = await core.prepareChatModel();
    const stop = new AbortController();
    const streamed = streamText({
      model,
      prompt: "When do spring tides come?",
      maxRetries: 0,
      abortSignal: stop.signal,
      onError: () => {},
    });
    for await (const part of streamed.fullStream) {
      if (part.type === "text-delta") stop.abort();
    }

    await expect.poll(() => rows(dataDir)[0]?.status).toBe("settled");
    const [row] = rows(dataDir);
    expect(row?.cost).toBeGreaterThan(0);
    expect(row?.cost).toBeLessThan((row?.reserved ?? 0) / 100);
  });

  test("a provider that can't be reached is sent nothing it bills, and nothing is counted", async () => {
    const unreachable = new MockLanguageModelV4({
      doStream: async () => {
        throw new APICallError({
          message: "Cannot connect to API: connect ECONNREFUSED",
          url: "https://api.anthropic.com/v1/messages",
          requestBodyValues: {},
          cause: new TypeError("fetch failed"),
          isRetryable: false,
        });
      },
    });
    const { core } = await cloudCore(unreachable);

    await expect(request(core)).rejects.toThrow();

    expect((await core.getCostSummary()).spent).toEqual([]);
  });

  test("a model on this computer is free, and nothing is recorded", async () => {
    const { core, dataDir } = await cloudCore(reportingModel([{ input: 500, output: 50 }]), {
      kind: "ollama",
      modelId: "qwen3.5:4b",
    });

    await request(core);

    expect(rows(dataDir)).toEqual([]);
    expect(await core.estimateRun({ items: 10 })).toEqual({ billing: "free" });
  });
});

describe("The monthly cap", () => {
  test("four requests made at once near the cap never pass it: the one that would is refused before anything is sent", async () => {
    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    // Each reports all it reserved for its output: settled, they still fit.
    const model = reportingModel([{ input: 40, output: 100_000 }], held);
    const { core, dataDir } = await cloudCore(model);
    // 100,000 output tokens of Haiku reserve $0.05: three fit under $0.175, a fourth doesn't.
    await setCap(core, { amount: 0.175, currency: "USD" });

    const requests = [0, 1, 2, 3].map(() =>
      request(core, 100_000).then(
        () => "answered",
        (error: unknown) => classifyProviderError(error).kind,
      ),
    );
    await expect.poll(() => rows(dataDir).length).toBe(3);
    const during = await core.getCostSummary();
    expect(during.spentTowardCap?.amount).toBeLessThanOrEqual(0.175);
    release();

    expect((await Promise.all(requests)).sort()).toEqual([
      "answered",
      "answered",
      "answered",
      "cost-cap",
    ]);
    expect(model.doStreamCalls).toHaveLength(3);
    const after = await core.getCostSummary();
    expect(after.spentTowardCap?.amount).toBeLessThanOrEqual(0.175);
    expect(after.spentTowardCap?.amount).toBeGreaterThan(0.15);
  });

  test("a lowered cap applies to the next request, and spending in yuan counts at the fixed rate", async () => {
    const { core } = await cloudCore(reportingModel([{ input: 1_000, output: 100 }]));
    await request(core);
    const spent = await spentUsd(core);

    await setCap(core, { amount: inCurrency(spent, "USD", "CNY") / 2, currency: "CNY" });

    await expect(request(core)).rejects.toSatisfy(
      (error) => classifyProviderError(error).kind === "cost-cap",
    );
    const summary = await core.getCostSummary();
    expect(summary.spentTowardCap?.currency).toBe("CNY");
    expect(summary.spentTowardCap?.amount).toBeCloseTo(inCurrency(spent, "USD", "CNY"), 10);
  });

  test("an Answer stopped by the cap says so, and nothing is sent", async () => {
    const model = reportingModel([{ input: 3_000, output: 200 }]);
    const { core } = await cloudCore(model);
    await setCap(core, { amount: 0, currency: "USD" });
    const mind = await core.createMind({ title: "Tides" });
    const client = await connectToMind(core, mind.id);
    const asked = question("What is attention?");
    writeMind(client, [asked]);
    await client.settled();
    const failed = new Promise((resolve) =>
      core.on("answer.failed", ({ error }) => resolve(error)),
    );

    const result = await core.askQuestion({ mindId: mind.id, questionId: asked.attrs.id });

    expect(await failed).toMatchObject({ kind: "cost-cap" });
    if (!result.asked) throw new Error("The Question wasn't asked.");
    await client.settled();
    expect(answerIn(client, result.answerId).attrs).toMatchObject({
      status: "failed",
      errorKind: "cost-cap",
    });
    expect(model.doStreamCalls).toHaveLength(0);
  });

  test("a crash with open reservations settles them at their reserved amount on the next start", async () => {
    // A reply that never ends: the app quits while it is open, as a crash would leave it.
    const hung = new MockLanguageModelV4({
      doStream: async () => ({
        stream: new ReadableStream<StreamPart>({ pull: () => new Promise(() => {}) }),
      }),
    });
    const first = await cloudCore(hung);
    void request(first.core, 2_000).catch(() => undefined);
    await expect.poll(() => rows(first.dataDir).map((row) => row.status)).toEqual(["reserved"]);
    const [open] = rows(first.dataDir);
    first.core.close();

    const again = startCore(first.dataDir, { createChatModel: () => hung });

    expect(rows(first.dataDir)).toEqual([
      expect.objectContaining({ status: "settled", cost: open?.reserved }),
    ]);
    expect(await spentUsd(again)).toBeCloseTo(open?.reserved ?? Number.NaN, 12);
  });
});

describe("Plans", () => {
  test("a ChatGPT plan request records its tokens, reserves no dollars, and isn't stopped by the cap", async () => {
    const clock = manualClock("2026-10-12T09:00:00.000Z");
    const fake = await startFakeOpenAI({ now: () => clock.now().getTime() });
    const dataDir = await createTempDataFolder();
    const core = startCore(dataDir, {
      browser: createFakeBrowser(fake),
      now: clock.now,
      createChatModel: createAiSdkChatModel,
      chatGptPlan: fake.endpoints,
    });
    expect((await core.signInToChatGpt()).ok).toBe(true);
    core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
    const provider: ChatProvider = await core.saveChatProvider({
      kind: "chatgpt",
      modelId: "gpt-6-luna",
    });
    await setCap(core, { amount: 0, currency: "USD" });

    expect(await request(core)).toBe("Hello from your plan.");

    expect(rows(dataDir)).toEqual([
      expect.objectContaining({
        billing: "plan",
        status: "settled",
        currency: null,
        reserved: null,
        cost: null,
        input_tokens: 12,
        output_tokens: 5,
      }),
    ]);
    const summary = await core.getCostSummary();
    expect(summary.spent).toEqual([]);
    expect(summary.providers).toEqual([
      { providerId: provider.id, billing: "plan", requests: 1, spent: [], unpricedTokens: 0 },
    ]);
    expect(await core.estimateRun({ items: 50 })).toEqual({ billing: "plan" });
  });

  test("an Ollama Cloud request is priced at Ollama's rates, and reserved against the cap", async () => {
    const counts = { input: 20_000, cached: 5_000, output: 900 };
    const { core, provider } = await cloudCore(reportingModel([counts]), {
      kind: "ollama",
      modelId: "gpt-oss:120b-cloud",
    });
    expect(provider.service).toBeNull();

    await request(core);

    const price = ollamaCloudPrice("gpt-oss:120b-cloud");
    if (!price) throw new Error("No price for gpt-oss:120b on Ollama Cloud.");
    expect(price).toMatchObject({ currency: "USD", input: 0.15, output: 0.6 });
    const summary = await core.getCostSummary();
    expect(summary.providers[0]).toMatchObject({ billing: "per-token", requests: 1 });
    expect(await spentUsd(core)).toBeCloseTo(
      costOf(
        {
          inputTokens: counts.input,
          cachedInputTokens: counts.cached,
          outputTokens: counts.output,
        },
        price,
        new Date("2026-10-12T09:00:00.000Z"),
      ),
      12,
    );

    await setCap(core, { amount: 0, currency: "USD" });
    await expect(request(core)).rejects.toSatisfy(
      (error) => classifyProviderError(error).kind === "cost-cap",
    );
  });
});

describe("A run stopped at a limit", () => {
  const items = ["one", "two", "three", "four"];

  test("a usage limit refused mid-run stops it, its unfinished items are not checked, and it says when the limit resets", async () => {
    let calls = 0;
    const model = new MockLanguageModelV4({
      doGenerate: async () => {
        if (++calls === 2) {
          // Ollama answers a used-up allowance with 402, here saying when to try again.
          throw new APICallError({
            message: "You've reached your usage limit, please upgrade to continue",
            url: "http://127.0.0.1:11434/api/chat",
            requestBodyValues: {},
            statusCode: 402,
            responseHeaders: { "retry-after": "3600" },
            isRetryable: false,
          });
        }
        return {
          content: [{ type: "text", text: "OK" }],
          finishReason: { unified: "stop", raw: undefined },
          usage: usageOf({ input: 100, output: 10 }),
          warnings: [],
        };
      },
    });
    const { core } = await cloudCore(model, { kind: "ollama", modelId: "glm-5.3:cloud" });
    const { model: prepared } = await core.prepareChatModel();
    const before = Date.now();

    const run = await runItems(items, (item) =>
      generateText({ model: prepared, prompt: item, maxRetries: 0 }).then((result) => result.text),
    );

    expect(run.outcomes.map((each) => each.status)).toEqual([
      "done",
      "not-checked",
      "not-checked",
      "not-checked",
    ]);
    expect(run.stop).toMatchObject({ reason: "plan-limit" });
    const resetsAt = Date.parse(run.stop?.reason === "plan-limit" ? (run.stop.resetsAt ?? "") : "");
    expect(resetsAt).toBeGreaterThanOrEqual(before + 3_600_000);
    expect(resetsAt).toBeLessThan(Date.now() + 3_600_000 + 1_000);
    expect(calls).toBe(2);
  });

  test("the ChatGPT plan's usage limit mid-answer stops a run the same way, with no reset time when OpenAI gives none", async () => {
    const clock = manualClock("2026-10-12T09:00:00.000Z");
    const fake = await startFakeOpenAI({ now: () => clock.now().getTime() });
    const core = startCore(await createTempDataFolder(), {
      browser: createFakeBrowser(fake),
      now: clock.now,
      createChatModel: createAiSdkChatModel,
      chatGptPlan: fake.endpoints,
    });
    expect((await core.signInToChatGpt()).ok).toBe(true);
    core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
    await core.saveChatProvider({ kind: "chatgpt", modelId: "gpt-6-luna" });
    const { model } = await core.prepareChatModel();

    const run = await runItems(items, async (item) => {
      if (item === "two") {
        fake.reply = {
          kind: "fails-midway",
          error: { code: "subscription_sharing_usage_limit_exceeded", message: "Limit reached." },
        };
      }
      const streamed = streamText({ model, prompt: item, maxRetries: 0, onError: () => {} });
      let text = "";
      for await (const part of streamed.fullStream) {
        if (part.type === "error") throw part.error;
        if (part.type === "text-delta") text += part.text;
      }
      return text;
    });

    expect(run.outcomes.map((each) => each.status)).toEqual([
      "done",
      "not-checked",
      "not-checked",
      "not-checked",
    ]);
    expect(run.stop).toEqual({ reason: "plan-limit", resetsAt: null });
    expect(fake.responses()).toHaveLength(2);
  });

  test("the cap stops a run the same way: the items after it aren't sent", async () => {
    // Each uses all of its output limit: one fits under the cap, a second wouldn't.
    const model = reportingModel([{ input: 50, output: 1_000 }]);
    const { core } = await cloudCore(model);
    // One request of up to 1,000 output tokens reserves about $0.0005.
    await setCap(core, { amount: 0.0007, currency: "USD" });
    const { model: prepared } = await core.prepareChatModel();

    const run = await runItems(items, (item) =>
      generateText({ model: prepared, prompt: item, maxRetries: 0, maxOutputTokens: 1_000 }),
    );

    expect(run.outcomes.map((each) => each.status)).toEqual([
      "done",
      "not-checked",
      "not-checked",
      "not-checked",
    ]);
    expect(run.stop).toEqual({ reason: "cost-cap" });
    expect(model.doGenerateCalls).toHaveLength(1);
  });

  test("any other failure fails its item, and the run goes on", async () => {
    let calls = 0;
    const model = new MockLanguageModelV4({
      doGenerate: async () => {
        if (++calls === 1) {
          throw new APICallError({
            message: "Overloaded",
            url: "https://api.anthropic.com/v1/messages",
            requestBodyValues: {},
            statusCode: 529,
            isRetryable: false,
          });
        }
        return {
          content: [{ type: "text", text: "OK" }],
          finishReason: { unified: "stop", raw: undefined },
          usage: usageOf({ input: 10, output: 2 }),
          warnings: [],
        };
      },
    });
    const { core } = await cloudCore(model);
    const { model: prepared } = await core.prepareChatModel();

    const run = await runItems(["one", "two"], (item) =>
      generateText({ model: prepared, prompt: item, maxRetries: 0 }),
    );

    expect(run.outcomes.map((each) => each.status)).toEqual(["failed", "done"]);
    expect(run.stop).toBeNull();
  });
});

describe("The estimate before a large run", () => {
  /** Real Answers' calls, as their providers reported them (see the fixture's "about"). */
  const recorded = async (): Promise<Record<string, Counted[][]>> =>
    JSON.parse(
      await readFile(join(import.meta.dirname, "../fixtures/cost/recorded-answers.json"), "utf8"),
    ).answers;

  test.each([
    { key: "anthropic/claude-haiku-5-5", kind: "anthropic" as const },
    { key: "openai/gpt-6-luna", kind: "openai" as const },
  ])(
    "on $key, recorded Answers forecast the next run: tokens within 15%, price from 10% below to 2.5 times above",
    async ({ key, kind }) => {
      const answers = (await recorded())[key] ?? [];
      const [modelId] = key.split("/").slice(1);
      const history = answers.slice(0, 2).flat();
      const run = answers.slice(2);
      const model = reportingModel([...history, ...run.flat()]);
      const { core } = await cloudCore(model, { kind, modelId: modelId ?? "" });
      for (const _ of history) await request(core);

      const estimate = await core.estimateRun({ items: run.length });

      const price = catalogFacts(kind, modelId ?? "")?.price;
      if (!price || estimate.billing !== "per-token" || !estimate.cost) {
        throw new Error(`No price for ${key}.`);
      }
      const at = new Date("2026-10-12T09:00:00.000Z");
      const actual = run.flat().reduce(
        (sum, call) => ({
          tokens: sum.tokens + call.input + call.output,
          cost:
            sum.cost +
            costOf(
              {
                inputTokens: call.input,
                cachedInputTokens: call.cached,
                cacheWriteTokens: call.written,
                outputTokens: call.output,
              },
              price,
              at,
              { cacheWrites: kind === "anthropic" },
            ),
        }),
        { tokens: 0, cost: 0 },
      );
      expect(estimate.calls).toBe(run.flat().length);
      expect(estimate.tokens / actual.tokens).toBeGreaterThan(0.85);
      expect(estimate.tokens / actual.tokens).toBeLessThan(1.15);
      expect(estimate.cost.amount / actual.cost).toBeGreaterThan(0.9);
      expect(estimate.cost.amount / actual.cost).toBeLessThan(2.5);
    },
  );

  test("before two Answers are recorded on a model, it assumes an average Answer, which errs high for short ones", async () => {
    const answers = (await recorded())["anthropic/claude-haiku-5-5"] ?? [];
    const { core } = await cloudCore(reportingModel([{ input: 0, output: 0 }]));

    const estimate = await core.estimateRun({ items: answers.length });

    const tokens = answers.flat().reduce((sum, call) => sum + call.input + call.output, 0);
    expect(estimate).toMatchObject({
      billing: "per-token",
      calls: 3 * answers.length,
      tokens: answers.length * 12_750,
      cap: null,
    });
    if (estimate.billing !== "per-token") throw new Error("Not billed per token.");
    expect(estimate.tokens).toBeGreaterThan(tokens);
  });

  test("under a cap, it says how much of the month is used", async () => {
    const { core } = await cloudCore(reportingModel([{ input: 1_000, output: 100 }]));
    await request(core);
    await setCap(core, { amount: 20, currency: "USD" });

    const estimate = await core.estimateRun({ items: 10 });

    expect(estimate).toMatchObject({
      cap: { amount: 20, currency: "USD" },
      spentTowardCap: { currency: "USD", amount: await spentUsd(core) },
    });
  });
});

describe("Prices", () => {
  const deepseek = catalogFacts("deepseek", "deepseek-flash")?.price;

  test("DeepSeek's off-peak hours are half price, by Beijing time, and all weekend", () => {
    if (!deepseek) throw new Error("No DeepSeek price.");
    // Monday 2026-10-12: 10:00 and 15:00 in Beijing are peak; 13:00 and 20:00 aren't.
    expect(rateAt(deepseek, new Date("2026-10-12T02:00:00Z"))).toBe(1);
    expect(rateAt(deepseek, new Date("2026-10-12T07:00:00Z"))).toBe(1);
    expect(rateAt(deepseek, new Date("2026-10-12T05:00:00Z"))).toBe(0.5);
    expect(rateAt(deepseek, new Date("2026-10-12T12:00:00Z"))).toBe(0.5);
    // Saturday 10:00 in Beijing.
    expect(rateAt(deepseek, new Date("2026-10-17T02:00:00Z"))).toBe(0.5);
  });

  test("SiliconFlow's night discount holds every day, and Ollama's DeepSeek follows UTC", () => {
    const siliconflow = catalogFacts("siliconflow", "deepseek-ai/DeepSeek-V4-Flash")?.price;
    const ollama = ollamaCloudPrice("deepseek-v4-pro:cloud");
    if (!siliconflow || !ollama) throw new Error("A price is missing.");
    // Saturday 03:00 in Beijing: SiliconFlow's discount hours.
    expect(rateAt(siliconflow, new Date("2026-10-16T19:00:00Z"))).toBe(0.5);
    expect(rateAt(siliconflow, new Date("2026-10-17T02:00:00Z"))).toBe(1);
    expect(rateAt(ollama, new Date("2026-10-12T13:00:00Z"))).toBe(1);
    expect(rateAt(ollama, new Date("2026-10-12T19:00:00Z"))).toBe(0.5);
  });

  test("cached input costs its own rate, and only Anthropic bills cache writes above the input price", () => {
    const price = haikuPrice();
    const at = new Date("2026-10-12T09:00:00Z");
    const plain = costOf({ inputTokens: 10_000, outputTokens: 0 }, price, at);
    const cached = costOf(
      { inputTokens: 10_000, cachedInputTokens: 10_000, outputTokens: 0 },
      price,
      at,
    );
    const written = { inputTokens: 10_000, cacheWriteTokens: 10_000, outputTokens: 0 };
    expect(cached).toBeCloseTo(plain / 10, 12);
    expect(costOf(written, price, at, { cacheWrites: true })).toBeCloseTo(plain * 1.25, 12);
    // OpenAI reports what it wrote to its cache, at no extra cost.
    expect(costOf(written, price, at)).toBeCloseTo(plain, 12);
  });

  test("Ollama Cloud's price is found by the model's name in Ollama, or its only size", () => {
    expect(ollamaCloudPrice("gpt-oss:20b-cloud")?.input).toBe(0.07);
    expect(ollamaCloudPrice("gemma4:31b-cloud")?.input).toBe(0.14);
    // Two sizes are priced: a bare name could be either.
    expect(ollamaCloudPrice("gpt-oss:cloud")).toBeUndefined();
    expect(ollamaCloudPrice("unknown-model:cloud")).toBeUndefined();
  });
});

describe("Settings", () => {
  test("the cap takes an amount in dollars or yuan, and nothing else", async () => {
    const core = startCore(await createTempDataFolder());

    await setCap(core, { amount: 20, currency: "USD" });
    expect((await core.getSettings()).user.monthlyCap).toEqual({ amount: 20, currency: "USD" });
    const refused: unknown[] = [
      { amount: -1, currency: "USD" },
      { amount: 5, currency: "EUR" },
      { amount: 5 },
    ];
    for (const monthlyCap of refused) {
      await expect(core.updateSettings({ user: { monthlyCap } } as never)).rejects.toThrow(
        InvalidInputError,
      );
    }
    await setCap(core, null);
    expect((await core.getSettings()).user.monthlyCap).toBeNull();
  });

  test("records stay in this computer's database", async () => {
    const { core, dataDir } = await cloudCore(reportingModel([{ input: 100, output: 10 }]));
    await request(core);
    core.close();
    const db = openDatabase(join(dataDir, DATABASE_FILE));
    try {
      expect(db.get("SELECT count(*) AS count FROM model_requests")).toEqual({ count: 1 });
    } finally {
      db.close();
    }
    expect(await readFile(join(dataDir, DATABASE_FILE))).toBeTruthy();
  });
});
