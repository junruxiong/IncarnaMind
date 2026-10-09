/**
 * The Run engine's contract (src/core/runs/engine.ts; docs/designs/
 * agent-extensibility.md §4.5 and P3): what every engine that runs the
 * Tool-calling loop must do, whichever library it is built on. Parameterised
 * by an engine factory, so a second engine runs the same suite. The model's
 * moves are scripted (`MockLanguageModelV4`), so only the engine is tested.
 * Ported from the engine bake-off's contract suite (`prototype/engine-bakeoff`,
 * tests/01-contract.test.ts).
 */
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { APICallError } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import { createAiSdkRunEngine } from "../../src/core/runs/aiSdkEngine";
import type {
  GateDecision,
  RunEngine,
  RunEvent,
  RunMessage,
  RunRequest,
  RunTool,
  RunToolCall,
  RunWindow,
} from "../../src/core/runs/engine";
import { controlledModel, type ModelCall, scriptedModel } from "../helpers/models";

/** The engines that must pass: one today; another loop library adds itself here. */
const ENGINES: { name: string; create: () => RunEngine }[] = [
  { name: "AI SDK 7", create: createAiSdkRunEngine },
];

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

interface Run {
  events: RunEvent[];
  text: string;
  finished: readonly RunMessage[] | null;
  /** When the event stream ended. */
  settledAt: number;
}

/** Runs one Run to its end; `onEvent` may steer or stop it. */
async function collect(
  engine: RunEngine,
  request: RunRequest,
  onEvent?: (event: RunEvent) => void,
): Promise<Run> {
  const events: RunEvent[] = [];
  let text = "";
  let finished: readonly RunMessage[] | null = null;
  for await (const event of engine.run(request)) {
    events.push(event);
    if (event.type === "text-delta") text += event.text;
    if (event.type === "finished") finished = event.messages;
    onEvent?.(event);
  }
  return { events, text, finished, settledAt: performance.now() };
}

/** A gate decided per call: run, deny, or wait for `release`; it records each call it is asked about. */
function scriptedGate(decide: (call: RunToolCall) => "run" | "deny" | "wait" = () => "run") {
  const asked: RunToolCall[] = [];
  const waiting = new Map<string, (allowed: boolean) => void>();
  return {
    asked,
    release(id: string, allowed: boolean) {
      waiting.get(id)?.(allowed);
    },
    async gate(call: RunToolCall): Promise<GateDecision> {
      asked.push(call);
      const decision = decide(call);
      const allowed =
        decision === "wait"
          ? await new Promise<boolean>((resolve) => waiting.set(call.id, resolve))
          : decision === "run";
      return allowed
        ? { run: true }
        : { run: false, result: `The User denied ${call.tool}. Carry on without it.` };
    },
  };
}

interface ToolCallRecord {
  input: Record<string, unknown>;
  aborted: boolean;
  startedAt: number;
}

/** A Tool that records its calls, waits `delayMs` (until its signal aborts) and returns `result`. */
function testTool(
  name: string,
  options: {
    delayMs?: number;
    result?: string;
    schema?: Record<string, unknown>;
  } = {},
): RunTool & { calls: ToolCallRecord[] } {
  const calls: ToolCallRecord[] = [];
  return {
    name,
    description: `The ${name} Tool.`,
    inputSchema: options.schema ?? {
      type: "object",
      properties: { query: { type: "string", description: "What to look for." } },
      required: ["query"],
    },
    calls,
    async call(input, { signal }) {
      const record = { input, aborted: false, startedAt: performance.now() };
      calls.push(record);
      if (options.delayMs) {
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, options.delayMs);
          signal.addEventListener(
            "abort",
            () => {
              clearTimeout(timer);
              record.aborted = true;
              reject(signal.reason);
            },
            { once: true },
          );
        });
      }
      return options.result ?? `${name} found ${JSON.stringify(input)}`;
    },
  };
}

/** A request with defaults: one question, no Tools, every call allowed. */
function request(model: MockLanguageModelV4, overrides: Partial<RunRequest> = {}): RunRequest {
  return {
    model,
    instructions: "You are IncarnaMind. Answer the User.",
    messages: [{ role: "user", text: "When is high water?" }],
    tools: [],
    maxSteps: 10,
    gate: scriptedGate().gate,
    signal: new AbortController().signal,
    ...overrides,
  };
}

/** What the model read of each Tool result in request `index`: its kind and text. */
function resultsSent(model: MockLanguageModelV4, index: number) {
  return (model.doStreamCalls[index]?.prompt ?? []).flatMap((message) =>
    message.role === "tool"
      ? message.content.flatMap((part) =>
          part.type === "tool-result" &&
          (part.output.type === "text" || part.output.type === "error-text")
            ? [{ tool: part.toolName, type: part.output.type, text: part.output.value }]
            : [],
        )
      : [],
  );
}

/** The roles of request `index`'s messages, and the text of its user messages. */
function rolesSent(model: MockLanguageModelV4, index: number) {
  return (model.doStreamCalls[index]?.prompt ?? []).map((message) => message.role);
}

const SEARCH = { tool: "search_documents", input: { query: "high water" } };

/** A model that calls the Tools given, one step after another, then writes `text`. */
function stepsModel(steps: { tool: string; input: unknown }[][], text = "High water is at six.") {
  return scriptedModel((call: ModelCall) => {
    const calls = steps[call.index];
    return calls ? { calls } : { text };
  });
}

describe.each(ENGINES)("A Run engine: $name", ({ create }) => {
  test("a Tool call, its result, then text; a step's Tool events come before its end", async () => {
    const search = testTool("search_documents");
    const model = scriptedModel((call) =>
      call.index === 0
        ? { text: "Let me look.", calls: [SEARCH] }
        : { text: "High water is at six." },
    );

    const run = await collect(create(), request(model, { tools: [search] }));

    expect(search.calls.map((call) => call.input)).toEqual([{ query: "high water" }]);
    expect(run.events.map((event) => event.type)).toEqual([
      "text-delta",
      "text-delta",
      "tool-call",
      "tool-result",
      "step-finished",
      "text-delta",
      "text-delta",
      "text-delta",
      "step-finished",
      "finished",
    ]);
    expect(run.events).toContainEqual({
      type: "tool-call",
      id: "call-0-0",
      tool: "search_documents",
      input: { query: "high water" },
    });
    expect(run.events).toContainEqual({ type: "tool-result", id: "call-0-0", ok: true });
    expect(resultsSent(model, 1)).toEqual([
      {
        tool: "search_documents",
        type: "text",
        text: 'search_documents found {"query":"high water"}',
      },
    ]);
    expect(model.doStreamCalls[0]?.tools?.map((each) => each.name)).toEqual(["search_documents"]);
    expect(run.finished).toEqual([
      {
        role: "assistant",
        text: "Let me look.",
        toolCalls: [{ id: "call-0-0", tool: "search_documents", input: { query: "high water" } }],
      },
      {
        role: "tool",
        id: "call-0-0",
        tool: "search_documents",
        result: 'search_documents found {"query":"high water"}',
        ok: true,
      },
      { role: "assistant", text: "High water is at six.", toolCalls: [] },
    ]);
  });

  test("a denied gate: the Tool never runs, the model reads the gate's text as an ordinary result, and the Run goes on", async () => {
    const close = testTool("close_issue");
    const gate = scriptedGate(() => "deny");
    const model = stepsModel(
      [[{ tool: "close_issue", input: { query: "42" } }]],
      "I couldn't close it.",
    );

    const run = await collect(create(), request(model, { tools: [close], gate: gate.gate }));

    expect(close.calls).toEqual([]);
    expect(gate.asked).toEqual([{ id: "call-0-0", tool: "close_issue", input: { query: "42" } }]);
    expect(resultsSent(model, 1)).toEqual([
      {
        tool: "close_issue",
        type: "text",
        text: "The User denied close_issue. Carry on without it.",
      },
    ]);
    expect(run.events).toContainEqual({ type: "tool-result", id: "call-0-0", ok: true });
    expect(run.finished?.find((message) => message.role === "tool")).toMatchObject({ ok: true });
    expect(run.text).toBe("I couldn't close it.");
  });

  test("a gate that fails fails the call: the Tool never runs, and the model reads why", async () => {
    const close = testTool("close_issue");
    const model = stepsModel(
      [[{ tool: "close_issue", input: { query: "42" } }]],
      "It couldn't run.",
    );

    const run = await collect(
      create(),
      request(model, {
        tools: [close],
        gate: async () => {
          throw new Error("That script can't run here.");
        },
      }),
    );

    expect(close.calls).toEqual([]);
    expect(run.events).toContainEqual({ type: "tool-result", id: "call-0-0", ok: false });
    expect(resultsSent(model, 1)).toEqual([
      {
        tool: "close_issue",
        type: "error-text",
        text: expect.stringContaining("That script can't run here."),
      },
    ]);
    expect(run.finished?.find((message) => message.role === "tool")).toMatchObject({ ok: false });
  });

  test("Stop while the gate waits: the Run settles at once, the Tool never runs, and nothing more is sent", async () => {
    const close = testTool("close_issue");
    const gate = scriptedGate(() => "wait");
    const controller = new AbortController();
    const model = stepsModel([[{ tool: "close_issue", input: { query: "42" } }]], "never");
    let stoppedAt = 0;

    const run = await collect(
      create(),
      request(model, { tools: [close], gate: gate.gate, signal: controller.signal }),
      (event) => {
        if (event.type !== "tool-call") return;
        setTimeout(() => {
          stoppedAt = performance.now();
          controller.abort();
        }, 50);
      },
    );
    await sleep(100);

    expect(run.settledAt - stoppedAt).toBeLessThan(1_000);
    expect(gate.asked).toHaveLength(1);
    expect(close.calls).toEqual([]);
    expect(model.doStreamCalls).toHaveLength(1);
    expect(run.events.map((event) => event.type)).not.toContain("finished");
    expect(run.events.map((event) => event.type)).not.toContain("failed");
  });

  test("Stop during a Tool call: the Tool's signal fires, and the Run settles at once", async () => {
    const search = testTool("search_documents", { delayMs: 10_000 });
    const controller = new AbortController();
    const model = stepsModel([[SEARCH]], "never");
    let stoppedAt = 0;

    const run = await collect(
      create(),
      request(model, { tools: [search], signal: controller.signal }),
      (event) => {
        if (event.type !== "tool-call") return;
        setTimeout(() => {
          stoppedAt = performance.now();
          controller.abort();
        }, 50);
      },
    );

    expect(run.settledAt - stoppedAt).toBeLessThan(1_000);
    expect(search.calls.map((call) => call.aborted)).toEqual([true]);
    expect(model.doStreamCalls).toHaveLength(1);
  });

  test("Stop while the model writes: its request is closed, and the Run settles at once", async () => {
    const controlled = controlledModel();
    const controller = new AbortController();
    let stoppedAt = 0;
    const running = collect(create(), request(controlled.model, { signal: controller.signal }));
    await controlled.requested();
    controlled.push("High water ");

    await sleep(50);
    stoppedAt = performance.now();
    controller.abort();
    const run = await running;

    expect(run.settledAt - stoppedAt).toBeLessThan(1_000);
    expect(controlled.aborted).toBe(true);
    expect(run.events.map((event) => event.type)).not.toContain("finished");
  });

  test("the last step is offered no Tools: it must write", async () => {
    const search = testTool("search_documents");
    const model = stepsModel([[SEARCH], [SEARCH]], "Written.");

    const run = await collect(create(), request(model, { tools: [search], maxSteps: 3 }));

    expect(model.doStreamCalls.map((call) => call.toolChoice?.type)).toEqual([
      "auto",
      "auto",
      "none",
    ]);
    expect(run.text).toBe("Written.");
  });

  test("the window: every result the model reads is cut to fit, Tools are withheld when there's no room, and each step's usage is reported before its end", async () => {
    const search = testTool("search_documents", { result: "x".repeat(5_000) });
    const close = testTool("close_issue");
    const gate = scriptedGate((call) => (call.tool === "close_issue" ? "deny" : "run"));
    const log: string[] = [];
    let room = true;
    const window: RunWindow = {
      fitResult: (text, tool) => `${tool}: ${text.slice(0, 20)}[cut]`,
      canCallTools: () => room,
      stepFinished: (usage) => {
        log.push(`usage ${usage.inputTokens}/${usage.outputTokens}`);
        room = false;
      },
    };
    const model = stepsModel([[SEARCH, { tool: "close_issue", input: { query: "42" } }]], "Done.");

    await collect(
      create(),
      request(model, { tools: [search, close], gate: gate.gate, window }),
      (event) => {
        if (event.type === "step-finished") log.push("step-finished");
      },
    );

    expect(resultsSent(model, 1)).toEqual([
      {
        tool: "search_documents",
        type: "text",
        text: "search_documents: xxxxxxxxxxxxxxxxxxxx[cut]",
      },
      { tool: "close_issue", type: "text", text: "close_issue: The User denied clos[cut]" },
    ]);
    expect(model.doStreamCalls.map((call) => call.toolChoice?.type)).toEqual(["auto", "none"]);
    expect(log).toEqual(["usage 5/1", "step-finished", "usage 5/1", "step-finished"]);
  });

  test("steering is taken after the current Tool calls, sent once, and kept in the history", async () => {
    const search = testTool("search_documents", { delayMs: 100 });
    const queued: RunMessage[] = [];
    let steered = false;
    const model = scriptedModel((call) =>
      call.index === 0
        ? {
            calls: [
              { tool: "search_documents", input: { query: "alpha" } },
              { tool: "search_documents", input: { query: "beta" } },
            ],
          }
        : call.index === 1
          ? { calls: [{ tool: "search_documents", input: { query: "gamma" } }] }
          : { text: "Gamma is at six." },
    );

    const run = await collect(
      create(),
      request(model, { tools: [search], steering: () => queued.splice(0) }),
      (event) => {
        if (event.type === "tool-call" && !steered) {
          steered = true;
          queued.push({ role: "user", text: "Only gamma." });
        }
      },
    );

    // After both of the first step's calls, before the second request.
    expect(rolesSent(model, 1)).toEqual(["system", "user", "assistant", "tool", "user"]);
    expect(model.doStreamCalls[1]?.prompt.at(-1)?.content).toEqual([
      { type: "text", text: "Only gamma." },
    ]);
    // Once, and still there a step later.
    expect(rolesSent(model, 2)).toEqual([
      "system",
      "user",
      "assistant",
      "tool",
      "user",
      "assistant",
      "tool",
    ]);
    expect(run.finished?.map((message) => message.role)).toEqual([
      "assistant",
      "tool",
      "tool",
      "user",
      "assistant",
      "tool",
      "assistant",
    ]);
    expect(run.text).toBe("Gamma is at six.");
  });

  test("the calls of one step are each gated as they arrive, so they can wait for the User at once", async () => {
    const search = testTool("search_documents");
    const gate = scriptedGate(() => "wait");
    const model = stepsModel(
      [
        [
          { tool: "search_documents", input: { query: "a" } },
          { tool: "search_documents", input: { query: "b" } },
          { tool: "search_documents", input: { query: "c" } },
        ],
      ],
      "Three results.",
    );

    const running = collect(create(), request(model, { tools: [search], gate: gate.gate }));
    await expect.poll(() => gate.asked.length).toBe(3);
    expect(search.calls).toEqual([]);
    for (const call of gate.asked) gate.release(call.id, true);
    const run = await running;

    expect(search.calls.map((call) => call.input)).toEqual([
      { query: "a" },
      { query: "b" },
      { query: "c" },
    ]);
    expect(run.text).toBe("Three results.");
  });

  test('a provider\'s refusal before any output is "refused": Tools, a temperature, or a request too long for the window', async () => {
    const refusing = (message: string) =>
      scriptedModel(() => ({ error: { status: 400, message } }));
    const tooLong = "request (9000 tokens) exceeds the available context size (8192 tokens)";
    const window: RunWindow = {
      fitResult: (text) => text,
      canCallTools: () => true,
      stepFinished: () => undefined,
    };
    const last = async (model: MockLanguageModelV4, overrides: Partial<RunRequest>) =>
      (await collect(create(), request(model, overrides))).events.at(-1);

    expect(
      await last(refusing("registry.ollama.ai/library/gemma:2b does not support tools"), {
        tools: [testTool("search_documents")],
      }),
    ).toEqual({ type: "refused", what: "tools" });
    expect(
      await last(
        refusing("Unsupported parameter: 'temperature' is not supported with this model."),
        { temperature: 0.2 },
      ),
    ).toEqual({ type: "refused", what: "temperature" });
    expect(await last(refusing(tooLong), { window })).toEqual({
      type: "refused",
      what: "too-long",
      promptTokens: 9000,
    });
    // Without a window it is a failure like any other.
    expect(await last(refusing(tooLong), {})).toMatchObject({
      type: "failed",
      error: { kind: "too-long" },
    });
  });

  test('a failure after output is "failed", never "refused"; and the engine never throws', async () => {
    const refusal = new APICallError({
      message: "This model does not support tools.",
      url: "http://127.0.0.1:11434/v1/chat/completions",
      requestBodyValues: {},
      statusCode: 400,
      isRetryable: false,
    });
    const midway = new MockLanguageModelV4({
      doStream: async () => ({
        stream: new ReadableStream({
          start(controller) {
            controller.enqueue({ type: "stream-start", warnings: [] });
            controller.enqueue({ type: "text-start", id: "answer" });
            controller.enqueue({ type: "text-delta", id: "answer", delta: "High water" });
            controller.enqueue({ type: "error", error: refusal });
            controller.close();
          },
        }),
      }),
    });
    const broken = new MockLanguageModelV4({
      doStream: async () => {
        throw new Error("Something broke.");
      },
    });

    const afterOutput = await collect(
      create(),
      request(midway, { tools: [testTool("search_documents")] }),
    );
    const thrown = await collect(create(), request(broken));

    expect(afterOutput.events.at(-1)).toMatchObject({ type: "failed" });
    expect(thrown.events).toEqual([
      { type: "failed", error: expect.objectContaining({ message: "Something broke." }) },
    ]);
  });

  test("a history that ends in a Tool result continues from it (a resumed Run)", async () => {
    const model = scriptedModel(() => ({ text: "Resumed: the issue is closed." }));
    const messages: RunMessage[] = [
      { role: "user", text: "Close issue 42." },
      {
        role: "assistant",
        text: "",
        toolCalls: [{ id: "call_x", tool: "close_issue", input: { query: "42" } }],
      },
      { role: "tool", id: "call_x", tool: "close_issue", result: "Closed issue #42.", ok: true },
    ];

    const run = await collect(
      create(),
      request(model, { messages, tools: [testTool("close_issue")] }),
    );

    expect(rolesSent(model, 0)).toEqual(["system", "user", "assistant", "tool"]);
    expect(resultsSent(model, 0)).toEqual([
      { tool: "close_issue", type: "text", text: "Closed issue #42." },
    ]);
    expect(run.text).toBe("Resumed: the issue is closed.");
    expect(run.finished).toEqual([
      { role: "assistant", text: "Resumed: the issue is closed.", toolCalls: [] },
    ]);
  });

  test("a follow-up runs on the first Run's messages", async () => {
    const search = testTool("search_documents");
    const model = scriptedModel((call) =>
      call.index === 0 ? { calls: [SEARCH] } : { text: call.index === 1 ? "At six." : "Six." },
    );
    const question: RunMessage = { role: "user", text: "When is high water?" };

    const first = await collect(
      create(),
      request(model, { messages: [question], tools: [search] }),
    );
    const history: RunMessage[] = [
      question,
      ...(first.finished ?? []),
      { role: "user", text: "In one word?" },
    ];
    const second = await collect(create(), request(model, { messages: history, tools: [search] }));

    expect(rolesSent(model, 2)).toEqual([
      "system",
      "user",
      "assistant",
      "tool",
      "assistant",
      "user",
    ]);
    expect(second.text).toBe("Six.");
  });

  test("compaction: each later request sends the history the window gives back, compacted afresh; the Run's own history stays whole", async () => {
    const read = testTool("read_file", { result: "LONG ".repeat(200) });
    const seen: number[] = [];
    const window: RunWindow = {
      fitResult: (text) => text,
      canCallTools: () => true,
      stepFinished: () => undefined,
      compact: (history) => {
        seen.push(history.length);
        return history.map((message) =>
          message.role === "tool" ? { ...message, result: "[elided]" } : message,
        );
      },
    };
    const model = stepsModel(
      [
        [{ tool: "read_file", input: { query: "a" } }],
        [{ tool: "read_file", input: { query: "b" } }],
      ],
      "Done.",
    );

    const run = await collect(create(), request(model, { tools: [read], window }));

    expect(seen).toEqual([3, 5]);
    expect(resultsSent(model, 2).map((each) => each.text)).toEqual(["[elided]", "[elided]"]);
    expect(
      run.finished?.flatMap((message) => (message.role === "tool" ? [message.result.length] : [])),
    ).toEqual([1_000, 1_000]);
  });

  test("a call's arguments are checked against its Tool's schema: one that doesn't match never reaches the gate or the Tool, and the model reads why", async () => {
    const search = testTool("search_documents");
    const gate = scriptedGate();
    const model = stepsModel(
      [[{ tool: "search_documents", input: { terms: "high water" } }]],
      "Sorry.",
    );

    const run = await collect(create(), request(model, { tools: [search], gate: gate.gate }));

    expect(gate.asked).toEqual([]);
    expect(search.calls).toEqual([]);
    expect(run.events).toContainEqual({ type: "tool-result", id: "call-0-0", ok: false });
    expect(resultsSent(model, 1)).toEqual([
      {
        tool: "search_documents",
        type: "error-text",
        text: expect.stringMatching(/must have required property 'query'/),
      },
    ]);
  });

  test("leniently: numbers and true/false sent as text, and lists sent as JSON text, reach the Tool converted", async () => {
    const add = testTool("add_rows", {
      schema: {
        type: "object",
        properties: {
          count: { type: "integer" },
          dry: { type: "boolean" },
          rows: { type: "array", items: { type: "object", properties: { n: { type: "number" } } } },
        },
        required: ["count", "rows"],
      },
    });
    const model = stepsModel(
      [[{ tool: "add_rows", input: { count: "2", dry: "false", rows: '[{"n": "1"}, {"n": 2}]' } }]],
      "Added.",
    );

    await collect(create(), request(model, { tools: [add] }));

    expect(add.calls.map((call) => call.input)).toEqual([
      { count: 2, dry: false, rows: [{ n: 1 }, { n: 2 }] },
    ]);
  });
});

test("the Run engine's interface imports no agent library, so the loop's library can change without what is above it", async () => {
  const source = await readFile(resolve(__dirname, "../../src/core/runs/engine.ts"), "utf8");
  const imports = [...source.matchAll(/^import[\s\S]*?from\s+"([^"]+)";/gm)].map(
    (match) => match[1] ?? "",
  );
  expect(imports).toEqual(["../api", "../providers/models", "../tools"]);
  expect(source).toMatch(/^import type \{ ChatLanguageModel \}/m);
});
