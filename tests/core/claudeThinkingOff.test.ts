import { generateText, streamText } from "ai";
import { afterEach, describe, expect, test, vi } from "vitest";
import { createAiSdkChatModel } from "../../src/core/providers/models";

/*
 * A small call asks a Claude model not to think in the form that model takes (#907): the
 * newest refuse `thinking: disabled`, and their refusal names what they take instead.
 */

type Body = { thinking?: { type?: string }; output_config?: { effort?: string }; stream?: boolean };

const SONNET_REFUSAL =
  'To turn thinking off on this model, send "thinking": {"type": "between_tools"} instead of {"type": "disabled"}.';
const OPUS_REFUSAL =
  '"thinking.type.disabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort" to control thinking behavior.';

const message = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  model: "claude",
  content: [{ type: "text", text: "OK" }],
  stop_reason: "end_turn",
  stop_sequence: null,
  usage: { input_tokens: 5, output_tokens: 1 },
};

const events = [
  {
    type: "message_start",
    message: { ...message, content: [], stop_reason: null },
  },
  { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "OK" } },
  { type: "content_block_stop", index: 0 },
  { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
  { type: "message_stop" },
];

/** Anthropic's Messages API, refusing a request whose `thinking` is `refusal.type` with its message. */
function anthropic(refusal?: { type: string; message: string }): Body[] {
  const bodies: Body[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as Body;
      bodies.push(body);
      if (refusal && body.thinking?.type === refusal.type) {
        return Response.json(
          { type: "error", error: { type: "invalid_request_error", message: refusal.message } },
          { status: 400 },
        );
      }
      if (!body.stream) return Response.json(message);
      const sse = events
        .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
        .join("");
      return new Response(sse, { headers: { "content-type": "text/event-stream" } });
    }),
  );
  return bodies;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

const claude = (modelId: string, thinking?: "off") =>
  createAiSdkChatModel({
    kind: "anthropic",
    baseUrl: null,
    apiKey: "key",
    modelId,
    ...(thinking && { thinking }),
  });

const ask = (modelId: string, thinking?: "off") =>
  generateText({ model: claude(modelId, thinking), prompt: "Hi", maxRetries: 0 });

describe("A small call to a Claude model", () => {
  test.each([
    ["claude-haiku-5-5", { type: "disabled" }, undefined],
    ["claude-opus-5", { type: "disabled" }, undefined],
    ["claude-sonnet-5-5", { type: "between_tools" }, undefined],
    ["claude-opus-5-5", { type: "adaptive" }, { effort: "low" }],
    ["claude-fable-5-1", { type: "adaptive" }, { effort: "low" }],
    ["claude-fable-5", { type: "adaptive" }, { effort: "low" }],
  ])("%s is asked not to think in the form it takes", async (modelId, thinking, outputConfig) => {
    const bodies = anthropic();
    const reply = await ask(modelId, "off");
    expect(bodies).toHaveLength(1);
    expect(bodies[0]?.thinking).toEqual(thinking);
    expect(bodies[0]?.output_config).toEqual(outputConfig);
    // Sent as the model takes it, nothing is swapped on the way.
    expect(reply.warnings ?? []).toEqual([]);
  });

  test("a call that writes the Answer isn't asked not to think", async () => {
    const bodies = anthropic();
    await ask("claude-opus-5-5");
    expect(bodies[0]?.output_config?.effort).toBeUndefined();
    expect(bodies[0]?.thinking?.type).not.toBe("disabled");
  });

  test("a model the catalog doesn't know is asked once more in the form its refusal names, and its later calls start with it", async () => {
    const bodies = anthropic({ type: "disabled", message: SONNET_REFUSAL });
    expect((await ask("claude-next-6", "off")).text).toBe("OK");
    expect(bodies.map((body) => body.thinking?.type)).toEqual(["disabled", "between_tools"]);

    bodies.length = 0;
    await ask("claude-next-6", "off");
    expect(bodies.map((body) => body.thinking?.type)).toEqual(["between_tools"]);
  });

  test("streamed, one that always thinks is asked again at the lowest effort", async () => {
    const bodies = anthropic({ type: "disabled", message: OPUS_REFUSAL });
    const result = streamText({
      model: claude("claude-next-7", "off"),
      prompt: "Hi",
      maxRetries: 0,
    });
    expect(await result.text).toBe("OK");
    expect(bodies.map((body) => [body.thinking?.type, body.output_config?.effort])).toEqual([
      ["disabled", undefined],
      ["adaptive", "low"],
    ]);
  });

  test("a refusal that names no other form is the call's failure, asked once", async () => {
    const bodies = anthropic({ type: "disabled", message: "max_tokens: too large" });
    await expect(ask("claude-next-8", "off")).rejects.toThrow("max_tokens");
    expect(bodies).toHaveLength(1);
  });
});
