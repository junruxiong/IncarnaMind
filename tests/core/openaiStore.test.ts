/**
 * Requests to OpenAI's Responses API ask OpenAI not to store them (#159): every
 * request the app makes with an API-key OpenAI model carries `store: false`,
 * whatever it is for. The tests run the real OpenAI provider against a stand-in
 * for `fetch` that records each request body.
 */
import { afterEach, describe, expect, test, vi } from "vitest";
import type {
  AnswerEngineEvent,
  AnswerRequest,
  AnswerTools,
  Tool,
} from "../../src/core/answers/engine";
import { createAiSdkAnswerEngine } from "../../src/core/answers/engine";
import { chatGroupClassifier } from "../../src/core/library/classifier";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import { chatClassifier } from "../../src/core/tags/classify";
import { createTempDataFolder, startCore } from "../helpers/core";

type Reply = { text: string } | { call: { name: string; input: unknown } };

interface Captured {
  url: string;
  body: Record<string, unknown>;
}

/** Stands in for OpenAI: records every request and answers from `replies`, in turn (the last again). */
function fakeOpenAI(replies: Reply[]) {
  const requests: Captured[] = [];
  const respond = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input instanceof Request ? input.url : input);
    const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    requests.push({ url, body });
    const reply = replies[Math.min(requests.length - 1, replies.length - 1)] as Reply;
    const output =
      "text" in reply
        ? {
            id: "msg_1",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: reply.text, annotations: [] }],
          }
        : {
            id: "fc_1",
            type: "function_call",
            call_id: "call_1",
            name: reply.call.name,
            arguments: JSON.stringify(reply.call.input),
            status: "completed",
          };
    const response = {
      id: "resp_1",
      object: "response",
      created_at: 1_790_000_000,
      model: String(body.model),
      status: "completed",
      output: [output],
      usage: {
        input_tokens: 10,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens: 4,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 14,
      },
    };
    if (body.stream !== true) {
      return new Response(JSON.stringify(response), {
        headers: { "content-type": "application/json" },
      });
    }
    const events: object[] = [
      { type: "response.created", response: { ...response, status: "in_progress", output: [] } },
      {
        type: "response.output_item.added",
        output_index: 0,
        item: "text" in reply ? { ...output, status: "in_progress", content: [] } : output,
      },
    ];
    if ("text" in reply) {
      events.push({
        type: "response.content_part.added",
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      });
      events.push({
        type: "response.output_text.delta",
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        delta: reply.text,
      });
    }
    events.push(
      { type: "response.output_item.done", output_index: 0, item: output },
      { type: "response.completed", response },
    );
    return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
      headers: { "content-type": "text/event-stream" },
    });
  };
  vi.stubGlobal("fetch", vi.fn(respond));
  return requests;
}

afterEach(() => vi.unstubAllGlobals());

const openai = (modelId = "gpt-5.5") =>
  createAiSdkChatModel({ kind: "openai", baseUrl: null, apiKey: "sk-test", modelId });

/** Every request went to the Responses API and said not to store. */
function expectNoneStored(requests: Captured[], count?: number) {
  if (count !== undefined) expect(requests).toHaveLength(count);
  expect(requests.length).toBeGreaterThan(0);
  for (const { url, body } of requests) {
    expect(url).toBe("https://api.openai.com/v1/responses");
    expect(body.store).toBe(false);
    // Reasoning summaries stay off: nothing here asks for them.
    expect(JSON.stringify(body.reasoning ?? {})).not.toContain("summary");
  }
}

const documents = (searches: string[] = []): AnswerTools => ({
  documentCount: 1,
  documentLanguages: [{ language: "Chinese", documents: 1 }],
  async searchDocuments(query) {
    searches.push(query);
    return { text: "No Passages found.", passageCount: 0 };
  },
  cite: () => "Recorded.",
});

function request(overrides: Partial<AnswerRequest> = {}): AnswerRequest {
  return {
    instructions: () => "Answer from the Documents.",
    messages: [{ role: "user", content: "What is a Mind?" }],
    question: "What is a Mind?",
    model: openai(),
    documents: documents(),
    tools: [] as readonly Tool[],
    signal: new AbortController().signal,
    ...overrides,
  };
}

async function run(answer: AnswerRequest): Promise<AnswerEngineEvent[]> {
  const events: AnswerEngineEvent[] = [];
  for await (const event of createAiSdkAnswerEngine().generate(answer)) events.push(event);
  return events;
}

describe("Requests to OpenAI say not to store", () => {
  test("an Answer with the Tool loop: every step of the loop", async () => {
    const requests = fakeOpenAI([
      { call: { name: "search_documents", input: { query: "Mind" } } },
      { text: "A Mind is a note." },
    ]);

    const events = await run(request());

    expect(events.at(-1)).toEqual({ type: "finished" });
    expect(events.filter((event) => event.type === "text-delta")).not.toHaveLength(0);
    expectNoneStored(requests, 2);
    // The second step holds the first's Tool call: it was sent in full, not by reference.
    expect(JSON.stringify(requests[1]?.body.input)).toContain("function_call");
  });

  test("an Answer by structured output, with its query rewritten and translated", async () => {
    const requests = fakeOpenAI([
      { text: "what is a Mind" },
      { text: "什么是 Mind" },
      { text: JSON.stringify({ answer: "A Mind is a note.", citations: [] }) },
    ]);

    const events = await run(
      request({
        support: "structured-output",
        messages: [
          { role: "user", content: "Tell me about tides." },
          { role: "assistant", content: "Tides are caused by the Moon." },
          { role: "user", content: "What is a Mind?" },
        ],
      }),
    );

    expect(events.at(-1)).toEqual({ type: "finished" });
    // The rewrite, the translation and the Answer itself.
    expectNoneStored(requests, 3);
    expect(requests[2]?.body.text).toMatchObject({ format: { type: "json_schema" } });
  });

  test("a plain Answer, with no Documents to search", async () => {
    const requests = fakeOpenAI([{ text: "A Mind is a note." }]);

    const events = await run(
      request({ documents: { ...documents(), documentCount: 0 }, support: "none" }),
    );

    expect(events.at(-1)).toEqual({ type: "finished" });
    expectNoneStored(requests, 1);
  });

  test("automatic tagging and Organize", async () => {
    const requests = fakeOpenAI([
      { text: JSON.stringify({ tags: ["Report"] }) },
      { text: JSON.stringify({ groupId: "g-reports", tags: ["t-report"] }) },
    ]);
    const tags = [{ id: "t-report", name: "Report", description: "A report of findings" }];
    const excerpt = { name: "QBR", kind: "pdf" as const, pageCount: 2, text: "Quarterly review." };
    const signal = new AbortController().signal;

    const tagged = await chatClassifier(openai()).decide({ tags, excerpt, signal });
    const organized = await chatGroupClassifier(openai(), false).organize(
      [{ id: "g-reports", name: "Reports", description: "Reports", createdAt: "", updatedAt: "" }],
      tags,
      excerpt,
      signal,
    );

    expect(tagged.map((tag) => tag.tagId)).toEqual(["t-report"]);
    expect(organized.groupId).toBe("g-reports");
    expectNoneStored(requests, 2);
  });

  test("the connection test", async () => {
    const requests = fakeOpenAI([{ text: "OK" }]);
    const core = startCore(await createTempDataFolder(), { createChatModel: createAiSdkChatModel });
    core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));

    const result = await core.testChatConnection({
      kind: "openai",
      apiKey: "sk-test",
      modelId: "gpt-5.5",
    });

    expect(result).toEqual({ ok: true });
    expectNoneStored(requests, 1);
  });

  test("an option set by a caller is kept, and only store is forced", async () => {
    const requests = fakeOpenAI([{ text: "OK" }]);
    const { generateText } = await import("ai");

    await generateText({
      model: openai(),
      prompt: "Hi",
      providerOptions: { openai: { store: true, user: "someone" } },
      maxRetries: 0,
    });

    expectNoneStored(requests, 1);
    expect(requests[0]?.body.user).toBe("someone");
  });

  test("other providers are left alone", async () => {
    const requests = fakeOpenAI([{ text: "OK" }]);
    const { generateText } = await import("ai");
    const compatible = createAiSdkChatModel({
      kind: "openai-compatible",
      baseUrl: "http://127.0.0.1:9/v1",
      apiKey: null,
      modelId: "local",
    });

    await generateText({ model: compatible, prompt: "Hi", maxRetries: 0 }).catch(() => undefined);

    expect(requests.every(({ body }) => !("store" in body))).toBe(true);
  });
});
