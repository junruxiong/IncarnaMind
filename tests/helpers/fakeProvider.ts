import { vi } from "vitest";
import { type ShownPassage, shownPassages } from "./citations";

/**
 * A stand-in for a model provider's API, in place of `fetch`: it answers the
 * model list and the chat endpoint (Chat Completions, or the Responses API for
 * xAI), and records every request. The real AI SDK providers talk to it, so
 * what the tests see is what the provider's package puts on the wire.
 *
 * The model it plays searches the Documents once, cites a Passage with a
 * quote, and writes its Answer: the same plan as `citingModel`.
 */

/** A request body or provider reply as the fake reads it: JSON that is not checked for shape. */
// biome-ignore lint/suspicious/noExplicitAny: a fake reads JSON of several providers' shapes
export type Loose = Record<string, any>;

export interface RecordedRequest {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: Loose;
}

export interface FakeProviderOptions {
  /** "chat": Chat Completions (`/chat/completions`); "responses": the Responses API (`/responses`). */
  protocol: "chat" | "responses";
  /** Mistral puts the usage on the last chunk; the others send it in a chunk of its own, with no choices. */
  usageOnLastChunk?: boolean;
  /** The body of the model list endpoint (`GET …/models`). */
  models: unknown;
  /** What the model searches for, and what it cites and says. */
  query: string;
  records(passages: ShownPassage[]): unknown[];
  answer: string;
}

/** What the model has been told so far: its Tool results, and the Tools it may call. */
interface Turn {
  tools: string[];
  results: { tool: string; text: string }[];
}

type Reply = { text: string } | { call: { name: string; input: unknown } };

function chatTurn(body: Loose): Turn {
  const names = new Map<string, string>();
  for (const message of body.messages ?? []) {
    for (const call of message.tool_calls ?? []) names.set(call.id, call.function.name);
  }
  return {
    tools: (body.tools ?? []).map((tool: Loose) => tool.function.name),
    results: (body.messages ?? [])
      .filter((message: Loose) => message.role === "tool")
      .map((message: Loose) => ({
        tool: names.get(message.tool_call_id) ?? "",
        text:
          typeof message.content === "string" ? message.content : JSON.stringify(message.content),
      })),
  };
}

function responsesTurn(body: Loose): Turn {
  const names = new Map<string, string>();
  const input: Loose[] = Array.isArray(body.input) ? body.input : [];
  for (const item of input) if (item.type === "function_call") names.set(item.call_id, item.name);
  return {
    tools: (body.tools ?? []).map((tool: Loose) => tool.name),
    results: input
      .filter((item) => item.type === "function_call_output")
      .map((item) => ({
        tool: names.get(item.call_id) ?? "",
        text: typeof item.output === "string" ? item.output : JSON.stringify(item.output),
      })),
  };
}

function next(turn: Turn, options: FakeProviderOptions): Reply {
  if (!turn.tools.includes("search_documents")) return { text: "OK" };
  const searches = turn.results.filter((result) => result.tool === "search_documents");
  if (searches.length === 0) {
    return { call: { name: "search_documents", input: { query: options.query } } };
  }
  if (!turn.results.some((result) => result.tool === "cite")) {
    const passages = shownPassages(searches.at(-1)?.text ?? "");
    return { call: { name: "cite", input: { citations: options.records(passages) } } };
  }
  return { text: options.answer };
}

const sse = (events: object[]) =>
  `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`;

function chatResponse(
  model: string,
  reply: Reply,
  stream: boolean,
  usageOnLastChunk: boolean,
): Response {
  const usage = { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 };
  const finish = "text" in reply ? "stop" : "tool_calls";
  const toolCall =
    "call" in reply
      ? {
          id: `call_${reply.call.name}`,
          type: "function",
          function: { name: reply.call.name, arguments: JSON.stringify(reply.call.input) },
        }
      : null;
  if (!stream) {
    return Response.json({
      id: "chat_1",
      object: "chat.completion",
      created: 1_790_000_000,
      model,
      choices: [
        {
          index: 0,
          finish_reason: finish,
          message: {
            role: "assistant",
            content: "text" in reply ? reply.text : null,
            ...(toolCall && { tool_calls: [toolCall] }),
          },
        },
      ],
      usage,
    });
  }
  const chunk = (delta: object, finishReason: string | null) => ({
    id: "chat_1",
    object: "chat.completion.chunk",
    created: 1_790_000_000,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  const events: object[] = [chunk({ role: "assistant", content: "" }, null)];
  if ("text" in reply) events.push(chunk({ content: reply.text }, null));
  if (toolCall) events.push(chunk({ tool_calls: [{ index: 0, ...toolCall }] }, null));
  events.push(usageOnLastChunk ? { ...chunk({}, finish), usage } : chunk({}, finish));
  if (!usageOnLastChunk) events.push({ ...chunk({}, null), choices: [], usage });
  return new Response(sse(events), { headers: { "content-type": "text/event-stream" } });
}

function responsesResponse(model: string, reply: Reply, stream: boolean): Response {
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
          call_id: `call_${reply.call.name}`,
          name: reply.call.name,
          arguments: JSON.stringify(reply.call.input),
          status: "completed",
        };
  const response = {
    id: "resp_1",
    object: "response",
    created_at: 1_790_000_000,
    model,
    status: "completed",
    output: [output],
    usage: {
      input_tokens: 12,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: 5,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 17,
    },
  };
  if (!stream) return Response.json(response);
  const events: object[] = [
    { type: "response.created", response: { ...response, status: "in_progress", output: [] } },
    {
      type: "response.output_item.added",
      output_index: 0,
      item: "text" in reply ? { ...output, status: "in_progress", content: [] } : output,
    },
  ];
  if ("text" in reply) {
    events.push(
      {
        type: "response.content_part.added",
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      },
      {
        type: "response.output_text.delta",
        item_id: "msg_1",
        output_index: 0,
        content_index: 0,
        delta: reply.text,
      },
    );
  }
  events.push(
    { type: "response.output_item.done", output_index: 0, item: output },
    { type: "response.completed", response },
  );
  return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
    headers: { "content-type": "text/event-stream" },
  });
}

const headersOf = (init?: RequestInit): Record<string, string> =>
  Object.fromEntries(
    [...new Headers(init?.headers ?? {}).entries()].map(([name, value]) => [
      name.toLowerCase(),
      value,
    ]),
  );

/**
 * Replaces `fetch` with the fake provider; returns the requests it gets. A
 * request to anywhere but `host` is a failure.
 */
export function fakeProvider(host: string, options: FakeProviderOptions): RecordedRequest[] {
  const requests: RecordedRequest[] = [];
  const answer = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(String(input instanceof Request ? input.url : input));
    if (url.host !== host) throw new Error(`Unexpected request to ${url.href}`);
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? (JSON.parse(String(init.body)) as Loose) : {};
    requests.push({ method, url: url.href, headers: headersOf(init), body });
    if (method === "GET" && url.pathname.endsWith("/models")) return Response.json(options.models);
    const stream = body.stream === true;
    const model = String(body.model);
    if (options.protocol === "responses") {
      return responsesResponse(model, next(responsesTurn(body), options), stream);
    }
    return chatResponse(model, next(chatTurn(body), options), stream, !!options.usageOnLastChunk);
  };
  vi.stubGlobal("fetch", vi.fn(answer));
  return requests;
}
