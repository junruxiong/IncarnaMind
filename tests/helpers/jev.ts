import { createServer, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { onTestFinished } from "vitest";

/** The key the fake accepts unless told otherwise. */
export const JEV_KEY = "jev-test-key";

export interface JevQuestion {
  type: string;
  instructions: string;
  criteria?: { true?: string; false?: string };
}

export interface JevRequestBody {
  state: unknown;
  model: string;
  images?: string[];
  questions: Record<string, JevQuestion>;
}

/** One request the fake received. */
export interface JevRequest {
  method: string;
  path: string;
  authorization: string | undefined;
  body: JevRequestBody;
}

export interface FakeJev {
  chooseGroup(id: string): void;
  readonly unloadedModels: string[];
  /** The base URL, e.g. "http://127.0.0.1:51234". */
  readonly url: string;
  /** Every request, in order. */
  readonly requests: JevRequest[];
  /** Answers each Tag's question with its probability, by Tag name; any other Tag gets 0.02. */
  answer(probabilities: Readonly<Record<string, number>>): void;
  /** Answers every request with this HTTP error from now on (null: answer again). */
  fail(failure: { status: number; message: string; headers?: Record<string, string> } | null): void;
  /** Answers with a body that isn't Jev's (null: answer properly again). */
  garble(body: unknown): void;
  /** Holds every answer until `release()`. */
  hold(): void;
  release(): void;
  /** Resolves once the fake has received `count` requests. */
  requested(count?: number): Promise<void>;
}

/** The Tag name a tagging question asks about: the text between “ and ”. */
const tagNameOf = (question: JevQuestion) => /“(.+)”/.exec(question.instructions)?.[1] ?? "";

/**
 * A local stand-in for TypeSafe's `POST /v1/systemone`, answering Noul
 * questions as https://docs.typesafe.ai/api documents them. It checks the
 * bearer key, and is stopped when the current test finishes.
 */
export async function startFakeJev({
  apiKey = JEV_KEY,
  ollamaModels = [] as string[],
} = {}): Promise<FakeJev> {
  const requests: JevRequest[] = [];
  const unloadedModels: string[] = [];
  const loadedModels = new Set<string>();
  let probabilities: Readonly<Record<string, number>> = {};
  let failure: { status: number; message: string; headers?: Record<string, string> } | null = null;
  let garbled: unknown = null;
  let group = "__unsorted__";
  let gate: Promise<void> | null = null;
  let open = () => {};
  const waiting: { count: number; resolve: () => void }[] = [];

  const send = (
    response: ServerResponse,
    status: number,
    payload: unknown,
    headers: Record<string, string> = {},
  ) => {
    response.writeHead(status, { "content-type": "application/json", ...headers });
    response.end(JSON.stringify(payload));
  };

  const server = createServer(async (request, response) => {
    let text = "";
    for await (const chunk of request) text += chunk;
    const body = (text ? JSON.parse(text) : {}) as JevRequestBody;
    if (request.method === "GET" && request.url === "/api/tags") {
      send(response, 200, { models: ollamaModels.map((name) => ({ name })) });
      return;
    }
    if (request.method === "GET" && request.url === "/api/ps") {
      send(response, 200, { models: [...loadedModels].map((name) => ({ name })) });
      return;
    }
    if (request.method === "POST" && request.url === "/api/generate") {
      unloadedModels.push(body.model);
      loadedModels.delete(body.model);
      send(response, 200, { done: true });
      return;
    }
    requests.push({
      method: request.method ?? "",
      path: request.url ?? "",
      authorization: request.headers.authorization,
      body,
    });
    for (const wait of waiting.filter((each) => each.count <= requests.length)) wait.resolve();
    if (gate) await gate;

    if (request.method !== "POST" || request.url !== "/v1/systemone") {
      send(response, 404, { detail: "Not Found" });
    } else if (request.headers.authorization !== `Bearer ${apiKey}`) {
      send(response, 401, { detail: "Invalid API key." });
    } else if (failure) {
      send(response, failure.status, { detail: failure.message }, failure.headers);
    } else if (garbled !== null) {
      loadedModels.add(body.model);
      send(response, 200, garbled);
    } else {
      const answers = Object.fromEntries(
        Object.entries(body.questions ?? {}).map(([id, question]) => [
          id,
          question.type === "choice"
            ? { type: "choice", choice: group }
            : { type: "noul", noul: probabilities[tagNameOf(question)] ?? 0.02 },
        ]),
      );
      loadedModels.add(body.model);
      send(response, 200, {
        model: "jev-1.13.0",
        answers,
        usage: { input_tokens: 300, output_tokens: 20 },
      });
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  onTestFinished(
    () =>
      new Promise<void>((resolve) => {
        open();
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  );
  const { port } = server.address() as AddressInfo;

  return {
    chooseGroup(id) {
      group = id;
    },
    url: `http://127.0.0.1:${port}`,
    requests,
    unloadedModels,
    answer(next) {
      probabilities = next;
    },
    fail(next) {
      failure = next;
    },
    garble(body) {
      garbled = body;
    },
    hold() {
      gate = new Promise((resolve) => {
        open = resolve;
      });
    },
    release() {
      gate = null;
      open();
    },
    requested: (count = 1) =>
      requests.length >= count
        ? Promise.resolve()
        : new Promise((resolve) => waiting.push({ count, resolve })),
  };
}
