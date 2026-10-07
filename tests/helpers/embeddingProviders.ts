import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { APICallError } from "ai";
import { MockEmbeddingModelV4, MockRerankingModelV4 } from "ai/test";
import { onTestFinished } from "vitest";
import type {
  EmbeddingModelFactory,
  EmbeddingModelSpec,
  RerankingModelFactory,
  RerankingModelSpec,
} from "../../src/core";
import { fakeVector } from "../../src/core/embedding/fake";

/** An HTTP failure a fake provider answers with; no status: no connection. */
export interface ProviderFailure {
  status?: number;
  message: string;
}

const failureOf = ({ status, message }: ProviderFailure, url: string) =>
  new APICallError({
    message,
    url,
    requestBodyValues: {},
    statusCode: status,
    cause: status === undefined ? new TypeError("fetch failed") : undefined,
    isRetryable: false,
  });

/** One request to a mock embedding model. */
export interface EmbeddingCall {
  spec: EmbeddingModelSpec;
  values: string[];
  providerOptions: unknown;
}

/**
 * Passages are sent as their Document's name, a line break, then their text;
 * search queries have no line break.
 */
export const isPassageCall = (call: Pick<EmbeddingCall, "values">) =>
  call.values.some((value) => value.includes("\n"));

export interface MockEmbeddings {
  createEmbeddingModel: EmbeddingModelFactory;
  /** Every request, in order. */
  readonly calls: EmbeddingCall[];
  /** The texts of every Passage sent, in order. */
  passages(): string[];
  /** The search queries sent, in order. */
  queries(): string[];
  /** Answers every request with this failure from now on (null: answer again). */
  fail(failure: ProviderFailure | null): void;
  /** Holds each request for Passages until `release` lets it through. Queries aren't held. */
  hold(): void;
  /** Lets `count` held requests through (all of them, and stops holding, without a count). */
  release(count?: number): void;
  /** Resolves once `count` requests for Passages have arrived (held or not). */
  passageRequests(count: number): Promise<void>;
}

/**
 * AI SDK mock embedding models. Each text's vector is the fake embedder's (it
 * counts words), of `dimensions` numbers, turned by the model's name: a model
 * whose name contains "mirror" returns them backwards, a different vector
 * space of the same size, so comparing its vectors with another model's gives
 * nonsense. Every request is recorded.
 */
export function mockEmbeddingModels({ dimensions = 64 } = {}): MockEmbeddings {
  const calls: EmbeddingCall[] = [];
  let failure: ProviderFailure | null = null;
  let holding = false;
  const held: (() => void)[] = [];
  const waiting: { count: number; resolve: () => void }[] = [];
  const passageCount = () => calls.filter(isPassageCall).length;

  const createEmbeddingModel: EmbeddingModelFactory = (spec) =>
    new MockEmbeddingModelV4({
      provider: spec.kind,
      modelId: spec.modelId,
      maxEmbeddingsPerCall: 100,
      doEmbed: async ({ values, providerOptions }) => {
        const call = { spec, values: [...values], providerOptions };
        calls.push(call);
        for (const wait of waiting.filter((each) => each.count <= passageCount())) wait.resolve();
        if (holding && isPassageCall(call))
          await new Promise<void>((resolve) => held.push(resolve));
        if (failure) throw failureOf(failure, "https://api.example.com/v1/embeddings");
        const embeddings = values.map((value) => {
          const vector = Array.from(fakeVector(value, dimensions));
          return spec.modelId.includes("mirror") ? vector.reverse() : vector;
        });
        return { embeddings, warnings: [] };
      },
    });

  return {
    createEmbeddingModel,
    calls,
    passages: () => calls.filter(isPassageCall).flatMap((call) => call.values),
    queries: () => calls.filter((call) => !isPassageCall(call)).flatMap((call) => call.values),
    fail(next) {
      failure = next;
    },
    hold() {
      holding = true;
    },
    release(count) {
      if (count === undefined) holding = false;
      for (const resolve of held.splice(0, count ?? held.length)) resolve();
    },
    passageRequests: (count) =>
      passageCount() >= count
        ? Promise.resolve()
        : new Promise((resolve) => waiting.push({ count, resolve })),
  };
}

/** One request to a mock reranking model. */
export interface RerankCall {
  spec: RerankingModelSpec;
  query: string;
  documents: string[];
}

export interface MockRerankers {
  createRerankingModel: RerankingModelFactory;
  readonly calls: RerankCall[];
  fail(failure: ProviderFailure | null): void;
}

/**
 * AI SDK mock reranking models: each document's relevance is `score(text)`,
 * best first. Every request is recorded.
 */
export function mockRerankingModels(score: (document: string) => number): MockRerankers {
  const calls: RerankCall[] = [];
  let failure: ProviderFailure | null = null;
  return {
    calls,
    fail(next) {
      failure = next;
    },
    createRerankingModel: (spec) =>
      new MockRerankingModelV4({
        provider: spec.kind,
        modelId: spec.modelId,
        doRerank: async ({ documents, query, topN }) => {
          const texts = documents.type === "text" ? documents.values : [];
          calls.push({ spec, query, documents: [...texts] });
          if (failure) throw failureOf(failure, "https://api.cohere.com/v2/rerank");
          const ranking = texts
            .map((text, index) => ({ index, relevanceScore: score(text) }))
            .sort((a, b) => b.relevanceScore - a.relevanceScore)
            .slice(0, topN ?? texts.length);
          return { ranking };
        },
      }),
  };
}

/** One request a fake OpenAI-compatible embeddings server received. */
export interface EmbeddingsRequest {
  path: string;
  authorization: string | undefined;
  body: { model?: string; input?: string[] | string };
}

/**
 * A local server that speaks the OpenAI embeddings API, `POST /v1/embeddings`,
 * as Ollama and other OpenAI-compatible servers do. It knows the models in
 * `models` (others get Ollama's 404), and checks the bearer key if given one.
 * Stopped when the test finishes.
 */
export async function startEmbeddingsServer({
  models = ["bge-m3"],
  dimensions = 48,
  apiKey,
}: {
  models?: string[];
  dimensions?: number;
  apiKey?: string;
} = {}): Promise<{ baseUrl: string; requests: EmbeddingsRequest[] }> {
  const requests: EmbeddingsRequest[] = [];
  const server = createServer(async (request, response) => {
    let text = "";
    for await (const chunk of request) text += chunk;
    const body = (text ? JSON.parse(text) : {}) as EmbeddingsRequest["body"];
    requests.push({ path: request.url ?? "", authorization: request.headers.authorization, body });
    const send = (status: number, payload: unknown) => {
      response.writeHead(status, { "content-type": "application/json" });
      response.end(JSON.stringify(payload));
    };
    if (request.method !== "POST" || request.url !== "/v1/embeddings") {
      send(404, { error: { message: "404 page not found" } });
    } else if (apiKey && request.headers.authorization !== `Bearer ${apiKey}`) {
      send(401, { error: { message: "Incorrect API key provided." } });
    } else if (!models.includes(body.model ?? "")) {
      send(404, {
        error: { message: `model "${body.model}" not found, try pulling it first` },
      });
    } else {
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      send(200, {
        object: "list",
        data: inputs.map((input, index) => ({
          object: "embedding",
          index,
          embedding: Array.from(fakeVector(input, dimensions)),
        })),
        model: body.model,
        usage: { prompt_tokens: 10, total_tokens: 10 },
      });
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  onTestFinished(
    () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  );
  const { port } = server.address() as AddressInfo;
  return { baseUrl: `http://127.0.0.1:${port}`, requests };
}
