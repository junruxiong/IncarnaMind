/**
 * The built-in reranking model run somewhere else, talking in messages: both
 * ends of the protocol, as for the embedding model (see ../embedding/channel).
 * The desktop app runs it in an Electron utility process of its own
 * (src/main/reranker.ts and src/main/rerankerProcess.ts); the evaluation
 * (eval/) runs it on a Node worker thread.
 *
 * Requests are { id, type: "load", files } or { id, type: "score", query,
 * texts }; the other end answers { id, scores } (or { id } for a load) or
 * { id, error }, taking one request at a time, in order.
 */
import type { CrossEncoder, RerankingModelFiles } from "../adapters";

export type CrossEncoderCommand =
  | { type: "load"; files: RerankingModelFiles }
  | { type: "score"; query: string; texts: string[] };

/** `id` pairs a response with its request. */
export type CrossEncoderRequest = CrossEncoderCommand & { id: number };

/** Scores for "score", nothing more for "load", or what went wrong. */
export interface CrossEncoderResponse {
  id: number;
  scores?: number[];
  error?: string;
}

/** A running process or thread that serves the model, as the side that sends requests sees it. */
export interface CrossEncoderChannel {
  send(request: CrossEncoderRequest): void;
  onResponse(listener: (response: CrossEncoderResponse) => void): void;
  /** It stopped: crashed, ran out of memory, or was stopped. */
  onExit(listener: (code: number | null) => void): void;
  stop(): void;
}

type Pending = { resolve(response: CrossEncoderResponse): void; reject(error: Error): void };

/**
 * The core's `CrossEncoder`, backed by a channel that `start` opens on the
 * first `load`. If it stops, pending requests fail, and the next `load` opens
 * a new one.
 */
export function createChannelCrossEncoder(start: () => CrossEncoderChannel): CrossEncoder {
  let channel: CrossEncoderChannel | undefined;
  let loadedKey: string | undefined;
  let loading: { key: string; done: Promise<void> } | undefined;
  const pending = new Map<number, Pending>();
  let nextId = 1;

  function open(): CrossEncoderChannel {
    const started = start();
    started.onResponse((response) => {
      const waiting = pending.get(response.id);
      pending.delete(response.id);
      waiting?.resolve(response);
    });
    started.onExit((code) => {
      if (channel !== started) return;
      channel = undefined;
      loadedKey = undefined;
      const error = new Error(`The reranking process stopped (exit code ${code}).`);
      for (const waiting of pending.values()) waiting.reject(error);
      pending.clear();
    });
    return started;
  }

  function request(command: CrossEncoderCommand): Promise<CrossEncoderResponse> {
    channel ??= open();
    const target = channel;
    const id = nextId++;
    return new Promise<CrossEncoderResponse>((resolve, reject) => {
      pending.set(id, { resolve, reject });
      target.send({ ...command, id });
    }).then((response) => {
      if (response.error !== undefined) throw new Error(response.error);
      return response;
    });
  }

  return {
    async load(files) {
      const key = JSON.stringify(files);
      if (channel && loadedKey === key) return;
      if (loading?.key !== key) {
        const done = request({ type: "load", files })
          .then(() => {
            loadedKey = key;
          })
          .finally(() => {
            if (loading?.done === done) loading = undefined;
          });
        loading = { key, done };
      }
      await loading.done;
    },

    async score(query, texts) {
      if (!channel || loadedKey === undefined) {
        throw new Error("The reranking model isn't loaded.");
      }
      const { scores } = await request({ type: "score", query, texts: [...texts] });
      if (!scores || scores.length !== texts.length) {
        throw new Error("The reranking process returned no scores.");
      }
      return scores;
    },

    close() {
      const stopping = channel;
      channel = undefined;
      loadedKey = undefined;
      stopping?.stop();
    },
  };
}

/** The serving end's connection to the side that sends requests. */
export interface CrossEncoderPort {
  onRequest(listener: (request: CrossEncoderRequest) => void): void;
  respond(response: CrossEncoderResponse): void;
}

/** Serves `crossEncoder` over `port`: one request at a time, in the order they came. */
export function serveCrossEncoder(crossEncoder: CrossEncoder, port: CrossEncoderPort): void {
  async function handle(request: CrossEncoderRequest): Promise<CrossEncoderResponse> {
    try {
      if (request.type === "load") {
        await crossEncoder.load(request.files);
        return { id: request.id };
      }
      return { id: request.id, scores: await crossEncoder.score(request.query, request.texts) };
    } catch (error) {
      return { id: request.id, error: error instanceof Error ? error.message : String(error) };
    }
  }

  let queue = Promise.resolve();
  port.onRequest((request) => {
    queue = queue.then(async () => port.respond(await handle(request)));
  });
}
