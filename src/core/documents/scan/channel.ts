/**
 * The text recognition model run somewhere else, talking in messages (#397):
 * both ends of the protocol, as for the embedding model (see
 * ../../embedding/channel). The desktop app runs it in an Electron utility
 * process of its own (src/main/scanReader.ts and src/main/scanReaderProcess.ts).
 *
 * Requests are { id, type: "load", files } or { id, type: "read", picture,
 * page }; the other end answers { id, lines } (or { id } for a load) or
 * { id, error }, taking one request at a time, in order.
 */
import type {
  PagePicture,
  RecognizedLine,
  TextRecognitionFiles,
  TextRecognizer,
} from "../../adapters";

export type TextRecognizerCommand =
  | { type: "load"; files: TextRecognitionFiles }
  | { type: "read"; picture: PagePicture; page: number };

/** `id` pairs a response with its request. */
export type TextRecognizerRequest = TextRecognizerCommand & { id: number };

/** The lines read for "read", nothing more for "load", or what went wrong. */
export interface TextRecognizerResponse {
  id: number;
  lines?: RecognizedLine[];
  error?: string;
}

/** A running process that serves the model, as the side that sends requests sees it. */
export interface TextRecognizerChannel {
  send(request: TextRecognizerRequest): void;
  onResponse(listener: (response: TextRecognizerResponse) => void): void;
  /** It stopped: crashed, ran out of memory, or was stopped. */
  onExit(listener: (code: number | null) => void): void;
  stop(): void;
}

type Pending = { resolve(response: TextRecognizerResponse): void; reject(error: Error): void };

/**
 * The core's `TextRecognizer`, backed by a channel that `start` opens on the
 * first `load`. If it stops, pending requests fail, and the next `load` opens
 * a new one.
 */
export function createChannelTextRecognizer(start: () => TextRecognizerChannel): TextRecognizer {
  let channel: TextRecognizerChannel | undefined;
  let loadedKey: string | undefined;
  let loading: { key: string; done: Promise<void> } | undefined;
  const pending = new Map<number, Pending>();
  let nextId = 1;

  function open(): TextRecognizerChannel {
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
      const error = new Error(`The text recognition process stopped (exit code ${code}).`);
      for (const waiting of pending.values()) waiting.reject(error);
      pending.clear();
    });
    return started;
  }

  function request(command: TextRecognizerCommand): Promise<TextRecognizerResponse> {
    channel ??= open();
    const target = channel;
    const id = nextId++;
    return new Promise<TextRecognizerResponse>((resolve, reject) => {
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

    async read(picture, page) {
      if (!channel || loadedKey === undefined) {
        throw new Error("The text recognition model isn't loaded.");
      }
      const { lines } = await request({ type: "read", picture, page });
      if (!lines) throw new Error("The text recognition process returned no lines.");
      return lines;
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
export interface TextRecognizerPort {
  onRequest(listener: (request: TextRecognizerRequest) => void): void;
  respond(response: TextRecognizerResponse): void;
}

/** Serves `recognizer` over `port`: one request at a time, in the order they came. */
export function serveTextRecognizer(recognizer: TextRecognizer, port: TextRecognizerPort): void {
  async function handle(request: TextRecognizerRequest): Promise<TextRecognizerResponse> {
    try {
      if (request.type === "load") {
        await recognizer.load(request.files);
        return { id: request.id };
      }
      return { id: request.id, lines: await recognizer.read(request.picture, request.page) };
    } catch (error) {
      return { id: request.id, error: error instanceof Error ? error.message : String(error) };
    }
  }

  let queue = Promise.resolve();
  port.onRequest((request) => {
    queue = queue.then(async () => port.respond(await handle(request)));
  });
}
