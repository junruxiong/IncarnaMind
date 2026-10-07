/**
 * The embedding utility process: runs the built-in embedding model off the
 * main process (see ./embedder). It takes one request at a time from the main
 * process, in order: { id, type: "load", files } or { id, type: "embed", text },
 * and answers { id, vector } (or { id } for a load) or { id, error }.
 *
 * Started with --fake by the smoke tests' INCARNAMIND_TEST_EMBEDDER=fake flag:
 * it then runs the deterministic fake model instead, with no files.
 */
import type { Embedder } from "../core";
import { createFakeEmbedder } from "../core/embedding/fake";
import { createOnnxEmbedder } from "../core/embedding/onnx";
import type { EmbedderRequest, EmbedderResponse } from "./embedderProtocol";

const fake = process.argv.includes("--fake");
// The fake is slowed down so a smoke test can see a Document's "embedding" status.
const embedder: Embedder = fake ? createFakeEmbedder({ delayMs: 50 }) : createOnnxEmbedder();

const port = process.parentPort;

async function handle(request: EmbedderRequest): Promise<EmbedderResponse> {
  try {
    if (request.type === "load") {
      await embedder.load(request.files);
      return { id: request.id };
    }
    return { id: request.id, vector: await embedder.embed(request.text) };
  } catch (error) {
    return { id: request.id, error: error instanceof Error ? error.message : String(error) };
  }
}

// One at a time, in order (ADR-0009): a query waits for at most the Passage in progress.
let queue = Promise.resolve();
port.on("message", ({ data }: { data: EmbedderRequest }) => {
  queue = queue.then(async () => port.postMessage(await handle(data)));
});
