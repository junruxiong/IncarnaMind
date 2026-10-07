/**
 * The evaluation's embedding worker thread: the built-in model on
 * onnxruntime-node, served with the same code and protocol as the desktop
 * app's embedding utility process (src/main/embedderProcess.ts). Started
 * through the `?nodeWorker` import in ./embedder.
 */
import { parentPort } from "node:worker_threads";
import { type EmbedderRequest, serveEmbedder } from "../../src/core/embedding/channel";
import { createOnnxEmbedder } from "../../src/core/embedding/onnx";

const port = parentPort;
if (!port) throw new Error("The embedding worker must run as a worker thread.");

serveEmbedder(createOnnxEmbedder(), {
  onRequest: (listener) => port.on("message", (request: EmbedderRequest) => listener(request)),
  respond: (response) => port.postMessage(response),
});
