/**
 * The evaluation's reranking worker thread: a built-in reranking model on
 * onnxruntime-node, served with the same code and protocol as the desktop
 * app's reranking utility process (src/main/rerankerProcess.ts). Started
 * through the `?nodeWorker` import in ./rerank.
 */
import { parentPort } from "node:worker_threads";
import { type CrossEncoderRequest, serveCrossEncoder } from "../../src/core/reranking/channel";
import { createOnnxCrossEncoder } from "../../src/core/reranking/onnx";

const port = parentPort;
if (!port) throw new Error("The reranking worker must run as a worker thread.");

serveCrossEncoder(createOnnxCrossEncoder(), {
  onRequest: (listener) => port.on("message", (request: CrossEncoderRequest) => listener(request)),
  respond: (response) => port.postMessage(response),
});
