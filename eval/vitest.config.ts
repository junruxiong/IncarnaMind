import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { nodeWorkers } from "../vitest.config";

/**
 * `npm run eval` (eval/README.md): the retrieval and Citation evaluation,
 * with the same worker-thread handling as the unit tests. Never part of
 * `npm test`, whose config only includes tests/.
 */
export default defineConfig({
  root: fileURLToPath(new URL("..", import.meta.url)),
  plugins: [nodeWorkers()],
  test: {
    include: ["eval/**/*.eval.ts"],
    environment: "node",
    // The run prints its own progress and summary, as it goes.
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    // Downloading the model, embedding every Passage and, with a chat model, rounds of Answers.
    testTimeout: 4 * 60 * 60_000,
  },
});
