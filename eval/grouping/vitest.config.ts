import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { nodeWorkers } from "../../vitest.config";

/**
 * `npm run eval:grouping` (eval/grouping/README.md): the check before
 * building the grouping, with the same worker-thread handling as the
 * evaluation. Neither `npm test` (tests/ only) nor `npm run eval`
 * (`*.eval.ts`) runs it.
 */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  plugins: [nodeWorkers()],
  test: {
    include: ["eval/grouping/**/*.check.ts"],
    environment: "node",
    // The run prints its own progress and summary, as it goes.
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    // Downloading the model, embedding every Passage twice, the timings and, with a classifier, a request per Document.
    testTimeout: 4 * 60 * 60_000,
  },
});
