import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { HOME_GUARD, nodeWorkers } from "../../vitest.config";
import { evalCache } from "../lib/cache";

/**
 * `npm run eval:hard` (eval/hard/README.md): the hard tier, with the same
 * worker-thread handling as the unit tests. Neither `npm test` nor
 * `npm run eval` runs it: their configs include only tests/ and `*.eval.ts`.
 */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  plugins: [nodeWorkers()],
  test: {
    include: ["eval/hard/**/*.hard.ts"],
    environment: "node",
    setupFiles: [HOME_GUARD],
    env: { INCARNAMIND_EVAL_CACHE: evalCache() },
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    // Fetching the library, embedding tens of thousands of Passages and, with a chat model, an Answer per Question.
    testTimeout: 8 * 60 * 60_000,
  },
});
