import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { HOME_GUARD, nodeWorkers } from "../../vitest.config";
import { evalCache } from "../lib/cache";

/**
 * `npm run eval:dump` (eval/README.md, "Offline scoring"): writes the
 * dumps of the gating, every-format and hard sets, with the same worker-thread
 * handling as the unit tests. Neither `npm test` nor `npm run eval` runs it:
 * their configs include only tests/ and `*.eval.ts`.
 */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  plugins: [nodeWorkers()],
  test: {
    include: ["eval/dumps/**/*.dump.ts"],
    environment: "node",
    setupFiles: [HOME_GUARD],
    env: { INCARNAMIND_EVAL_CACHE: evalCache() },
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    // Fetching the hard tier's library, if it isn't cached, and indexing it.
    testTimeout: 2 * 60 * 60_000,
  },
});
