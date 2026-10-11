import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { nodeWorkers } from "../../vitest.config";

/**
 * `npm run eval:agents` (eval/agents/README.md): the agent-integration
 * bake-off, with the same worker-thread handling as the unit tests. Neither
 * `npm test` nor `npm run eval` runs it: their configs include only tests/
 * and `*.eval.ts`.
 */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  plugins: [nodeWorkers()],
  test: {
    include: ["eval/agents/**/*.agents.ts"],
    environment: "node",
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    // Indexing the Library, then up to a few hundred agent runs, one at a time.
    testTimeout: 12 * 60 * 60_000,
  },
});
