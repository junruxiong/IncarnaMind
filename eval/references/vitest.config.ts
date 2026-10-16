import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * `npm run eval:references` (eval/references/README.md): checks the 50
 * references against Crossref itself. It reaches the network, so neither
 * `npm test` nor CI runs it.
 */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  test: {
    include: ["eval/references/**/*.references.ts"],
    environment: "node",
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    // One request a second, as Crossref's public pool asks.
    testTimeout: 30 * 60_000,
  },
});
