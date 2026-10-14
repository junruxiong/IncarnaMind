import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * `npm run eval:word` (eval/word/README.md): the Word corpus check. Neither `npm test` nor
 * `npm run eval` runs it: their configs include only tests/ and `*.eval.ts`.
 */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  test: {
    include: ["eval/word/**/*.word.ts"],
    environment: "node",
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    // A LibreOffice start for each of the 50 files, at up to three minutes.
    testTimeout: 60 * 60_000,
  },
});
