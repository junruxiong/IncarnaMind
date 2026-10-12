import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * `npm run eval:transcription` (eval/transcription/README.md): the real
 * speech-to-text engine on FLEURS clips. Neither `npm test` nor `npm run
 * eval` runs it: their configs include only tests/ and `*.eval.ts`.
 */
export default defineConfig({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  test: {
    include: ["eval/transcription/**/*.transcription.ts"],
    environment: "node",
    disableConsoleIntercept: true,
    reporters: [["default", { summary: false }]],
    // Loading the model, then each language's stream at the pace it was spoken.
    testTimeout: 60 * 60_000,
  },
});
